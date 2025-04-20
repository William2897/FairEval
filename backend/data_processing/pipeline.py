# backend/fair_eval_backend/data_processing/pipeline.py

import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import torch
from tqdm import tqdm # For progress tracking
import gc # Garbage collection

# Import functions from other modules
from data_processing.ingestion import ingest_csv_to_df
from data_processing.cleaning import clean_data
from data_processing.gender_assignment import engineer_gender
from data_processing.dept_mapping import map_departments
from data_processing.text_preprocessing import preprocess_comments

# Import ML components
from machine_learning.gender_bias_explainer import GenderBiasExplainer
from machine_learning.ml_model_dev.lstm import CustomSentimentLSTM

# Import Django settings to get BASE_DIR - Ensure Django is configured
try:
    from django.conf import settings
except ImportError:
    print("Warning: Django settings could not be imported. Model paths might need adjustment.")
    # Basic fallback for BASE_DIR if Django settings aren't available
    settings = type('obj', (object,), {'BASE_DIR': os.path.dirname(os.path.dirname(os.path.abspath(__file__)))})()


# --- Utility Functions ---

def validate_sentiment_data(df):
    """Validate and clean sentiment values to ensure they are 0 or 1 integers or NA."""
    # Convert sentiment to numeric, coercing errors to NaN, then to nullable Int64
    df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce').astype('Int64')

    # Only allow 0 or 1 values, otherwise set to NA
    valid_mask = df['sentiment'].isin([0, 1])
    invalid_count = (~valid_mask & df['sentiment'].notna()).sum() # Count invalid non-null values
    if invalid_count > 0:
        print(f"Warning: Found {invalid_count} invalid sentiment values (not 0 or 1). Setting them to NA.")
        df.loc[~valid_mask, 'sentiment'] = pd.NA # Use pandas NA for nullable integer

    return df

def load_bias_explainer():
    """Loads the LSTM model, vocab, and initializes the GenderBiasExplainer."""
    try:
        model_path = os.path.join(settings.BASE_DIR, 'machine_learning/ml_models_trained/lstm_sentiment.pt')
        vocab_path = os.path.join(settings.BASE_DIR, 'machine_learning/ml_models_trained/vocab.json')

        if not os.path.exists(model_path) or not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Model ({model_path}) or vocab ({vocab_path}) file not found.")

        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Ensure model class definition matches the saved state dict
        model = CustomSentimentLSTM(
            vocab_size=len(vocab), embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.5
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # Set model to evaluation mode
        print("LSTM Model loaded successfully.")

        explainer = GenderBiasExplainer(model, vocab) # Assumes default max_len=100
        print("GenderBiasExplainer initialized.")
        return explainer

    except Exception as e:
        print(f"ERROR loading ML model/explainer: {e}")
        import traceback
        print(traceback.format_exc())
        return None

# --- Core Pipeline Functions ---

def run_initial_processing(csv_path):
    """
    Runs only the initial ingestion and processing steps (Stages 1 & 2).
    Saves the result to '_processed.csv'.
    Returns the path to the processed CSV.
    """
    print("\n--- Running Initial Processing Pipeline ---")
    print("\n--- Stage 1: Data Ingestion ---")
    df = ingest_csv_to_df(csv_path)
    if df is None or df.empty:
        print("ERROR: Data ingestion failed or returned empty DataFrame.")
        return None
    print(f"[DEBUG ROWS] After ingestion: {len(df)} rows")

    print("\n--- Stage 2: Data Processing ---")
    df = clean_data(df)
    print(f"[DEBUG ROWS] After cleaning: {len(df)} rows")
    df = engineer_gender(df)
    print(f"[DEBUG ROWS] After gender assignment: {len(df)} rows")
    df = map_departments(df)
    print(f"[DEBUG ROWS] After department mapping: {len(df)} rows")

    print("\nProcessing comments...")
    try:
        # Assuming preprocess_comments returns a tuple/list where the first item is the DataFrame
        result = preprocess_comments(df)
        df = result[0] if isinstance(result, (list, tuple)) else result
        print(f"[DEBUG ROWS] After comment processing: {len(df)} rows")
    except Exception as e:
        print(f"ERROR during comment processing: {e}")
        return None # Cannot proceed without processed comments

    # Save processed data (without bias info)
    processed_csv_path = csv_path.replace('.csv', '_processed.csv')
    try:
        df.to_csv(processed_csv_path, index=False)
        print(f"\nInitial processed data saved to: {processed_csv_path}")
    except Exception as e:
         print(f"ERROR saving processed CSV: {e}")
         return None # Fail if we cannot save the intermediate result

    return processed_csv_path

def run_bias_analysis_only(processed_csv_path, output_csv_path, chunk_size=5000):
    """
    Loads a processed CSV in chunks, runs BATCHED bias analysis,
    and saves the results (with bias info) to a NEW CSV file.
    Returns the path to the final CSV with bias info, or None on failure.
    """
    print("\n--- Starting Bias Analysis Only Workflow ---")
    print(f"Input processed CSV: {processed_csv_path}")
    print(f"Output CSV (with bias): {output_csv_path}")

    # --- Load Explainer Once ---
    explainer = load_bias_explainer()
    if not explainer:
        print("ERROR: Cannot perform bias analysis without explainer.")
        return None

    # --- Process CSV in Chunks and Write Output ---
    output_header_written = False
    total_rows_analyzed = 0

    # Define dtypes for loading chunks
    dtypes = {
        'professor_id': str, 'sentiment': 'Int64', 'confidence': float,
        'helpful_rating': float, 'clarity_rating': float, 'difficulty_rating': float,
        'avg_rating': float,
        # Bias columns don't exist yet in the input
    }

    try:
        print(f"Reading input CSV in chunks of size {chunk_size}...")
        # Use context manager for file writing
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as outfile:
            for i, chunk_df in enumerate(pd.read_csv(
                    processed_csv_path,
                    chunksize=chunk_size,
                    dtype=dtypes,
                    keep_default_na=True,
                    na_values=['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null']
                )):

                print(f"\nProcessing Chunk {i+1} for Bias Analysis...")
                if chunk_df.empty:
                    print("  Skipping empty chunk.")
                    continue

                # --- Fill NaNs needed for analysis ---
                str_cols_fill_empty = ['gender', 'discipline', 'processed_comment']
                for col in str_cols_fill_empty:
                    if col in chunk_df.columns: chunk_df[col] = chunk_df[col].fillna('')

                # --- Perform Batched Bias Analysis ---
                comments_batch = chunk_df['processed_comment'].tolist()
                genders_batch = chunk_df['gender'].tolist()
                disciplines_batch = chunk_df['discipline'].tolist()

                valid_indices = [idx for idx, g in enumerate(genders_batch) if g in ['Male', 'Female'] and pd.notna(comments_batch[idx]) and comments_batch[idx].strip()]
                valid_comments = [comments_batch[idx] for idx in valid_indices]
                valid_genders = [genders_batch[idx] for idx in valid_indices]
                valid_disciplines = [disciplines_batch[idx] for idx in valid_indices]

                bias_results_list = []
                if valid_comments:
                    try:
                        bias_results_list = explainer.explain_batch(valid_comments, valid_genders, valid_disciplines)
                    except Exception as bias_err:
                        print(f"  ERROR during batch bias analysis for chunk {i+1}: {bias_err}")
                        bias_results_list = [{'gender_bias': {'interpretation': ['Batch Analysis Error'], 'stereotype_bias_score': None, 'category_attention_pct': {}}} for _ in valid_comments]

                # --- Map results back ---
                chunk_df['bias_tag'] = None
                chunk_df['bias_interpretation'] = None
                chunk_df['stereotype_bias_score'] = pd.NA # Use pd.NA for nullable floats
                chunk_df['objective_focus_percentage'] = pd.NA

                res_idx = 0
                for df_idx in valid_indices:
                     if res_idx < len(bias_results_list):
                         explanation = bias_results_list[res_idx]
                         gb = explanation.get('gender_bias', {})
                         interp_list = gb.get('interpretation', [])
                         st_score = gb.get('stereotype_bias_score')
                         obj_pct = gb.get('category_attention_pct', {}).get('objective_pedagogical', 0)
                         tag = explainer._determine_bias_tag(interp_list, st_score, obj_pct)

                         # Use .iloc for potentially more reliable index-based assignment
                         chunk_df.iloc[df_idx, chunk_df.columns.get_loc('bias_tag')] = tag
                         chunk_df.iloc[df_idx, chunk_df.columns.get_loc('bias_interpretation')] = interp_list[0] if interp_list else None
                         chunk_df.iloc[df_idx, chunk_df.columns.get_loc('stereotype_bias_score')] = float(st_score) if pd.notna(st_score) else pd.NA
                         chunk_df.iloc[df_idx, chunk_df.columns.get_loc('objective_focus_percentage')] = float(obj_pct) if pd.notna(obj_pct) else pd.NA
                     res_idx += 1
                print(f"  Bias analysis applied to chunk {i+1}.")                # --- Write chunk (with bias info) to output CSV ---
                if not output_header_written:
                    chunk_df.to_csv(outfile, index=False, header=True, lineterminator='\n')
                    output_header_written = True
                    # print(f"  Header written to {output_csv_path}") # Less verbose
                else:
                    chunk_df.to_csv(outfile, index=False, header=False, lineterminator='\n')
                # print(f"  Chunk {i+1} appended to {output_csv_path}") # Less verbose

                total_rows_analyzed += len(chunk_df)
                gc.collect() # Clean up memory

        print(f"\n--- Bias Analysis Complete ---")
        print(f"Total rows processed and saved: {total_rows_analyzed}")
        print(f"Final data with bias info saved to: {output_csv_path}")
        return output_csv_path # Return path to the new file

    except FileNotFoundError:
        print(f"ERROR: Input processed CSV file not found at {processed_csv_path}")
        return None
    except Exception as e:
        print(f"CRITICAL ERROR during bias analysis: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None


def run_db_population(csv_with_bias_path, db_config):
    """
    Populates the database using a CSV file that already contains bias information.
    """
    print("\n--- Starting Database Population Only Workflow ---")
    print(f"Input CSV (with bias info): {csv_with_bias_path}")

    # Define dtypes for loading, including new bias columns
    dtypes = {
        'professor_id': str, 'sentiment': 'Int64', 'confidence': float,
        'helpful_rating': float, 'clarity_rating': float, 'difficulty_rating': float,
        'avg_rating': float, 'stereotype_bias_score': float, # Bias field
        'objective_focus_percentage': float # Bias field
    }
    try:
        df = pd.read_csv(
            csv_with_bias_path,
            dtype=dtypes,
            keep_default_na=True,
            na_values=['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null']
        )
        print(f"Loaded {len(df)} records from {csv_with_bias_path}")
    except FileNotFoundError:
        print(f"ERROR: CSV file with bias info not found at {csv_with_bias_path}")
        return None
    except Exception as e:
        print(f"ERROR loading CSV {csv_with_bias_path}: {e}")
        return None

    # --- Fill NaNs AFTER loading ---
    float_cols_to_fill = ['avg_rating', 'helpful_rating', 'clarity_rating', 'difficulty_rating', 'confidence', 'stereotype_bias_score', 'objective_focus_percentage']
    for col in float_cols_to_fill:
        if col in df.columns:
            if df[col].dtype == 'object': df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0.0)
    bool_cols = ['is_online', 'is_for_credit']
    for col in bool_cols:
        if col in df.columns: df[col] = df[col].fillna(False).astype(bool)
    str_cols_fill_empty = ['gender', 'discipline', 'sub_discipline', 'flag_status', 'rating_class', 'comment_topic', 'bias_tag', 'bias_interpretation', 'rating_comment', 'processed_comment']
    for col in str_cols_fill_empty:
        if col in df.columns: df[col] = df[col].fillna('')
    json_cols = ['positive_terms_lexicon', 'negative_terms_lexicon', 'positive_terms_vader', 'negative_terms_vader']
    for col in json_cols:
        if col in df.columns: df[col] = df[col].astype(object).fillna('[]')

    original_record_count = len(df)
    print(f"Total records to process for DB: {original_record_count}")

    # Validate sentiment data
    df = validate_sentiment_data(df)

    print("\n--- Database Population Starting ---")
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        print("Database connection established.")

        # --- Prepare DB INSERT statements (including bias) ---
        # (Professor and User/Role SQL remains the same)
        insert_professors_sql = """
            INSERT INTO api_professor (professor_id, first_name, last_name, gender, discipline, sub_discipline)
            VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (professor_id) DO UPDATE SET
            first_name = EXCLUDED.first_name, last_name = EXCLUDED.last_name, gender = EXCLUDED.gender,
            discipline = EXCLUDED.discipline, sub_discipline = EXCLUDED.sub_discipline;"""
        insert_ratings_sql = """
            INSERT INTO api_rating (professor_id, avg_rating, flag_status, helpful_rating, clarity_rating, difficulty_rating, is_online, is_for_credit, created_at)
            SELECT r.* FROM unnest(%s::text[], %s::float[], %s::text[], %s::float[], %s::float[], %s::float[], %s::boolean[], %s::boolean[], %s::timestamp[])
            AS r(prof_id, avg_rating, flag_status, helpful_rating, clarity_rating, difficulty_rating, is_online, is_for_credit, created_at)
            WHERE EXISTS (SELECT 1 FROM api_professor p WHERE p.professor_id = r.prof_id)
            ON CONFLICT DO NOTHING;"""
        insert_sentiment_sql = """
            INSERT INTO api_sentiment (professor_id, comment, processed_comment, sentiment, positive_terms_lexicon, negative_terms_lexicon, positive_terms_vader, negative_terms_vader, created_at, bias_tag, bias_interpretation, stereotype_bias_score, objective_focus_percentage)
            VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING;"""
        
        # 2. Create users for professors
        insert_users_sql = """
            INSERT INTO auth_user (
                username, first_name, last_name, email,
                password, is_staff, is_active, date_joined,
                is_superuser
            )
            SELECT 
                p.professor_id,
                p.first_name,
                p.last_name,
                LOWER(CONCAT(p.first_name, '.', p.last_name, '@institution.edu')),
                'pbkdf2_sha256$600000$default_hash',
                TRUE,
                TRUE,
                NOW(),
                FALSE
            FROM api_professor p
            LEFT JOIN auth_user u ON u.username = p.professor_id
            WHERE u.id IS NULL
            RETURNING id, username
        """
        cursor.execute(insert_users_sql)
        user_mappings = cursor.fetchall()
        conn.commit()

        # 2.1 Assign roles (20% Admin, 80% Academic)
        admin_count = int(len(user_mappings) * 0.2)
        insert_roles_sql = """
            INSERT INTO api_userrole (user_id, role, discipline)
            SELECT 
                u.id,
                CASE 
                    WHEN ROW_NUMBER() OVER (ORDER BY RANDOM()) <= %s THEN 'ADMIN'
                    ELSE 'ACADEMIC'
                END,
                p.discipline
            FROM auth_user u
            JOIN api_professor p ON p.professor_id = u.username
            LEFT JOIN api_userrole ur ON ur.user_id = u.id
            WHERE ur.id IS NULL
        """
        cursor.execute(insert_roles_sql, (admin_count,))
        conn.commit()

        # --- Insert Data Chunk by Chunk ---
        batch_size = 5000 # Define DB batch size
        df_chunks = [df[i:i + batch_size] for i in range(0, len(df), batch_size)]
        successful_inserts = 0
        failed_inserts = 0

        print(f"Starting DB insertion in {len(df_chunks)} chunks...")
        for i, chunk in enumerate(df_chunks):
            if chunk.empty: continue

            try:
                current_time_db = datetime.now()

                # 1. Upsert Professors
                prof_data_chunk = chunk[['professor_id', 'first_name', 'last_name', 'gender', 'discipline', 'sub_discipline']].drop_duplicates().dropna(subset=['professor_id'])
                prof_tuples_chunk = [
                     (str(row['professor_id']), str(row['first_name']), str(row['last_name']),
                      str(row['gender']) if pd.notna(row['gender']) else None,
                      str(row['discipline']) if pd.notna(row['discipline']) else None,
                      str(row['sub_discipline']) if pd.notna(row['sub_discipline']) else None)
                    for _, row in prof_data_chunk.iterrows()
                ]
                if prof_tuples_chunk: cursor.executemany(insert_professors_sql, prof_tuples_chunk)

                # 2. Insert Ratings
                prof_ids_r = chunk['professor_id'].astype(str).tolist()
                avg_ratings = chunk['avg_rating'].astype(float).tolist()
                # ... (prepare other rating lists as before) ...
                flag_statuses = chunk['flag_status'].astype(str).tolist()
                helpful_ratings = chunk['helpful_rating'].astype(float).tolist()
                clarity_ratings = chunk['clarity_rating'].astype(float).tolist()
                difficulty_ratings = chunk['difficulty_rating'].astype(float).tolist()
                is_onlines = chunk['is_online'].astype(bool).tolist()
                is_for_credits = chunk['is_for_credit'].astype(bool).tolist()
                timestamps_r = [current_time_db] * len(chunk)
                cursor.execute(insert_ratings_sql, (
                    prof_ids_r, avg_ratings, flag_statuses, helpful_ratings,
                    clarity_ratings, difficulty_ratings, is_onlines,
                    is_for_credits, timestamps_r
                ))

                # 3. Insert Sentiments (with bias)
                def safe_convert_list_to_json(x):
                    try:
                        if pd.isna(x): return '[]'
                        if isinstance(x, str):
                            try: return json.dumps(list(eval(x))) if x.strip().startswith('[') else '[]'
                            except: return '[]'
                        return json.dumps(list(x))
                    except: return '[]'

                sentiment_data = []
                for _, row in chunk.iterrows():
                    sentiment_val = int(row['sentiment']) if pd.notna(row['sentiment']) else None
                    stereotype_score_val = float(row['stereotype_bias_score']) if pd.notna(row['stereotype_bias_score']) else None
                    objective_pct_val = float(row['objective_focus_percentage']) if pd.notna(row['objective_focus_percentage']) else None
                    sentiment_data.append((
                        str(row['professor_id']), str(row['rating_comment']), str(row['processed_comment']),
                        sentiment_val,
                        safe_convert_list_to_json(row.get('positive_terms_lexicon')),
                        safe_convert_list_to_json(row.get('negative_terms_lexicon')),
                        safe_convert_list_to_json(row.get('positive_terms_vader')),
                        safe_convert_list_to_json(row.get('negative_terms_vader')),
                        current_time_db,
                        str(row['bias_tag']) if pd.notna(row['bias_tag']) else None, # Bias field
                        str(row['bias_interpretation']) if pd.notna(row['bias_interpretation']) else None, # Bias field
                        stereotype_score_val, # Bias field
                        objective_pct_val    # Bias field
                    ))
                cursor.executemany(insert_sentiment_sql, sentiment_data)

                # Commit transaction for the chunk
                conn.commit()
                successful_inserts += len(chunk)
                print(f"  Chunk {i+1}/{len(df_chunks)} DB insertion committed ({len(chunk)} records). Total successful: {successful_inserts}")

            except Exception as db_err:
                conn.rollback()
                failed_inserts += len(chunk)
                print(f"  ERROR during DB insertion for chunk {i+1}: {db_err}. Rolled back chunk.")
                import traceback
                print(traceback.format_exc())
                continue

            gc.collect()

        print(f"\n--- Database Population Complete ---")
        print(f"Original records read: {original_record_count}")
        print(f"Successful DB inserts: {successful_inserts}")
        print(f"Failed DB inserts (entire chunks): {failed_inserts}")

        return { # Return status
            "original_count": original_record_count,
            "successful_inserts": successful_inserts,
            "failed_inserts": failed_inserts
        }

    except Exception as e:
        if conn: conn.rollback()
        print(f"CRITICAL ERROR during database population: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None # Indicate failure
    finally:
        if cursor: cursor.close()
        if conn: conn.close()
        print("Database connection closed.")


# --- Original Full Pipeline Function (Optional - for reference or initial processing) ---
# def run_full_pipeline(csv_path, db_config, run_sentiment=True):
#     """Runs initial processing (Stages 1 & 2) and optionally populates DB WITHOUT bias info."""
#     processed_csv = run_initial_processing(csv_path)
#     if not processed_csv:
#         print("Initial processing failed. Aborting.")
#         return
#
#     if run_sentiment:
#         # Load the processed data again to add placeholder columns if needed
#         df = pd.read_csv(processed_csv)
#         df['bias_tag'] = None
#         df['bias_interpretation'] = None
#         df['stereotype_bias_score'] = None
#         df['objective_focus_percentage'] = None
#         df.to_csv(processed_csv, index=False) # Overwrite with placeholders
#
#         print("Proceeding to populate DB with processed data (no bias info)...")
#         run_db_population(processed_csv, db_config)
#     else:
#         print("\nSkipping database population as requested.")
#
#     print("Initial processing pipeline complete.")