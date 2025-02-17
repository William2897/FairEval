# backend/fair_eval_backend/data_processing/pipeline.py

import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime
import json  # Add this import at the top
from data_processing.ingestion import ingest_csv_to_df
from data_processing.cleaning import clean_data
from data_processing.gender_assignment import engineer_gender
from data_processing.dept_mapping import map_departments
from data_processing.text_preprocessing import preprocess_comments
import gc

def validate_sentiment_data(df):
    """Validate and clean sentiment values to ensure they are 0 or 1 integers"""
    # Convert sentiment to integer, invalid values become NaN
    df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce').astype('Int64')
    
    # Only allow 0 or 1 values
    valid_mask = df['sentiment'].isin([0, 1])
    if not valid_mask.all():
        invalid_count = (~valid_mask).sum()
        print(f"Warning: Found {invalid_count} invalid sentiment values. Setting them to NaN.")
        df.loc[~valid_mask, 'sentiment'] = pd.NA
    
    return df

def run_full_pipeline(csv_path, db_config):
    """Optimized pipeline with batch processing"""
    print("\n--- Stage 1: Data Ingestion ---")
    df = ingest_csv_to_df(csv_path)
    
    print("\n--- Stage 2: Data Processing ---")
    df = clean_data(df)
    df = engineer_gender(df)
    df = map_departments(df)  # This now returns discipline and sub_discipline
    
    # Process comments and save data before term frequency calculation
    print("\nProcessing comments...")
    processed_data = preprocess_comments(df)
    df = processed_data[0]  # Get the processed dataframe
    
    # Save processed data
    processed_csv_path = csv_path.replace('.csv', '_processed.csv')
    df.to_csv(processed_csv_path, index=False)
    print(f"\nProcessed data saved to: {processed_csv_path}")
    
    # Proceed directly to database population
    run_db_population(processed_csv_path, db_config)

def run_db_population(processed_csv_path, db_config):
    """Run only the database population stage using processed CSV"""
    print("\n--- Loading processed data ---")
    df = pd.read_csv(processed_csv_path)
    
    # Validate sentiment data before insertion
    df = validate_sentiment_data(df)
    
    print("\n--- Stage 3: Database Population ---")
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    
    try:
        # 1. Insert professors with discipline information
        prof_data = df[['professor_id', 'first_name', 'last_name', 'gender', 
                        'discipline', 'sub_discipline']].drop_duplicates()

        insert_professors_sql = """
            INSERT INTO api_professor (
                professor_id, first_name, last_name, gender,
                discipline, sub_discipline
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (professor_id) DO NOTHING
        """
        cursor.executemany(insert_professors_sql, 
                          [(row['professor_id'], row['first_name'], row['last_name'],
                            row['gender'], row['discipline'], row['sub_discipline']) 
                           for _, row in prof_data.iterrows()])
        conn.commit()

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

        # 3. Insert ratings - FIXED to use professor_id directly
        batch_size = 5000
        insert_ratings_sql = """
            INSERT INTO api_rating (
                professor_id, avg_rating, flag_status, helpful_rating,
                clarity_rating, difficulty_rating, is_online,
                is_for_credit, created_at
            )
            SELECT 
                r.prof_id,
                r.avg_rating,
                r.flag_status,
                r.helpful_rating,
                r.clarity_rating,
                r.difficulty_rating,
                r.is_online,
                r.is_for_credit,
                r.created_at
            FROM (
                SELECT * FROM unnest(
                    %s::text[],       -- professor_id
                    %s::float[],      -- avg_rating
                    %s::text[],       -- flag_status
                    %s::float[],      -- helpful_rating
                    %s::float[],      -- clarity_rating
                    %s::float[],      -- difficulty_rating
                    %s::boolean[],    -- is_online
                    %s::boolean[],    -- is_for_credit
                    %s::timestamp[]   -- created_at
                ) AS r(
                    prof_id, avg_rating, flag_status, helpful_rating,
                    clarity_rating, difficulty_rating, is_online,
                    is_for_credit, created_at
                )
            ) r
            WHERE EXISTS (SELECT 1 FROM api_professor p WHERE p.professor_id = r.prof_id)
        """

        # Modify the sentiment insertion SQL to properly handle JSON arrays
        insert_sentiment_sql = """
            INSERT INTO api_sentiment (
                professor_id, comment, processed_comment,
                sentiment, positive_terms_lexicon, negative_terms_lexicon,
                positive_terms_vader, negative_terms_vader,
                created_at
            )
            VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s)
        """
        
        current_time = datetime.now()
        rating_chunks = [df[i:i + batch_size] for i in range(0, len(df), batch_size)]
        
        successful_inserts = 0
        failed_inserts = 0
        
        for chunk in rating_chunks:
            try:
                # Prepare data arrays for ratings
                prof_ids = chunk['professor_id'].tolist()
                avg_ratings = chunk['avg_rating'].tolist()
                flag_statuses = chunk['flag_status'].tolist()
                helpful_ratings = chunk['helpful_rating'].tolist()
                clarity_ratings = chunk['clarity_rating'].tolist()
                difficulty_ratings = chunk['difficulty_rating'].tolist()
                is_onlines = chunk['is_online'].tolist()
                is_for_credits = chunk['is_for_credit'].tolist()
                timestamps = [current_time] * len(chunk)
                
                # Execute ratings insertion
                cursor.execute(insert_ratings_sql, (
                    prof_ids, avg_ratings, flag_statuses, helpful_ratings,
                    clarity_ratings, difficulty_ratings, is_onlines,
                    is_for_credits, timestamps
                ))

                # Convert list columns to proper JSON strings
                def convert_list_to_json(x):
                    try:
                        if isinstance(x, str):
                            # Handle string representation of lists
                            x = eval(x) if x else []
                        return json.dumps(list(x)) if x else '[]'
                    except:
                        return '[]'

                sentiment_data = [
                    (
                        row['professor_id'],
                        row['rating_comment'],
                        row['processed_comment'],
                        row['sentiment'],
                        convert_list_to_json(row['positive_terms_lexicon']),
                        convert_list_to_json(row['negative_terms_lexicon']),
                        convert_list_to_json(row['positive_terms_vader']),
                        convert_list_to_json(row['negative_terms_vader']),
                        current_time
                    )
                    for _, row in chunk.iterrows()
                ]

                # Execute sentiment insertion
                cursor.executemany(insert_sentiment_sql, sentiment_data)
                conn.commit()
                successful_inserts += len(chunk)
                
            except Exception as e:
                conn.rollback()
                failed_inserts += len(chunk)
                print(f"Error processing chunk: {str(e)}")
                continue
            
            gc.collect()

        print(f"\nInsertion complete:")
        print(f"Successful inserts: {successful_inserts}")
        print(f"Failed inserts: {failed_inserts}")

    except Exception as e:
        conn.rollback()
        print(f"Error during database population: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()
        print("Pipeline complete! Data saved to normalized tables.")