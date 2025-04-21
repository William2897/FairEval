# backend/api/tasks.py

from celery import shared_task
from celery.exceptions import MaxRetriesExceededError
from django.db import transaction
from django.db.models import F
import gc
import os
import json
import pandas as pd
import torch
import numpy as np
from typing import List, Dict, Any
from django.conf import settings
from tqdm import tqdm

# Import models and utils (ensure paths are correct)
try:
    from .models import Rating, Sentiment, Professor
except ImportError:
    print("Warning: Could not import models/utils directly in tasks.py.")
    Rating, Sentiment, Professor = None, None, None

# Import text processing functions used by analyze_comments_sentiment
from data_processing.text_preprocessing import extract_opinion_terms, extract_vader_terms, clean_text

# --- IMPORT THE *NEW* PIPELINE FUNCTIONS ---
from data_processing.pipeline import (
    run_initial_processing,
    run_bias_analysis_only,
    run_db_population
)
# --- END IMPORT ---

# Import LSTM model (needed for predict_sentiment_with_lstm)
from machine_learning.ml_model_dev.lstm import CustomSentimentLSTM

# --- Celery Configuration ---
BATCH_SIZE = getattr(settings, 'CELERY_BATCH_SIZE', 1000) # Used by other tasks
MAX_RETRIES = getattr(settings, 'CELERY_MAX_RETRIES', 3)
RETRY_DELAY = getattr(settings, 'CELERY_RETRY_DELAY', 60) # seconds
PIPELINE_CHUNK_SIZE = getattr(settings, 'PIPELINE_CHUNK_SIZE', 5000) # Chunk size for pipeline functions

# ==============================================================
# === Other Celery Tasks (Unchanged from original file) ========
# ==============================================================

@shared_task(
    bind=True,
    max_retries=MAX_RETRIES,
    rate_limit='100/m',
    priority=10
)
def process_rating_upload(self, ratings_data: List[Dict]):
    """Process uploaded ratings data in batches"""
    processed = 0
    affected_professors = set()
    errors = []

    # Ensure models are available
    global Rating, Professor
    if Rating is None: from .models import Rating
    if Professor is None: from .models import Professor

    try:
        for i in range(0, len(ratings_data), BATCH_SIZE):
            batch = ratings_data[i:i + BATCH_SIZE]
            with transaction.atomic():
                for rating_data in batch:
                    try:
                        # Make sure professor_id exists before creating rating
                        prof_id = rating_data.get('professor_id')
                        if prof_id and Professor.objects.filter(professor_id=prof_id).exists():
                            rating = Rating.objects.create(**rating_data)
                            processed += 1
                            affected_professors.add(prof_id)
                            # Queue professor metrics update if needed
                            # chain(...).delay() # Example
                        else:
                            errors.append(f"Professor ID {prof_id} not found for rating.")
                    except Exception as e:
                        errors.append(f"Error processing rating: {str(e)}")

            gc.collect()

        # Notify connected clients (if using Channels)
        # ... (notification logic) ...

    except Exception as e:
        if self.request.retries < self.max_retries:
            self.retry(exc=e, countdown=RETRY_DELAY)
        else:
            # Log the final error before raising MaxRetriesExceededError
            print(f"FINAL ERROR in process_rating_upload after {self.max_retries} retries: {e}")
            raise MaxRetriesExceededError(f"Failed to process ratings after {self.max_retries} retries: {str(e)}") from e

    return {
        'processed': processed,
        'errors': errors,
        'affected_professors': list(affected_professors)
    }

@shared_task(
    bind=True,
    max_retries=MAX_RETRIES,
    rate_limit='50/m',
    priority=5
)
def analyze_comments_sentiment(self, professor_id: str):
    """Process comments for existing Sentiment records to extract lexicon/vader terms (Not LSTM/Bias)"""
    # This task remains focused on lexicon/vader term extraction if needed separately.
    # LSTM/Bias analysis is handled by the main pipeline task.
    global Professor, Sentiment
    if Sentiment is None: from .models import Sentiment
    if Professor is None: from .models import Professor

    try:
        professor = Professor.objects.get(professor_id=professor_id)
        # Process sentiments where lexicon/vader terms might be missing
        sentiments_to_process = Sentiment.objects.filter(
            professor_id=professor_id,
            comment__isnull=False # Must have comment
            # Add condition if needed, e.g., positive_terms_lexicon__isnull=True
        )

        if not sentiments_to_process.exists():
            print(f"No comments requiring lexicon/vader analysis for professor {professor_id}.")
            return {'status': 'skipped', 'message': 'No comments needed analysis.'}

        processed_count = 0
        errors = []
        for i in range(0, sentiments_to_process.count(), BATCH_SIZE):
            batch = list(sentiments_to_process[i:i + BATCH_SIZE]) # Fetch batch
            updates = []
            with transaction.atomic(): # Process batch transactionally
                for sentiment in batch:
                    try:
                        if not sentiment.processed_comment: # Ensure processed comment exists
                            processed_comment = clean_text(sentiment.comment or '')
                            sentiment.processed_comment = processed_comment
                        else:
                            processed_comment = sentiment.processed_comment

                        processed_tokens = processed_comment.split()

                        pos_terms_lexicon, neg_terms_lexicon = extract_opinion_terms(processed_tokens)
                        pos_terms_vader, neg_terms_vader = extract_vader_terms(processed_comment)

                        # Update fields only if they are currently null or empty
                        if not sentiment.positive_terms_lexicon: sentiment.positive_terms_lexicon = pos_terms_lexicon
                        if not sentiment.negative_terms_lexicon: sentiment.negative_terms_lexicon = neg_terms_lexicon
                        if not sentiment.positive_terms_vader: sentiment.positive_terms_vader = pos_terms_vader
                        if not sentiment.negative_terms_vader: sentiment.negative_terms_vader = neg_terms_vader

                        updates.append(sentiment)
                        processed_count += 1
                    except Exception as e:
                        errors.append(f"Error analyzing sentiment ID {sentiment.id}: {e}")

                # Bulk update the processed batch
                if updates:
                    Sentiment.objects.bulk_update(updates, [
                        'processed_comment', 'positive_terms_lexicon', 'negative_terms_lexicon',
                        'positive_terms_vader', 'negative_terms_vader'
                    ])

            gc.collect()

        # Optionally queue recommendations update
        # generate_recommendations.delay(professor_id)

        return {
            'status': 'success' if not errors else 'partial_error',
            'processed_count': processed_count,
            'errors': errors
        }

    except Professor.DoesNotExist:
         print(f"Professor {professor_id} not found for comment analysis.")
         # Do not retry if professor doesn't exist
         return {'status': 'error', 'message': f'Professor {professor_id} not found.'}
    except Exception as e:
        if self.request.retries < self.max_retries:
            self.retry(exc=e, countdown=RETRY_DELAY)
        else:
            print(f"FINAL ERROR in analyze_comments_sentiment for {professor_id} after {self.max_retries} retries: {e}")
            raise MaxRetriesExceededError(f"Failed to analyze comments for {professor_id} after {self.max_retries} retries: {str(e)}") from e


@shared_task(
    bind=True,
    max_retries=MAX_RETRIES,
    rate_limit='10/m',
    priority=3
)
def calculate_discipline_analytics(self, discipline: str):
    """Placeholder for potentially updating discipline-wide aggregates"""
    # This task's implementation depends on what 'discipline analytics' means.
    # It might involve re-calculating average ratings, gender distributions etc.
    # for a specific discipline and caching the results.
    global Professor
    if Professor is None: from .models import Professor
    print(f"Simulating calculation of analytics for discipline: {discipline}")
    try:
        professors = Professor.objects.filter(discipline=discipline)
        professor_count = professors.count()
        if professor_count == 0:
            print(f"No professors found for discipline {discipline}.")
            return {'status': 'skipped', 'message': 'No professors in discipline.'}

        # Example: Recalculate average rating for the discipline
        # discipline_avg = Rating.objects.filter(professor__discipline=discipline).aggregate(Avg('avg_rating'))
        # Cache.set(f'discipline_avg_{discipline}', discipline_avg['avg_rating__avg'])

        print(f"Discipline analytics calculation for {discipline} complete (Placeholder).")
        return {'status': 'success', 'discipline': discipline, 'professors_found': professor_count}

    except Exception as e:
         if self.request.retries < self.max_retries:
             self.retry(exc=e, countdown=RETRY_DELAY)
         else:
             print(f"FINAL ERROR in calculate_discipline_analytics for {discipline} after {self.max_retries} retries: {e}")
             raise MaxRetriesExceededError(f"Failed to process discipline {discipline} after {self.max_retries} retries: {str(e)}") from e

# --- UNCHANGED: update_discipline_analytics just queues calculate_discipline_analytics ---
@shared_task(
    bind=True,
    max_retries=MAX_RETRIES,
    priority=3
)
def update_discipline_analytics(self, discipline: str):
    """Update analytics for a discipline"""
    return calculate_discipline_analytics.delay(discipline)


# ==============================================================
# === UPDATED: Main Pipeline Task ==============================
# ==============================================================

@shared_task(
    bind=True,
    max_retries=MAX_RETRIES, # Allow retries for the whole process
    default_retry_delay=RETRY_DELAY, # Use default delay
    priority=9 # High priority as it handles uploads
)
def process_evaluation_data_task(self, file_path: str, db_config: Dict[str, Any]):
    """
    Processes an uploaded evaluation data CSV through multiple stages:
    1. Initial Preprocessing (Cleaning, Gender, Dept Map, Text) -> _processed.csv
    2. LSTM Sentiment Prediction -> Overwrites _processed.csv with sentiment/confidence
    3. Bias Analysis -> _with_bias.csv
    4. Database Population -> Loads _with_bias.csv into DB
    Cleans up intermediate files.
    """
    print(f"[TASK START] Processing file: {file_path}")
    processed_csv = None
    csv_with_bias = None
    original_row_count = 0 # Initialize

    try:
        # --- Validate input file ---
        if not os.path.exists(file_path):
            print(f"ERROR: Input file not found at {file_path}")
            # No retry needed if file is missing
            return {"status": "error", "message": f"Input file not found: {file_path}"}

        # Get original row count for final report
        try:
            original_df = pd.read_csv(file_path, usecols=[0]) # Read only first col for count
            original_row_count = len(original_df)
            del original_df # Free memory
            gc.collect()
            print(f"Original file has {original_row_count} rows.")
        except Exception as read_err:
            print(f"Warning: Could not read original file {file_path} to get count: {read_err}")
            # Continue processing if possible

        # --- Stage 1 & 2: Initial Processing ---
        print("\n--- Running Initial Processing ---")
        processed_csv = run_initial_processing(file_path)
        if not processed_csv or not os.path.exists(processed_csv):
            raise ValueError("Initial processing failed or did not produce an output CSV.")
        print(f"Initial processing complete. Output: {processed_csv}")

        # --- Stage 2.5: Sentiment Prediction ---
        print("\n--- Running Sentiment Prediction ---")
        # Load the intermediate CSV generated by initial processing
        df_processed = pd.read_csv(processed_csv)
        model_path = os.path.join(settings.BASE_DIR, 'machine_learning/ml_models_trained/lstm_sentiment.pt')
        vocab_path = os.path.join(settings.BASE_DIR, 'machine_learning/ml_models_trained/vocab.json')
        df_with_sentiment = predict_sentiment_with_lstm(df_processed, model_path, vocab_path)
        # Overwrite the processed CSV with the sentiment data
        df_with_sentiment.to_csv(processed_csv, index=False)
        print(f"Sentiment prediction complete. Updated: {processed_csv}")
        del df_processed, df_with_sentiment # Free memory
        gc.collect()

        # --- Stage 2.75: Bias Analysis ---
        print("\n--- Running Bias Analysis ---")
        csv_with_bias = processed_csv.replace('_processed.csv', '_with_bias.csv')
        analyzed_csv_path = run_bias_analysis_only(processed_csv, csv_with_bias, chunk_size=PIPELINE_CHUNK_SIZE)
        if not analyzed_csv_path or not os.path.exists(analyzed_csv_path):
             raise ValueError("Bias analysis failed or did not produce an output CSV.")
        print(f"Bias analysis complete. Output: {analyzed_csv_path}")

        # --- Stage 3: Database Population ---
        print("\n--- Running Database Population ---")
        db_results = run_db_population(analyzed_csv_path, db_config)
        if not db_results:
            raise ValueError("Database population step failed.")
        print(f"Database population finished. Results: {db_results}")

        # --- Success Reporting ---
        processed_count = db_results.get("successful_inserts", 0)
        failed_inserts = db_results.get("failed_inserts", 0)
        final_status = "success" if failed_inserts == 0 else "partial_success"

        result_data = {
            "status": final_status,
            "message": f"Successfully processed and populated {processed_count} records ({failed_inserts} failed DB inserts).",
            "processed_records": processed_count,
            "original_count": original_row_count if original_row_count > 0 else db_results.get("original_count", 0),
            "failed_inserts": failed_inserts
        }
        print(f"[TASK END] Completed successfully. Result: {result_data}")
        return result_data

    except Exception as e:
        import traceback
        error_msg = f"Error in pipeline task: {str(e)}\n{traceback.format_exc()}"
        print(f"ERROR: {error_msg}")
        # Retry logic
        try:
            # Use self.retry for Celery's built-in retry mechanism
            print(f"Retrying task... (Attempt {self.request.retries + 1}/{self.max_retries})")
            raise self.retry(exc=e, countdown=RETRY_DELAY * (self.request.retries + 1)) # Exponential backoff
        except MaxRetriesExceededError as max_retry_e:
             final_error_msg = f"Pipeline task failed after {self.max_retries} retries: {str(e)}"
             print(f"CRITICAL ERROR: {final_error_msg}")
             # Return error state instead of raising MaxRetriesExceededError further?
             return {"status": "error", "message": final_error_msg, "original_count": original_row_count}
             # Or re-raise if you want Celery to mark it as failed definitely
             # raise max_retry_e from e
        except AttributeError:
             # self.request might not be available if run outside Celery context
             print("CRITICAL ERROR: Could not retry task (likely not run via Celery worker).")
             return {"status": "error", "message": error_msg, "original_count": original_row_count}


    finally:
        # --- Cleanup ---
        print("Cleaning up temporary files...")
        if os.path.exists(file_path):
            try: os.remove(file_path) ; print(f"  Removed: {file_path}")
            except OSError as e: print(f"  Error removing {file_path}: {e}")
        if processed_csv and os.path.exists(processed_csv):
             try: os.remove(processed_csv) ; print(f"  Removed: {processed_csv}")
             except OSError as e: print(f"  Error removing {processed_csv}: {e}")
        if csv_with_bias and os.path.exists(csv_with_bias):
             try: os.remove(csv_with_bias) ; print(f"  Removed: {csv_with_bias}")
             except OSError as e: print(f"  Error removing {csv_with_bias}: {e}")
        gc.collect()


# --- predict_sentiment_with_lstm (Keep as is) ---
# This function is now correctly called within the main task
def predict_sentiment_with_lstm(df: pd.DataFrame, model_path: str, vocab_path: str) -> pd.DataFrame:
    """
    Use the LSTM model to predict sentiment for each comment in the dataframe.
    Args: ... Returns: ...
    """
    print("  Loading vocab...")
    with open(vocab_path, 'r', encoding='utf-8') as f: vocab = json.load(f)
    print("  Setting up device...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    print("  Initializing model...")
    model = CustomSentimentLSTM(len(vocab), 128, 256, 2, 0.5).to(device)
    print("  Loading model weights...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("  Model ready for prediction.")

    max_len = 100
    df['sentiment'] = pd.NA # Use pd.NA for nullable Int64
    df['confidence'] = np.nan # Keep as float

    batch_size = 128 # Adjust based on GPU memory
    print(f"  Predicting sentiment in batches of {batch_size}...")
    num_batches = (len(df) + batch_size - 1) // batch_size

    # Use tqdm for progress bar
    for start_idx in tqdm(range(0, len(df), batch_size), total=num_batches, desc="Sentiment Prediction"):
        end_idx = min(start_idx + batch_size, len(df))
        batch = df.iloc[start_idx:end_idx]
        if batch.empty: continue

        text_inputs = []
        valid_indices_in_batch = [] # Keep track of original indices processed
        for idx, text in batch['processed_comment'].items():
            # Ensure text is a valid string
            text_str = str(text) if pd.notna(text) else ''
            tokens = text_str.split()
            indices = [vocab.get(t, 0) for t in tokens]
            if len(indices) > max_len: indices = indices[:max_len]
            else: indices += [1] * (max_len - len(indices)) # 1 = <pad>
            text_inputs.append(indices)            
            valid_indices_in_batch.append(idx) # Store original DataFrame index

        if text_inputs:
            inputs = torch.tensor(text_inputs, dtype=torch.long).to(device)
            with torch.no_grad():
                # Get outputs from model (don't assume it returns a tuple)
                outputs = model(inputs, return_attention=False)
                # If it's a tuple, take the first element (main outputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                probs = torch.sigmoid(outputs).cpu().numpy().flatten() # Apply sigmoid here

            sentiments = (probs >= 0.5).astype(int)
            confidences = np.abs(probs - 0.5) * 2

            # Update DataFrame using the original indices
            df.loc[valid_indices_in_batch, 'sentiment'] = sentiments
            df.loc[valid_indices_in_batch, 'confidence'] = confidences

        # Clean up GPU memory
        del text_inputs, batch
        if 'inputs' in locals(): del inputs
        if 'outputs' in locals(): del outputs
        if 'probs' in locals(): del probs
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Final cast to nullable integer after filling
    df['sentiment'] = df['sentiment'].astype('Int64')
    print("  Sentiment prediction batch processing complete.")
    return df