# backend/api/tasks.py

from celery import shared_task, chain
from celery.exceptions import MaxRetriesExceededError
from django.db import transaction
# Removed unused F import: from django.db.models import F
# Other Django imports seem fine
from .models import Rating, Sentiment, Professor
from .utils import calculate_professor_metrics # Removed generate_recommendations if not used directly here

# Import the specific pipeline functions needed
from data_processing.pipeline import (
    run_initial_processing,
    run_bias_analysis_only,
    run_db_population
)
# Keep text processing imports if analyze_comments_sentiment is still used elsewhere
from data_processing.text_preprocessing import (
    extract_opinion_terms,
    extract_vader_terms,
    clean_text # Keep clean_text if used directly, else remove
    # process_texts is likely part of run_initial_processing now, remove if not directly used
)

import gc
import os
import json
import pandas as pd
import torch # Keep torch import
import numpy as np
from typing import List, Dict, Any # Removed Tuple if not used
from django.conf import settings
import tqdm
# Import LSTM model
from machine_learning.ml_model_dev.lstm import CustomSentimentLSTM

# Celery settings
BATCH_SIZE = getattr(settings, 'CELERY_DB_BATCH_SIZE', 1000) # Use a specific setting name
PIPELINE_CHUNK_SIZE = getattr(settings, 'CELERY_PIPELINE_CHUNK_SIZE', 5000) # For pipeline processing
MAX_RETRIES = getattr(settings, 'CELERY_MAX_RETRIES', 3)
RETRY_DELAY = getattr(settings, 'CELERY_RETRY_DELAY', 60)  # seconds

# --- Helper: Sentiment Prediction Function ---
# (Kept within tasks.py as it's tightly coupled with the upload task flow)
def predict_sentiment_with_lstm(df: pd.DataFrame, model_path: str, vocab_path: str) -> pd.DataFrame:
    """
    Use the LSTM model to predict sentiment for each comment in the dataframe.

    Args:
        df: Dataframe with processed_comment column
        model_path: Path to the LSTM model file
        vocab_path: Path to the vocabulary JSON file

    Returns:
        DataFrame with additional sentiment and confidence columns
    """
    print("Starting LSTM Sentiment Prediction...")
    if 'processed_comment' not in df.columns:
        print("Error: 'processed_comment' column missing for sentiment prediction.")
        # Add empty columns and return to prevent downstream errors
        df['sentiment'] = pd.NA
        df['confidence'] = np.nan
        return df

    # Load vocabulary
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        vocab_size = len(vocab)
        pad_idx = vocab.get('<pad>', 1) # Default pad index
        unk_idx = vocab.get('<unk>', 0) # Default unknown index
        print(f"Vocabulary loaded. Size: {vocab_size}")
    except Exception as e:
        print(f"Error loading vocabulary from {vocab_path}: {e}")
        raise # Re-raise to fail the task

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for LSTM: {device}")

    # Initialize model
    try:
        model = CustomSentimentLSTM(
            vocab_size=vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.5
        ).to(device)
        # Load model weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # Set to evaluation mode
        print("LSTM Model initialized and weights loaded.")
    except Exception as e:
        print(f"Error initializing/loading LSTM model from {model_path}: {e}")
        raise # Re-raise to fail the task

    # Tokenization and Padding settings
    max_len = 100  # Match the model's expected sequence length

    # Initialize sentiment and confidence columns if they don't exist
    if 'sentiment' not in df.columns: df['sentiment'] = pd.NA # Use pd.NA for nullable Int
    if 'confidence' not in df.columns: df['confidence'] = np.nan

    # Process in batches
    lstm_batch_size = 128 # Can be different from DB batch size
    num_batches = (len(df) + lstm_batch_size - 1) // lstm_batch_size
    print(f"Processing {len(df)} comments in {num_batches} batches (size {lstm_batch_size})...")

    all_sentiments = []
    all_confidences = []

    for i in tqdm(range(num_batches), desc="LSTM Prediction"):
        start_idx = i * lstm_batch_size
        end_idx = min(start_idx + lstm_batch_size, len(df))
        batch_texts = df['processed_comment'].iloc[start_idx:end_idx].fillna('').astype(str).tolist()

        if not batch_texts: continue # Skip empty batches

        # Prepare inputs
        batch_indices = []
        for text in batch_texts:
            tokens = text.split()
            indices = [vocab.get(t, unk_idx) for t in tokens][:max_len] # Truncate first
            # Pad
            if len(indices) < max_len:
                indices += [pad_idx] * (max_len - len(indices))
            batch_indices.append(indices)

        # Convert to tensor
        inputs = torch.tensor(batch_indices, dtype=torch.long).to(device)

        # Run inference
        with torch.no_grad():
            outputs, _ = model(inputs, return_attention=False) # Don't need attention here
            probs = torch.sigmoid(outputs).cpu().numpy().flatten() # Apply sigmoid *here*

            sentiments = (probs >= 0.5).astype(int)
            confidences = np.abs(probs - 0.5) * 2

            all_sentiments.extend(sentiments)
            all_confidences.extend(confidences)

        # Clean up GPU memory periodically
        del inputs, outputs, probs, sentiments, confidences
        if i % 10 == 0: # Every 10 batches
             gc.collect()
             if torch.cuda.is_available():
                 torch.cuda.empty_cache()

    # Assign results back to the DataFrame
    # Important: Ensure lengths match if there were issues
    if len(all_sentiments) == len(df):
        df['sentiment'] = all_sentiments
        df['confidence'] = all_confidences
        # Convert sentiment to nullable integer type after assignment
        df['sentiment'] = df['sentiment'].astype('Int64')
    else:
        print(f"ERROR: Length mismatch during sentiment prediction. Expected {len(df)}, got {len(all_sentiments)}. Sentiment columns not updated.")
        # Keep original/NaN columns

    print("LSTM Sentiment Prediction finished.")
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return df


# --- Main Upload Task (Updated Flow) ---

@shared_task(bind=True, max_retries=MAX_RETRIES, priority=9)
def process_evaluation_data_task(self, file_path: str, db_config: Dict[str, Any]):
    """
    Orchestrates the processing of an uploaded evaluation CSV file.
    Steps:
    1. Run initial preprocessing (cleaning, gender, dept map, text).
    2. Run LSTM sentiment prediction on processed comments.
    3. Run gender bias analysis on comments.
    4. Populate the database with the fully processed data.
    """
    print(f"\n--- Starting Evaluation Data Processing Task for: {file_path} ---")
    processed_csv_path = None
    sentiment_csv_path = None
    bias_csv_path = None
    task_failed = False
    error_message = "Unknown error"
    final_results = {}

    try:
        # --- Step 1: Initial Processing ---
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Original file not found: {file_path}")

        processed_csv_path = run_initial_processing(file_path)
        if not processed_csv_path or not os.path.exists(processed_csv_path):
            raise RuntimeError("Initial processing failed or did not produce an output CSV.")
        print(f"Step 1 Complete: Initial processing saved to {processed_csv_path}")

        # --- Step 2: Sentiment Prediction ---
        # Load the processed CSV for sentiment prediction
        df_processed = pd.read_csv(processed_csv_path)
        print(f"Loaded {len(df_processed)} records for sentiment prediction.")

        # Define model paths
        model_path = os.path.join(settings.BASE_DIR, 'machine_learning/ml_models_trained/lstm_sentiment.pt')
        vocab_path = os.path.join(settings.BASE_DIR, 'machine_learning/ml_models_trained/vocab.json')
        if not os.path.exists(model_path) or not os.path.exists(vocab_path):
            raise FileNotFoundError("LSTM model or vocab file not found for sentiment prediction.")

        # Run prediction function
        df_with_sentiment = predict_sentiment_with_lstm(df_processed, model_path, vocab_path)
        del df_processed # Free memory
        gc.collect()

        # Save intermediate result (optional but good for debugging)
        sentiment_csv_path = processed_csv_path.replace('_processed.csv', '_processed_with_sentiment.csv')
        df_with_sentiment.to_csv(sentiment_csv_path, index=False)
        print(f"Step 2 Complete: Sentiment prediction added, saved to {sentiment_csv_path}")

        # --- Step 3: Bias Analysis ---
        bias_csv_path = sentiment_csv_path.replace('_with_sentiment.csv', '_with_bias.csv')
        print(f"Starting bias analysis using {sentiment_csv_path}...")
        # run_bias_analysis_only loads the input, runs analysis, saves output
        output_path_from_bias = run_bias_analysis_only(
            sentiment_csv_path,
            bias_csv_path,
            chunk_size=PIPELINE_CHUNK_SIZE
        )
        del df_with_sentiment # Free memory
        gc.collect()

        if not output_path_from_bias or not os.path.exists(output_path_from_bias):
            raise RuntimeError("Bias analysis failed or did not produce an output CSV.")
        print(f"Step 3 Complete: Bias analysis finished, results saved to {output_path_from_bias}")
        # Ensure we use the correct final path
        bias_csv_path = output_path_from_bias


        # --- Step 4: Database Population ---
        print(f"Starting database population using {bias_csv_path}...")
        db_results = run_db_population(bias_csv_path, db_config)

        if not db_results:
             raise RuntimeError("Database population step failed.")
        print(f"Step 4 Complete: Database population finished. Results: {db_results}")

        final_results = {
            "status": "success",
            "message": f"Successfully processed and loaded {db_results.get('successful_inserts', 0)} records with sentiment and bias analysis.",
            "processed_records": db_results.get("successful_inserts", 0),
            "original_count": db_results.get("original_count", "N/A"),
            "failed_inserts": db_results.get("failed_inserts", 0)
        }

    except Exception as e:
        task_failed = True
        import traceback
        error_message = f"Error in evaluation data task: {str(e)}\n{traceback.format_exc()}"
        print(f"ERROR: {error_message}") # Log the detailed error
        # Attempt retry if applicable
        try:
            # Ensure self.request exists before accessing retries
            if hasattr(self, 'request') and self.request.retries < MAX_RETRIES:
                print(f"Retrying task (attempt {self.request.retries + 1}/{MAX_RETRIES})...")
                self.retry(exc=e, countdown=RETRY_DELAY)
            else:
                 print("Max retries exceeded or retry unavailable.")
                 # Raise MaxRetriesExceededError only if retries are actually exceeded
                 if hasattr(self, 'request') and self.request.retries >= MAX_RETRIES:
                      raise MaxRetriesExceededError(error_message) from e
                 else: # If retry mechanism isn't available (e.g., direct call)
                      final_results = {"status": "failure", "message": error_message}

        except MaxRetriesExceededError:
             print("Max retries exceeded exception caught.")
             final_results = {"status": "failure", "message": f"Task failed after {MAX_RETRIES} retries: {error_message}"}
        except AttributeError:
            # Handle cases where 'self.request' might not exist (e.g., direct function call)
            print("Retry mechanism not available (task might have been called directly).")
            final_results = {"status": "failure", "message": error_message}


    finally:
        # --- Cleanup ---
        print("Cleaning up temporary files...")
        files_to_remove = [file_path, processed_csv_path, sentiment_csv_path, bias_csv_path]
        for f_path in files_to_remove:
            if f_path and os.path.exists(f_path):
                try:
                    os.remove(f_path)
                    print(f"  Removed: {f_path}")
                except Exception as clean_e:
                    print(f"  Warning: Could not remove file {f_path}: {clean_e}")
        print("Cleanup finished.")

        # Return final status - ensure results are set even if retry fails immediately
        if not final_results: # If no result was set due to error before MaxRetriesExceededError
             final_results = {"status": "failure", "message": error_message}

        print(f"--- Evaluation Data Processing Task Finished --- Status: {final_results.get('status')}")
        return final_results


# --- Other Tasks (Likely unchanged, review if needed) ---

@shared_task(bind=True, max_retries=MAX_RETRIES, rate_limit='100/m', priority=10)
def process_rating_upload(self, ratings_data: List[Dict]):
    """Process uploaded ratings data in batches"""
    # This task seems focused on direct Rating creation and doesn't involve
    # the full sentiment/bias pipeline directly. Review if its logic
    # should trigger parts of the new pipeline.
    # For now, assuming it remains separate.
    print(f"Processing batch upload of {len(ratings_data)} ratings...")
    processed = 0
    affected_professors = set()
    errors = []
    try:
        for i in range(0, len(ratings_data), BATCH_SIZE):
            batch = ratings_data[i:i + BATCH_SIZE]
            with transaction.atomic():
                # ... (Rating creation logic) ...
                # Consider if creating a rating should trigger a sentiment/bias analysis task
                # for the associated comment if one exists.
                pass # Placeholder for existing logic
        # ... (Notification logic) ...
        print("Rating upload processing complete.")
    except Exception as e:
        # ... (Retry logic) ...
        print(f"Error during rating upload: {e}")
        raise
    return {'processed': processed, 'errors': errors, 'affected_professors': list(affected_professors)}


@shared_task(bind=True, max_retries=MAX_RETRIES, rate_limit='50/m', priority=5)
def analyze_comments_sentiment(self, professor_id: str):
    """
    Process comments for existing Sentiment records to extract
    LEXICON and VADER terms. (Does NOT run LSTM or Bias Tagging).
    """
    # This task focuses on lexicon/VADER terms, separate from the LSTM/Bias pipeline.
    # If you want *this* task to also trigger bias tagging, it needs modification.
    # Assuming it stays focused on lexicon/VADER for now.
    print(f"Analyzing comments for Lexicon/VADER terms for professor {professor_id}...")
    try:
        sentiments = Sentiment.objects.filter(
            professor_id=professor_id,
            # Maybe add condition: processed_comment__isnull=False AND positive_terms_lexicon__isnull=True ?
            # To only process ones that haven't been done yet.
        )
        if not sentiments.exists():
            print(f"  No unprocessed comments found for professor {professor_id}.")
            return f"No comments to analyze for {professor_id}"

        # Process in batches
        count = sentiments.count()
        num_updated = 0
        print(f"  Found {count} sentiments to analyze.")
        for i in range(0, count, BATCH_SIZE):
            batch_qs = sentiments[i:i + BATCH_SIZE] # Queryset slice
            ids_to_update = list(batch_qs.values_list('id', flat=True))
            if not ids_to_update: continue

            updates = []
            for sentiment in Sentiment.objects.filter(id__in=ids_to_update): # Fetch full objects for update
                if not sentiment.processed_comment: # Need processed comment
                     if sentiment.comment:
                         sentiment.processed_comment = clean_text(sentiment.comment) # Basic clean
                     else: continue # Skip if no comment

                processed_tokens = sentiment.processed_comment.split()
                pos_lex, neg_lex = extract_opinion_terms(processed_tokens)
                pos_vad, neg_vad = extract_vader_terms(sentiment.processed_comment)

                # Prepare update object (don't save yet)
                sentiment.positive_terms_lexicon = pos_lex
                sentiment.negative_terms_lexicon = neg_lex
                sentiment.positive_terms_vader = pos_vad
                sentiment.negative_terms_vader = neg_vad
                updates.append(sentiment)

            # Bulk update fields for the batch
            if updates:
                 Sentiment.objects.bulk_update(updates, [
                     'processed_comment', # If updated
                     'positive_terms_lexicon', 'negative_terms_lexicon',
                     'positive_terms_vader', 'negative_terms_vader'
                 ])
                 num_updated += len(updates)
                 print(f"  Updated lexicon/VADER terms for {len(updates)} sentiments in batch.")

            gc.collect()

        print(f"Lexicon/VADER analysis complete for professor {professor_id}. Updated {num_updated} records.")
        # Queue recommendations update? (Keep if desired)
        # generate_recommendations.delay(professor_id) # Assuming generate_recommendations uses the new bias info if available

    except Professor.DoesNotExist:
         print(f"Error: Professor {professor_id} not found for comment analysis.")
         # Don't retry if professor doesn't exist
    except Exception as e:
        # ... (Retry logic) ...
        print(f"Error during comment analysis for {professor_id}: {e}")
        raise
    return f"Comment analysis complete for {professor_id}. Updated {num_updated}."


@shared_task(bind=True, max_retries=MAX_RETRIES, rate_limit='10/m', priority=3)
def calculate_discipline_analytics(self, discipline: str):
    """Update analytics for an entire discipline by triggering individual professor updates."""
    # This task likely triggers calculate_professor_metrics, which is fine.
    # If it needs to trigger the full bias pipeline per professor, the logic needs change.
    print(f"Calculating analytics for discipline: {discipline}")
    try:
        professors = Professor.objects.filter(discipline=discipline)
        print(f"  Found {professors.count()} professors.")
        # Process professors in batches
        professor_ids = list(professors.values_list('professor_id', flat=True))
        for i in range(0, len(professor_ids), BATCH_SIZE):
            batch_ids = professor_ids[i:i + BATCH_SIZE]
            for prof_id in batch_ids:
                # Queue metrics calculation (assuming this is the desired action)
                calculate_professor_metrics.delay(prof_id)
                # If you need VADER/Lexicon terms updated:
                # analyze_comments_sentiment.delay(prof_id)
            print(f"  Queued updates for {len(batch_ids)} professors in batch.")
            gc.collect()
        print(f"Finished queueing tasks for discipline {discipline}.")
    except Exception as e:
        # ... (Retry logic) ...
        print(f"Error processing discipline {discipline}: {e}")
        raise
    return f"Analytics calculation queued for discipline {discipline}"

# This seems redundant if calculate_discipline_analytics uses .delay() itself
# @shared_task(bind=True, max_retries=MAX_RETRIES, priority=3)
# def update_discipline_analytics(self, discipline: str):
#     """Update analytics for a discipline"""
#     return calculate_discipline_analytics.delay(discipline)