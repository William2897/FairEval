from celery import shared_task, chain
from celery.exceptions import MaxRetriesExceededError
from django.db import transaction
from django.db.models import F
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from .models import Rating, Sentiment, Professor
from .utils import calculate_professor_metrics, generate_recommendations
from data_processing.text_preprocessing import extract_opinion_terms, extract_vader_terms, clean_text, process_texts
from data_processing.pipeline import run_full_pipeline
import gc
import os
import json
import pandas as pd
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from django.conf import settings
import tempfile

# Import LSTM model
from machine_learning.ml_model_dev.lstm import CustomSentimentLSTM

BATCH_SIZE = getattr(settings, 'CELERY_BATCH_SIZE', 1000)
MAX_RETRIES = getattr(settings, 'CELERY_MAX_RETRIES', 3)
RETRY_DELAY = getattr(settings, 'CELERY_RETRY_DELAY', 60)  # seconds

@shared_task(
    bind=True, 
    max_retries=MAX_RETRIES,
    rate_limit='100/m',  # Limit to 100 tasks per minute
    priority=10  # Higher priority task
)
def process_rating_upload(self, ratings_data: List[Dict]):
    """Process uploaded ratings data in batches"""
    processed = 0
    affected_professors = set()
    errors = []
    
    try:
        # Process in batches for better memory management
        for i in range(0, len(ratings_data), BATCH_SIZE):
            batch = ratings_data[i:i + BATCH_SIZE]
            with transaction.atomic():
                for rating_data in batch:
                    try:
                        rating = Rating.objects.create(**rating_data)
                        processed += 1
                        affected_professors.add(rating.professor_id)
                        
                        # Queue professor metrics update
                        chain(
                            calculate_professor_metrics.s(rating.professor_id),
                            analyze_comments_sentiment.s()
                        ).delay()
                        
                    except Exception as e:
                        errors.append(f"Error processing rating: {str(e)}")
            
            # Clean up after each batch
            gc.collect()

        # Notify connected clients about updates
        channel_layer = get_channel_layer()
        for professor_id in affected_professors:
            async_to_sync(channel_layer.group_send)(
                f"professor_{professor_id}",
                {
                    "type": "professor.update",
                    "professor_id": professor_id,
                }
            )
            
    except Exception as e:
        if self.request.retries < MAX_RETRIES:
            self.retry(exc=e, countdown=RETRY_DELAY)
        raise MaxRetriesExceededError(f"Failed to process ratings after {MAX_RETRIES} retries: {str(e)}")
    
    return {
        'processed': processed,
        'errors': errors,
        'affected_professors': list(affected_professors)
    }

@shared_task(
    bind=True, 
    max_retries=MAX_RETRIES,
    rate_limit='50/m',  # Limit to 50 tasks per minute
    priority=5  # Medium priority task
)
def analyze_comments_sentiment(self, professor_id: str):
    """Process comments for existing Sentiment records to extract additional sentiment information"""
    try:
        professor = Professor.objects.get(professor_id=professor_id)
        
        # Get Sentiment records that need processing
        sentiments = Sentiment.objects.filter(
            professor_id=professor_id,
            processed_comment__isnull=True
        )
        
        # Process in batches
        for i in range(0, sentiments.count(), BATCH_SIZE):
            batch = sentiments[i:i + BATCH_SIZE]
            
            with transaction.atomic():
                for sentiment in batch:
                    if not sentiment.comment:
                        continue
                        
                    # Clean and process comment
                    processed_comment = clean_text(sentiment.comment)
                    processed_tokens = processed_comment.split()
                    
                    # Get lexicon terms
                    pos_terms_lexicon, neg_terms_lexicon = extract_opinion_terms(processed_tokens)
                    
                    # Get VADER terms
                    pos_terms_vader, neg_terms_vader = extract_vader_terms(processed_comment)
                    
                    # Update existing sentiment record
                    sentiment.processed_comment = processed_comment
                    sentiment.positive_terms_lexicon = pos_terms_lexicon
                    sentiment.negative_terms_lexicon = neg_terms_lexicon
                    sentiment.positive_terms_vader = pos_terms_vader
                    sentiment.negative_terms_vader = neg_terms_vader
                    sentiment.save()
            
            # Clean up after each batch
            gc.collect()
        
        # Queue recommendations update
        generate_recommendations.delay(professor_id)
        
    except Exception as e:
        if self.request.retries < MAX_RETRIES:
            self.retry(exc=e, countdown=RETRY_DELAY)
        raise MaxRetriesExceededError(f"Failed to analyze comments after {MAX_RETRIES} retries: {str(e)}")

@shared_task(
    bind=True, 
    max_retries=MAX_RETRIES,
    rate_limit='10/m',  # Limit to 10 tasks per minute
    priority=3  # Lower priority since this is less time-sensitive
)
def calculate_discipline_analytics(self, discipline: str):
    """Update analytics for an entire discipline"""
    try:
        professors = Professor.objects.filter(discipline=discipline)
        
        # Process professors in batches
        for i in range(0, professors.count(), BATCH_SIZE):
            batch = professors[i:i + BATCH_SIZE]
            
            for professor in batch:
                chain(
                    calculate_professor_metrics.s(professor.id),
                    analyze_comments_sentiment.s()
                ).delay()
                
            gc.collect()
            
    except Exception as e:
        if self.request.retries < MAX_RETRIES:
            self.retry(exc=e, countdown=RETRY_DELAY)
        raise MaxRetriesExceededError(f"Failed to process discipline after {MAX_RETRIES} retries: {str(e)}")

@shared_task(
    bind=True, 
    max_retries=MAX_RETRIES,
    priority=3
)
def update_discipline_analytics(self, discipline: str):
    """Update analytics for a discipline"""
    return calculate_discipline_analytics.delay(discipline)

@shared_task(
    bind=True,
    max_retries=MAX_RETRIES,
    priority=9  # High priority task
)
def process_evaluation_data_task(self, file_path: str, db_config: Dict[str, Any]):
    """
    Process uploaded evaluation data CSV with sentiment analysis using LSTM model
    
    This task:
    1. Runs the preprocessing pipeline on the CSV file
    2. Uses the LSTM model to predict sentiment for each comment
    3. Populates the database with the processed data
    """
    processed_csv_path = None # Initialize to None
    try:
        # Ensure the file exists
        if not os.path.exists(file_path):
            print(f"[DEBUG] File not found: {file_path}")
            raise FileNotFoundError(f"File {file_path} not found")
            
        # Get LSTM model paths
        model_path = os.path.join(settings.BASE_DIR, 'machine_learning/ml_models_trained/lstm_sentiment.pt')
        vocab_path = os.path.join(settings.BASE_DIR, 'machine_learning/ml_models_trained/vocab.json')
        
        # Check if model and vocab files exist
        if not os.path.exists(model_path) or not os.path.exists(vocab_path):
            print(f"[DEBUG] LSTM model files not found (model: {model_path}, vocab: {vocab_path})")
            raise FileNotFoundError("LSTM model files not found")
            
        # Run initial data preprocessing pipeline to create a processed CSV file
        processed_csv_path = file_path.replace('.csv', '_processed.csv')
        print(f"[DEBUG] Running initial pipeline on {file_path}")
        
        # Load the original CSV to get row count before processing
        import pandas as pd
        original_df = pd.read_csv(file_path)
        original_row_count = len(original_df)
        print(f"[DEBUG] Original CSV has {original_row_count} rows")
        
        # Run the full pipeline excluding sentiment analysis first
        run_full_pipeline(file_path, db_config, run_sentiment=False)
        print(f"[DEBUG] Initial pipeline finished. Processed CSV should be at: {processed_csv_path}")
        
        # Now load the processed CSV to add sentiment analysis
        if not os.path.exists(processed_csv_path):
            print(f"[DEBUG] Processed CSV file not found after pipeline run: {processed_csv_path}")
            raise FileNotFoundError(f"Processed CSV file not found: {processed_csv_path}")
             
        df = pd.read_csv(processed_csv_path)
        print(f"[DEBUG] Loaded processed CSV. Shape: {df.shape}, Row count: {len(df)}")
        
        # Add sentiment analysis columns
        print(f"[DEBUG] Starting sentiment prediction...")
        df = predict_sentiment_with_lstm(df, model_path, vocab_path)
        print(f"[DEBUG] Sentiment prediction finished. Shape after prediction: {df.shape}, Row count: {len(df)}")
        
        # Save the dataframe with sentiment analysis back to CSV
        df.to_csv(processed_csv_path, index=False)
        print(f"[DEBUG] Saved CSV with sentiment. Shape: {df.shape}, Row count: {len(df)}")
          # Run the database population with sentiment included
        from data_processing.pipeline import run_db_population
        print(f"[DEBUG] Starting database population with {processed_csv_path}...")
        db_results = run_db_population(processed_csv_path, db_config)
        print(f"[DEBUG] Database population finished. Results: {db_results}")
        
        # Get the actual number of processed records from the database population
        processed_count = db_results["successful_inserts"] if db_results else len(df)
        
        # Cleanup temporary files
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"[DEBUG] Removed original file: {file_path}")
        if os.path.exists(processed_csv_path):
            os.remove(processed_csv_path)
            print(f"[DEBUG] Removed processed file: {processed_csv_path}")
            
        result_data = {
            "status": "success",
            "message": f"Successfully processed {processed_count} records with sentiment analysis",
            "processed_records": processed_count,
            "original_count": db_results.get("original_count", len(df)) if db_results else len(df),
            "failed_inserts": db_results.get("failed_inserts", 0) if db_results else 0
        }
        print(f"[DEBUG] Task completed successfully. Result data: {result_data}")
        return result_data
        
    except Exception as e:
        import traceback
        error_msg = f"Error processing evaluation data: {str(e)}\n{traceback.format_exc()}"
        
        # Cleanup on error
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(processed_csv_path):
            os.remove(processed_csv_path)
            
        if self.request.retries < MAX_RETRIES:
            self.retry(exc=e, countdown=RETRY_DELAY)
        raise MaxRetriesExceededError(error_msg)

def predict_sentiment_with_lstm(df: pd.DataFrame, model_path: str, vocab_path: str) -> pd.DataFrame:
    """
    Use the LSTM model to predict sentiment for each comment in the dataframe
    
    Args:
        df: Dataframe with processed_comment column
        model_path: Path to the LSTM model file
        vocab_path: Path to the vocabulary JSON file
        
    Returns:
        DataFrame with additional sentiment and confidence columns
    """
    # Load vocabulary
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
        
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = CustomSentimentLSTM(
        vocab_size=len(vocab),
        embed_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.5
    ).to(device)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Convert tokens to indices
    max_len = 100  # Match the model's expected sequence length
    
    # Initialize sentiment and confidence columns
    df['sentiment'] = np.nan
    df['confidence'] = np.nan
    
    # Process in batches to avoid memory issues
    batch_size = 128
    for start_idx in range(0, len(df), batch_size):
        end_idx = min(start_idx + batch_size, len(df))
        batch = df.iloc[start_idx:end_idx]
        
        # Prepare inputs
        text_inputs = []
        for text in batch['processed_comment'].fillna(''):
            # Convert to string if needed
            if not isinstance(text, str):
                text = str(text) if pd.notna(text) else ''
                
            # Tokenize and convert to indices
            tokens = text.split()
            indices = [vocab.get(t, 0) for t in tokens]  # 0 = <unk>
            
            # Pad or truncate
            if len(indices) > max_len:
                indices = indices[:max_len]
            else:
                indices += [1] * (max_len - len(indices))  # 1 = <pad>
                
            text_inputs.append(indices)
        
        # Convert to tensor
        if text_inputs:
            inputs = torch.tensor(text_inputs, dtype=torch.long).to(device)
            
            # Run inference
            with torch.no_grad():
                outputs, attention_weights = model(inputs, return_attention=True)
                
                # Convert to numpy
                probs = outputs.cpu().numpy().flatten()
                
                # 0.5 threshold for sentiment (0=negative, 1=positive)
                sentiments = (probs >= 0.5).astype(int)
                
                # Calculate confidence: distance from 0.5 threshold * 2
                # This gives a value from 0 to 1, where 1 is max confidence
                confidences = np.abs(probs - 0.5) * 2
                
                # Update dataframe
                df.loc[batch.index, 'sentiment'] = sentiments
                df.loc[batch.index, 'confidence'] = confidences
        
        # Clean up to avoid memory issues
        del text_inputs
        if 'inputs' in locals():
            del inputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return df