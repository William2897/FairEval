from celery import shared_task, chain
from celery.exceptions import MaxRetriesExceededError
from django.db import transaction
from django.db.models import F
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from .models import Rating, Sentiment, Professor
from .utils import calculate_professor_metrics, generate_recommendations
from data_processing.text_preprocessing import extract_opinion_terms, extract_vader_terms, clean_text, process_texts
import gc
from typing import List, Dict
from django.conf import settings

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