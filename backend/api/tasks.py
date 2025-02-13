from celery import shared_task
from django.db.models import Avg
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from .models import Rating, Sentiment, Professor
from .utils import calculate_professor_metrics, generate_recommendations

@shared_task
def process_rating_upload(ratings_data):
    """Process uploaded ratings data and calculate metrics"""
    processed = 0
    errors = []
    
    for rating_data in ratings_data:
        try:
            rating = Rating.objects.create(**rating_data)
            # Update professor metrics
            calculate_professor_metrics(rating.professor_id)
            processed += 1
        except Exception as e:
            errors.append(f"Error processing rating: {str(e)}")
    
    return {
        'processed': processed,
        'errors': errors
    }

@shared_task
def analyze_comments_sentiment(professor_id):
    """Analyze sentiment for a professor's ratings comments"""
    from data_processing.text_preprocessing import get_vader_sentiment, extract_opinion_terms
    
    ratings = Rating.objects.filter(
        professor_id=professor_id,
        flag_status__isnull=True  # Only analyze unflagged comments
    )
    
    for rating in ratings:
        if not rating.comment:
            continue
            
        # Get VADER scores
        vader_scores = get_vader_sentiment(rating.comment)
        
        # Get opinion lexicon terms
        processed_tokens = rating.comment.lower().split()
        pos_terms, neg_terms = extract_opinion_terms(processed_tokens)
        
        # Calculate overall sentiment (combine VADER compound with lexicon ratio)
        lexicon_ratio = (len(pos_terms) - len(neg_terms)) / (len(pos_terms) + len(neg_terms) + 1)
        combined_sentiment = (vader_scores['compound'] + lexicon_ratio) / 2
        
        Sentiment.objects.create(
            professor_id=professor_id,
            comment=rating.comment,
            processed_comment=' '.join(processed_tokens),
            sentiment=combined_sentiment,
            vader_compound=vader_scores['compound'],
            vader_positive=vader_scores['pos'],
            vader_negative=vader_scores['neg'],
            vader_neutral=vader_scores['neu'],
            positive_terms=pos_terms,
            negative_terms=neg_terms
        )
    
    # Generate new recommendations based on updated sentiment
    generate_recommendations(professor_id)

def calculate_discipline_analytics(discipline):
    """Update analytics for an entire discipline"""
    professors = Professor.objects.filter(discipline=discipline)
    for professor in professors:
        calculate_professor_metrics(professor.id)
        analyze_comments_sentiment.delay(professor.id)  # Async for each professor

# Remove department-specific task and use discipline-based one instead
@shared_task
def update_discipline_analytics(discipline):
    """Update analytics for a discipline"""
    calculate_discipline_analytics(discipline)