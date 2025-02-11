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
    analyzer = SentimentIntensityAnalyzer()
    ratings = Rating.objects.filter(
        professor_id=professor_id,
        flag_status__isnull=True  # Only analyze unflagged comments
    )
    
    for rating in ratings:
        if not rating.comment:
            continue
            
        # Use both VADER and TextBlob for better accuracy
        vader_scores = analyzer.polarity_scores(rating.comment)
        textblob_sentiment = TextBlob(rating.comment).sentiment
        
        # Combine both sentiment scores
        combined_sentiment = (vader_scores['compound'] + textblob_sentiment.polarity) / 2
        
        Sentiment.objects.create(
            professor_id=professor_id,
            comment=rating.comment,
            processed_comment=rating.comment,  # You could add text cleaning here
            sentiment=combined_sentiment
        )
    
    # Generate new recommendations based on updated sentiment
    generate_recommendations(professor_id)

@shared_task
def update_department_analytics(department_id):
    """Update analytics for an entire department"""
    professors = Professor.objects.filter(department_id=department_id)
    for professor in professors:
        calculate_professor_metrics(professor.id)
        analyze_comments_sentiment.delay(professor.id)  # Async for each professor