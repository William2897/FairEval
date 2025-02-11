from django.db.models import Avg, Count, Sum
from django.utils import timezone
from datetime import timedelta
from .models import Rating, Sentiment, Professor

def calculate_professor_metrics(professor_id):
    """Calculate aggregate metrics for a professor"""
    metrics = Rating.objects.filter(professor_id=professor_id).aggregate(
        avg_rating=Avg('avg_rating'),
        avg_helpful=Avg('helpful_rating'),
        avg_clarity=Avg('clarity_rating'),
        avg_difficulty=Avg('difficulty_rating'),
        total_ratings=Count('id')
    )
    
    # Calculate trends
    thirty_days_ago = timezone.now() - timedelta(days=30)
    recent_metrics = Rating.objects.filter(
        professor_id=professor_id,
        created_at__gte=thirty_days_ago
    ).aggregate(
        recent_avg=Avg('avg_rating')
    )
    
    metrics['trend'] = (recent_metrics['recent_avg'] or 0) - (metrics['avg_rating'] or 0)
    return metrics

def get_sentiment_summary(professor_id):
    """Get sentiment analysis summary for a professor"""
    sentiments = Sentiment.objects.filter(professor_id=professor_id)
    
    summary = {
        'total_comments': sentiments.count(),
        'sentiment_breakdown': {
            'positive': sentiments.filter(sentiment__gt=0).count(),
            'negative': sentiments.filter(sentiment__lt=0).count()
        },
        'recent_sentiments': list(
            sentiments.order_by('-created_at')[:5]
            .values('comment', 'processed_comment', 'sentiment', 'created_at')
        )
    }
    return summary

def analyze_department_bias(department_id):
    """Analyze potential gender bias in ratings within a department"""
    professors = Professor.objects.filter(department_id=department_id)
    
    male_stats = professors.filter(gender='M').annotate(
        avg_rating=Avg('ratings__avg_rating'),
        rating_count=Count('ratings')
    )
    
    female_stats = professors.filter(gender='F').annotate(
        avg_rating=Avg('ratings__avg_rating'),
        rating_count=Count('ratings')
    )
    
    return {
        'male_stats': {
            'avg_rating': male_stats.aggregate(Avg('avg_rating'))['avg_rating__avg'] or 0,
            'total_ratings': male_stats.aggregate(Sum('rating_count'))['rating_count__sum'] or 0
        },
        'female_stats': {
            'avg_rating': female_stats.aggregate(Avg('avg_rating'))['avg_rating__avg'] or 0,
            'total_ratings': female_stats.aggregate(Sum('rating_count'))['rating_count__sum'] or 0
        }
    }

def generate_recommendations(professor_id):
    """Generate teaching improvement recommendations based on ratings and comments"""
    metrics = calculate_professor_metrics(professor_id)
    sentiments = get_sentiment_summary(professor_id)
    
    recommendations = {
        'teaching_effectiveness': {
            'score': metrics['avg_rating'] or 0,
            'recommendations': []
        },
        'clarity': {
            'score': metrics['avg_clarity'] or 0,
            'recommendations': []
        },
        'workload': {
            'score': metrics['avg_difficulty'] or 0,
            'recommendations': []
        }
    }
    
    # Add recommendations based on metrics
    if metrics['avg_clarity'] and metrics['avg_clarity'] < 4.0:
        recommendations['clarity']['recommendations'].append({
            'text': 'Consider providing more detailed explanations and examples',
            'priority': 'high',
            'impact_score': 8.5,
            'supporting_ratings': metrics['total_ratings']
        })
    
    # Add more recommendations based on sentiment analysis
    if sentiments['sentiment_breakdown']['negative'] > sentiments['sentiment_breakdown']['positive']:
        recommendations['teaching_effectiveness']['recommendations'].append({
            'text': 'Review and address common themes in negative feedback',
            'priority': 'high',
            'impact_score': 9.0,
            'supporting_ratings': sentiments['sentiment_breakdown']['negative']
        })
    
    return recommendations