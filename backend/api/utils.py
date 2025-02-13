from django.db.models import Avg, Count, Sum, F
from django.utils import timezone
from datetime import timedelta
from .models import Rating, Sentiment, Professor
from collections import Counter
from itertools import chain
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd
import numpy as np

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
    from collections import Counter
    from itertools import chain
    
    sentiments = Sentiment.objects.filter(professor_id=professor_id)
    
    # Get word frequencies with gender information
    sentiments_with_gender = sentiments.annotate(
        gender=F('professor__gender')
    ).filter(gender__in=['Male', 'Female'])
    
    male_sentiments = sentiments_with_gender.filter(gender='Male')
    female_sentiments = sentiments_with_gender.filter(gender='Female')
    
    # Get gender-specific term frequencies
    def get_term_frequencies(queryset, terms_field):
        terms = chain.from_iterable(s[terms_field] for s in queryset if s[terms_field])
        return Counter(terms)
    
    male_pos_counter = get_term_frequencies(male_sentiments.values('positive_terms'), 'positive_terms')
    male_neg_counter = get_term_frequencies(male_sentiments.values('negative_terms'), 'negative_terms')
    female_pos_counter = get_term_frequencies(female_sentiments.values('positive_terms'), 'positive_terms')
    female_neg_counter = get_term_frequencies(female_sentiments.values('negative_terms'), 'negative_terms')
    
    # Calculate relative frequencies and bias
    def calculate_gender_bias(male_counter, female_counter, bias_threshold=1.1):
        all_terms = set(male_counter.keys()) | set(female_counter.keys())
        male_total = sum(male_counter.values()) or 1
        female_total = sum(female_counter.values()) or 1
        
        result = []
        for term in all_terms:
            male_freq = male_counter[term]
            female_freq = female_counter[term]
            male_rel_freq = male_freq / male_total
            female_rel_freq = female_freq / female_total
            
            # Determine bias
            if male_rel_freq > bias_threshold * female_rel_freq:
                bias = 'Male'
            elif female_rel_freq > bias_threshold * male_rel_freq:
                bias = 'Female'
            else:
                bias = 'Neutral'
                
            result.append({
                'term': term,
                'male_freq': male_freq,
                'female_freq': female_freq,
                'bias': bias
            })
        
        return sorted(result, key=lambda x: max(x['male_freq'], x['female_freq']), reverse=True)[:10]
    
    # Get existing sentiment summary data
    summary = {
        'total_comments': sentiments.count(),
        'sentiment_breakdown': {
            'positive': sentiments.filter(sentiment__gt=0).count(),
            'negative': sentiments.filter(sentiment__lt=0).count()
        },
        'vader_scores': {
            'compound': sentiments.aggregate(Avg('vader_compound'))['vader_compound__avg'] or 0,
            'positive': sentiments.aggregate(Avg('vader_positive'))['vader_positive__avg'] or 0,
            'negative': sentiments.aggregate(Avg('vader_negative'))['vader_negative__avg'] or 0
        },
        'top_words': {
            'positive': [{'word': word, 'count': count} 
                        for word, count in Counter(chain.from_iterable(
                            s.positive_terms for s in sentiments if s.positive_terms
                        )).most_common(20)],
            'negative': [{'word': word, 'count': count} 
                        for word, count in Counter(chain.from_iterable(
                            s.negative_terms for s in sentiments if s.negative_terms
                        )).most_common(20)]
        },
        'gender_analysis': {
            'positive_terms': calculate_gender_bias(male_pos_counter, female_pos_counter),
            'negative_terms': calculate_gender_bias(male_neg_counter, female_neg_counter)
        },
        'recent_sentiments': list(
            sentiments.order_by('-created_at')[:5]
            .values('comment', 'processed_comment', 'sentiment', 'created_at',
                   'vader_compound')
        )
    }
    
    return summary

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

def analyze_discipline_ratings():
    """Analyze ratings across disciplines"""
    discipline_stats = Professor.objects.values('discipline', 'sub_discipline').annotate(
        avg_rating=Avg('ratings__avg_rating'),
        total_ratings=Count('ratings'),
        professor_count=Count('id', distinct=True)
    ).exclude(discipline__isnull=True)
    
    return list(discipline_stats)

def analyze_discipline_gender_distribution():
    """Analyze discipline ratings by gender"""
    gender_discipline_stats = Professor.objects.values(
        'discipline', 'sub_discipline', 'gender'
    ).annotate(
        avg_rating=Avg('ratings__avg_rating'),
        total_ratings=Count('ratings')
    ).exclude(
        discipline__isnull=True
    ).exclude(
        gender__isnull=True
    )
    
    return list(gender_discipline_stats)

def perform_discipline_tukey_hsd():
    """Perform Tukey's HSD test for disciplines"""
    # Get all ratings with gender and discipline info
    ratings_data = Rating.objects.select_related('professor').values(
        'avg_rating',
        'professor__gender',
        'professor__discipline',
        'professor__sub_discipline'
    ).exclude(
        professor__discipline__isnull=True,
        professor__gender__isnull=True
    )
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(ratings_data)
    df = df.rename(columns={
        'professor__gender': 'gender',
        'professor__discipline': 'discipline',
        'professor__sub_discipline': 'sub_discipline'
    })
    
    # Group by discipline and gender
    grouped = df.groupby(['discipline', 'gender'])['avg_rating'].agg(['mean', 'count']).reset_index()
    
    results = []
    for discipline in df['discipline'].unique():
        discipline_data = df[df['discipline'] == discipline]
        if len(discipline_data['gender'].unique()) < 2:
            continue
            
        # Perform one-way ANOVA
        groups = [group['avg_rating'].values for _, group in discipline_data.groupby('gender')]
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Perform Tukey's HSD if we have enough data
        if len(discipline_data) >= 3:
            tukey = pairwise_tukeyhsd(
                discipline_data['avg_rating'],
                discipline_data['gender']
            )
            # Convert tukey results to dict, handling the data types properly
            tukey_data = []
            for row in tukey.summary().data[1:]:  # Skip header row
                tukey_data.append({
                    'group1': str(row[0]),
                    'group2': str(row[1]),
                    'meandiff': float(row[2]),
                    'lower': float(row[3]),
                    'upper': float(row[4]),
                    'reject': bool(row[5] == 'True')
                })
        else:
            tukey_data = None
            
        results.append({
            'discipline': discipline,
            'anova': {'f_stat': float(f_stat), 'p_value': float(p_value)},
            'tukey': tukey_data,
            'summary': [
                {
                    'discipline': row['discipline'],
                    'gender': row['gender'],
                    'mean': float(row['mean']),
                    'count': int(row['count'])
                }
                for _, row in grouped[grouped['discipline'] == discipline].iterrows()
            ]
        })
    
    return results

def calculate_gender_distribution():
    """Calculate gender distribution for top/bottom rated disciplines and sub-disciplines"""
    from django.db.models import Avg, Count, Case, When, F, FloatField
    from django.db.models.functions import Cast
    
    # Calculate overall gender distribution
    total_stats = Professor.objects.aggregate(
        total=Count('id'),
        female_count=Count(Case(When(gender='Female', then=1))),
        male_count=Count(Case(When(gender='Male', then=1))),
    )
    
    # Safely calculate percentages
    total = total_stats['total'] or 1  # Avoid division by zero
    total_stats['female_percent'] = round((total_stats['female_count'] * 100.0) / total, 2)
    total_stats['male_percent'] = round((total_stats['male_count'] * 100.0) / total, 2)
    
    # Calculate average ratings by discipline
    discipline_stats = Professor.objects.values('discipline').annotate(
        avg_rating=Avg('ratings__avg_rating')
    ).exclude(discipline__isnull=True).order_by('-avg_rating')
    
    # Get top 3 and bottom 3 disciplines
    top_3_disciplines = list(discipline_stats[:3].values_list('discipline', flat=True))
    bottom_3_disciplines = list(discipline_stats.reverse()[:3].values_list('discipline', flat=True))
    
    # Calculate average ratings by sub-discipline
    sub_discipline_stats = Professor.objects.values('sub_discipline').annotate(
        avg_rating=Avg('ratings__avg_rating')
    ).exclude(sub_discipline__isnull=True).order_by('-avg_rating')
    
    # Get top 10 and bottom 10 sub-disciplines
    top_10_sub_disciplines = list(sub_discipline_stats[:10].values_list('sub_discipline', flat=True))
    bottom_10_sub_disciplines = list(sub_discipline_stats.reverse()[:10].values_list('sub_discipline', flat=True))
    
    # Helper function to calculate distribution
    def calculate_distribution(queryset, is_sub_discipline=False):
        distributions = []
        for item in queryset:
            total = item['total'] or 1  # Avoid division by zero
            distributions.append({
                'discipline' if not is_sub_discipline else 'sub_discipline': item.get('discipline') or item.get('sub_discipline'),
                'total': item['total'],
                'female_count': item['female_count'],
                'male_count': item['male_count'],
                'female_percent': round((item['female_count'] * 100.0) / total, 2),
                'male_percent': round((item['male_count'] * 100.0) / total, 2)
            })
        return distributions
    
    # Calculate gender distribution for disciplines
    top_disciplines_dist = Professor.objects.filter(
        discipline__in=top_3_disciplines
    ).values('discipline').annotate(
        total=Count('id'),
        female_count=Count(Case(When(gender='Female', then=1))),
        male_count=Count(Case(When(gender='Male', then=1)))
    )
    
    bottom_disciplines_dist = Professor.objects.filter(
        discipline__in=bottom_3_disciplines
    ).values('discipline').annotate(
        total=Count('id'),
        female_count=Count(Case(When(gender='Female', then=1))),
        male_count=Count(Case(When(gender='Male', then=1)))
    )
    
    # Calculate gender distribution for sub-disciplines
    top_sub_disciplines_dist = Professor.objects.filter(
        sub_discipline__in=top_10_sub_disciplines
    ).values('sub_discipline').annotate(
        total=Count('id'),
        female_count=Count(Case(When(gender='Female', then=1))),
        male_count=Count(Case(When(gender='Male', then=1)))
    )
    
    bottom_sub_disciplines_dist = Professor.objects.filter(
        sub_discipline__in=bottom_10_sub_disciplines
    ).values('sub_discipline').annotate(
        total=Count('id'),
        female_count=Count(Case(When(gender='Female', then=1))),
        male_count=Count(Case(When(gender='Male', then=1)))
    )
    
    return {
        'total_stats': total_stats,
        'disciplines': {
            'top': calculate_distribution(top_disciplines_dist),
            'bottom': calculate_distribution(bottom_disciplines_dist)
        },
        'sub_disciplines': {
            'top': calculate_distribution(top_sub_disciplines_dist, True),
            'bottom': calculate_distribution(bottom_sub_disciplines_dist, True)
        }
    }