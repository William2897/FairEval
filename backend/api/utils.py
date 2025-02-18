from django.db.models import Avg, Count, F
from django.utils import timezone
from datetime import timedelta
from .models import Rating, Sentiment, Professor
from collections import Counter
from itertools import chain
from scipy import stats
import pandas as pd

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
    
    # Get word frequencies with gender information
    sentiments_with_gender = sentiments.annotate(
        gender=F('professor__gender')
    ).filter(gender__in=['Male', 'Female'])
    
    male_sentiments = sentiments_with_gender.filter(gender='Male')
    female_sentiments = sentiments_with_gender.filter(gender='Female')
    
    # Get gender-specific term frequencies for both VADER and LEXICON
    def get_term_frequencies(queryset, terms_field):
        terms = chain.from_iterable(s[terms_field] for s in queryset if s[terms_field])
        return Counter(terms)
    
    # Get frequencies for both VADER and LEXICON terms
    male_pos_lexicon = get_term_frequencies(male_sentiments.values('positive_terms_lexicon'), 'positive_terms_lexicon')
    male_neg_lexicon = get_term_frequencies(male_sentiments.values('negative_terms_lexicon'), 'negative_terms_lexicon')
    female_pos_lexicon = get_term_frequencies(female_sentiments.values('positive_terms_lexicon'), 'positive_terms_lexicon')
    female_neg_lexicon = get_term_frequencies(female_sentiments.values('negative_terms_lexicon'), 'negative_terms_lexicon')
    
    male_pos_vader = get_term_frequencies(male_sentiments.values('positive_terms_vader'), 'positive_terms_vader')
    male_neg_vader = get_term_frequencies(male_sentiments.values('negative_terms_vader'), 'negative_terms_vader')
    female_pos_vader = get_term_frequencies(female_sentiments.values('positive_terms_vader'), 'positive_terms_vader')
    female_neg_vader = get_term_frequencies(female_sentiments.values('negative_terms_vader'), 'negative_terms_vader')
    
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
            
            if male_rel_freq > bias_threshold * female_rel_freq:
                bias = 'Male'
            elif female_rel_freq > bias_threshold * male_rel_freq:
                bias = 'Female'
            else:
                continue
                
            result.append({
                'term': term,
                'male_freq': male_freq,
                'female_freq': female_freq,
                'bias': bias
            })
        return sorted(result, key=lambda x: max(x['male_freq'], x['female_freq']), reverse=True)[:20]
    
    # Calculate term frequencies for all sentiments
    all_pos_lexicon = Counter()
    all_neg_lexicon = Counter()
    all_pos_vader = Counter()
    all_neg_vader = Counter()
    
    for s in sentiments:
        if s.positive_terms_lexicon:
            all_pos_lexicon.update(s.positive_terms_lexicon)
        if s.negative_terms_lexicon:
            all_neg_lexicon.update(s.negative_terms_lexicon)
        if s.positive_terms_vader:
            all_pos_vader.update(s.positive_terms_vader)
        if s.negative_terms_vader:
            all_neg_vader.update(s.negative_terms_vader)
    
    summary = {
        'total_comments': sentiments.count(),
        'sentiment_breakdown': {
            'positive': sentiments.filter(sentiment=1).count(),
            'negative': sentiments.filter(sentiment=0).count()
        },
        'top_words': {
            'lexicon': {
                'positive': [{'word': word, 'count': count} 
                           for word, count in all_pos_lexicon.most_common(20)],
                'negative': [{'word': word, 'count': count} 
                           for word, count in all_neg_lexicon.most_common(20)]
            },
            'vader': {
                'positive': [{'word': word, 'count': count} 
                           for word, count in all_pos_vader.most_common(20)],
                'negative': [{'word': word, 'count': count} 
                           for word, count in all_neg_vader.most_common(20)]
            }
        },
        'gender_analysis': {
            'lexicon': {
                'positive_terms': calculate_gender_bias(male_pos_lexicon, female_pos_lexicon),
                'negative_terms': calculate_gender_bias(male_neg_lexicon, female_neg_lexicon)
            },
            'vader': {
                'positive_terms': calculate_gender_bias(male_pos_vader, female_pos_vader),
                'negative_terms': calculate_gender_bias(male_neg_vader, female_neg_vader)
            }
        },
        'recent_sentiments': list(
            sentiments.order_by('-created_at')[:5]
            .values('comment', 'processed_comment', 'sentiment', 'created_at')
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
        field_name = 'sub_discipline' if is_sub_discipline else 'discipline'
        for item in queryset:
            total = item['total'] or 1  # Avoid division by zero
            distributions.append({
                field_name: item[field_name],
                'total': item['total'],
                'female_count': item['female_count'],
                'male_count': item['male_count'],
                'female_percent': round((item['female_count'] * 100.0) / total, 2),
                'male_percent': round((item['male_count'] * 100.0) / total, 2),
                'rating': item['avg_rating']
            })
        # Sort by rating - will be used in descending order for top, ascending for bottom
        return sorted(distributions, key=lambda x: x['rating'] or 0)
    
    # Calculate gender distribution for disciplines with ratings
    top_disciplines_dist = Professor.objects.filter(
        discipline__in=top_3_disciplines
    ).values('discipline').annotate(
        total=Count('id'),
        female_count=Count(Case(When(gender='Female', then=1))),
        male_count=Count(Case(When(gender='Male', then=1))),
        avg_rating=Avg('ratings__avg_rating')
    )
    
    bottom_disciplines_dist = Professor.objects.filter(
        discipline__in=bottom_3_disciplines
    ).values('discipline').annotate(
        total=Count('id'),
        female_count=Count(Case(When(gender='Female', then=1))),
        male_count=Count(Case(When(gender='Male', then=1))),
        avg_rating=Avg('ratings__avg_rating')
    )
    
    # Calculate gender distribution for sub-disciplines with ratings
    top_sub_disciplines_dist = Professor.objects.filter(
        sub_discipline__in=top_10_sub_disciplines
    ).values('sub_discipline').annotate(
        total=Count('id'),
        female_count=Count(Case(When(gender='Female', then=1))),
        male_count=Count(Case(When(gender='Male', then=1))),
        avg_rating=Avg('ratings__avg_rating')
    )
    
    bottom_sub_disciplines_dist = Professor.objects.filter(
        sub_discipline__in=bottom_10_sub_disciplines
    ).values('sub_discipline').annotate(
        total=Count('id'),
        female_count=Count(Case(When(gender='Female', then=1))),
        male_count=Count(Case(When(gender='Male', then=1))),
        avg_rating=Avg('ratings__avg_rating')
    )
    
    # Sort the distributions appropriately
    top_disciplines = calculate_distribution(top_disciplines_dist)
    bottom_disciplines = calculate_distribution(bottom_disciplines_dist)
    top_sub_disciplines = calculate_distribution(top_sub_disciplines_dist, True)
    bottom_sub_disciplines = calculate_distribution(bottom_sub_disciplines_dist, True)
    
    return {
        'total_stats': total_stats,
        'disciplines': {
            'top': list(reversed(top_disciplines)),  # Reverse for descending order
            'bottom': bottom_disciplines  # Keep ascending order
        },
        'sub_disciplines': {
            'top': list(reversed(top_sub_disciplines)),  # Reverse for descending order
            'bottom': bottom_sub_disciplines  # Keep ascending order
        }
    }

def calculate_tukey_hsd():
    """Calculate Tukey's HSD test results for disciplines and sub-disciplines"""
    try:
        from scipy import stats
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        import pandas as pd
        import numpy as np
        
        # Fetch ratings data with professor gender info
        ratings_data = Rating.objects.select_related('professor').values(
            'avg_rating', 
            'professor__gender',
            'professor__discipline',
            'professor__sub_discipline'
        ).exclude(
            professor__gender__isnull=True
        ).exclude(
            professor__discipline__isnull=True
        ).exclude(
            avg_rating__isnull=True
        )
        
        if not ratings_data:
            raise ValueError("No valid ratings data found for analysis")
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(list(ratings_data))
        df.columns = ['avg_rating', 'gender', 'discipline', 'sub_discipline']
        
        # Create combined columns for gender-discipline interaction
        df['gender_discipline'] = df['gender'] + ' - ' + df['discipline']
        df['gender_sub_discipline'] = df.apply(
            lambda x: f"{x['gender']} - {x['sub_discipline']}" if pd.notnull(x['sub_discipline']) else None,
            axis=1
        )
        
        # Perform Tukey's HSD test for disciplines
        discipline_groups = df.groupby('gender_discipline')['avg_rating'].agg(['count', 'mean'])
        valid_disciplines = discipline_groups[discipline_groups['count'] >= 2].index
        
        df_filtered = df[df['gender_discipline'].isin(valid_disciplines)]
        if len(df_filtered) < 2:
            raise ValueError("Insufficient data for discipline analysis")
            
        discipline_results = pairwise_tukeyhsd(
            df_filtered['avg_rating'], 
            df_filtered['gender_discipline']
        )
        reject = discipline_results.reject  # Boolean array of rejected hypotheses

        
        # Filter discipline results for same-discipline comparisons
        discipline_comparisons = []
        for i, comp in enumerate(discipline_results.summary().data[1:]):
            if comp[0].split(' - ')[1] == comp[1].split(' - ')[1]:
                discipline_comparisons.append({
                    'group1': comp[0],
                    'group2': comp[1],
                    'meandiff': float(comp[2]),
                    'lower': float(comp[3]),
                    'upper': float(comp[4]),
                    'p_adj': float(comp[5]),
                    'reject': comp[6] == 'True'
                })
        
        # Perform Tukey's HSD test for sub-disciplines
        sub_df = df.dropna(subset=['sub_discipline'])
        sub_discipline_comparisons = []
        
        if not sub_df.empty:
            sub_discipline_groups = sub_df.groupby('gender_sub_discipline')['avg_rating'].agg(['count', 'mean'])
            valid_sub_disciplines = sub_discipline_groups[sub_discipline_groups['count'] >= 2].index
            
            sub_df_filtered = sub_df[sub_df['gender_sub_discipline'].isin(valid_sub_disciplines)]
            
            if len(sub_df_filtered) >= 2:
                sub_discipline_results = pairwise_tukeyhsd(
                    sub_df_filtered['avg_rating'], 
                    sub_df_filtered['gender_sub_discipline']
                )
                
                # Filter sub-discipline results
                for i, comp in enumerate(sub_discipline_results.summary().data[1:]):
                    if comp[0].split(' - ')[1] == comp[1].split(' - ')[1]:
                        sub_discipline_comparisons.append({
                            'group1': comp[0],
                            'group2': comp[1],
                            'meandiff': float(comp[2]),
                            'lower': float(comp[3]),
                            'upper': float(comp[4]),
                            'p_adj': float(comp[5]),
                            'reject': comp[6] == 'True'
                        })
        
        return {
            'discipline_comparisons': discipline_comparisons,
            'sub_discipline_comparisons': sub_discipline_comparisons
        }
        
    except Exception as e:
        import traceback
        print(f"Error in calculate_tukey_hsd: {str(e)}")
        print(traceback.format_exc())
        raise

def calculate_gender_discipline_heatmap():
    """Calculate average ratings by gender and discipline for heatmap visualization"""
    from django.db.models import Avg
    
    heatmap_data = Rating.objects.select_related('professor').values(
        'professor__gender', 
        'professor__discipline'
    ).annotate(
        avg_rating=Avg('avg_rating')
    ).filter(
        professor__gender__isnull=False,
        professor__discipline__isnull=False
    )
    
    # Transform into the format needed for the heatmap
    result = []
    for entry in heatmap_data:
        result.append({
            'gender': entry['professor__gender'],
            'discipline': entry['professor__discipline'],
            'avg_rating': round(float(entry['avg_rating']), 2) if entry['avg_rating'] else 0
        })
    
    return result