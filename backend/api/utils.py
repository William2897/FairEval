from django.db.models import Avg, Count, F, Value, CharField, Q
from django.db.models.expressions import RawSQL
from django.utils import timezone
from django.db import connection  # Add connection import
from datetime import timedelta
from .models import Rating, Sentiment, Professor
from collections import Counter
from itertools import chain
from scipy import stats

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

def get_sentiment_summary(professor_id=None, institution=False, page=1, page_size=10):
    """Get sentiment analysis summary for a professor or institution-wide"""    
    if institution:
        sentiments = Sentiment.objects.all()
    else:
        sentiments = Sentiment.objects.filter(professor_id=professor_id)
    
    # Get word frequencies with gender information using SQL
    sentiments_with_gender = sentiments.annotate(
        gender=F('professor__gender')
    ).filter(gender__in=['Male', 'Female'])
    
    male_sentiments = sentiments_with_gender.filter(gender='Male')
    female_sentiments = sentiments_with_gender.filter(gender='Female')
    
    # Get gender-specific term frequencies using SQL Unnest for both VADER and LEXICON
    def get_term_frequencies_sql(queryset, terms_field):
        """Use standard PostgreSQL syntax to count term frequencies from ArrayField"""
        if not queryset.exists():
            return {}
            
        # Get all professor IDs from the queryset
        professor_ids = list(queryset.values_list('professor_id', flat=True))
        
        if not professor_ids:
            return {}
        
        # Use parameters instead of embedding the queryset SQL
        query = f"""
            SELECT t.term, COUNT(*) as count
            FROM api_sentiment s
            JOIN api_professor p ON CAST(s.professor_id AS VARCHAR) = p.professor_id
            CROSS JOIN LATERAL jsonb_array_elements_text(s.{terms_field}) AS t(term)
            WHERE t.term IS NOT NULL
            AND s.professor_id IN %s
            GROUP BY t.term
            ORDER BY count DESC
        """
        
        with connection.cursor() as cursor:
            cursor.execute(query, [tuple(professor_ids)])
            results = cursor.fetchall()
        
        # Convert results to dictionary
        return {term: count for term, count in results}
    
    # Get frequencies for both VADER and LEXICON terms using SQL
    male_pos_lexicon = get_term_frequencies_sql(male_sentiments, 'positive_terms_lexicon')
    male_neg_lexicon = get_term_frequencies_sql(male_sentiments, 'negative_terms_lexicon')
    female_pos_lexicon = get_term_frequencies_sql(female_sentiments, 'positive_terms_lexicon')
    female_neg_lexicon = get_term_frequencies_sql(female_sentiments, 'negative_terms_lexicon')
    
    male_pos_vader = get_term_frequencies_sql(male_sentiments, 'positive_terms_vader')
    male_neg_vader = get_term_frequencies_sql(male_sentiments, 'negative_terms_vader')
    female_pos_vader = get_term_frequencies_sql(female_sentiments, 'positive_terms_vader')
    female_neg_vader = get_term_frequencies_sql(female_sentiments, 'negative_terms_vader')
    
    def calculate_gender_bias(male_counter, female_counter, bias_threshold=1.1):
        """Calculate gender bias in term usage"""
        # Convert dictionary counters to Counter objects for easy manipulation
        male_counter = Counter(male_counter)
        female_counter = Counter(female_counter)
        
        # Calculate totals for relative frequencies
        male_total = sum(male_counter.values()) or 1
        female_total = sum(female_counter.values()) or 1
        
        # Get all unique terms
        all_terms = set(male_counter.keys()) | set(female_counter.keys())
        
        # Calculate relative frequencies and bias
        terms_data = []
        
        for term in all_terms:
            male_freq = male_counter[term]
            female_freq = female_counter[term]
            
            # Calculate relative frequencies
            male_rel_freq = male_freq / male_total
            female_rel_freq = female_freq / female_total
            
            # Only include terms that appear more than once total
            if male_freq + female_freq < 2:
                continue
            
            # Determine bias
            if male_rel_freq > bias_threshold * female_rel_freq:
                bias = 'Male'
            elif female_rel_freq > bias_threshold * male_rel_freq:
                bias = 'Female'
            else:
                bias = 'Neutral'
            
            terms_data.append({
                'term': term,
                'male_freq': male_freq,
                'female_freq': female_freq,
                'male_rel_freq': male_rel_freq,
                'female_rel_freq': female_rel_freq,
                'bias': bias,
                'total_freq': male_freq + female_freq
            })
        
        return terms_data
    
    # Calculate term frequencies for all sentiments
    all_pos_lexicon = Counter()
    all_neg_lexicon = Counter()
    all_pos_vader = Counter()
    all_neg_vader = Counter()
    
    for s in sentiments:
        # Add term frequencies to counters
        if s.positive_terms_lexicon:
            all_pos_lexicon.update(s.positive_terms_lexicon)
        if s.negative_terms_lexicon:
            all_neg_lexicon.update(s.negative_terms_lexicon)
        if s.positive_terms_vader:
            all_pos_vader.update(s.positive_terms_vader)
        if s.negative_terms_vader:
            all_neg_vader.update(s.negative_terms_vader)
    
    # Add paginated comments to the response
    total_comments = sentiments.count()
    total_pages = (total_comments + page_size - 1) // page_size  # Calculate total pages
    
    # Get paginated comments
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_comments = list(sentiments.order_by('-created_at')[start_idx:end_idx].values(
        'comment', 'processed_comment', 'sentiment', 'created_at'
    ))
    
    summary = {
        'total_comments': total_comments,
        'total_pages': total_pages,
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
        ),
        'comments': paginated_comments
    }
    
    return summary

def get_term_frequencies_institutional(sentiment_type='vader', term_type='positive', gender=None):
    """Get term frequencies at institutional level using standard PostgreSQL"""
    # Determine which field to use based on sentiment_type and term_type
    if sentiment_type == 'vader':
        field = 'positive_terms_vader' if term_type == 'positive' else 'negative_terms_vader'
    else:
        field = 'positive_terms_lexicon' if term_type == 'positive' else 'negative_terms_lexicon'
    
    # Build the gender filter condition
    gender_condition = "AND p.gender = %s" if gender else "AND p.gender IN ('Male', 'Female')"
    gender_params = [gender] if gender else []
    
    # Use standard PostgreSQL query with jsonb_array_elements_text
    query = f"""
        SELECT t.term, COUNT(*) as count
        FROM api_sentiment s
        JOIN api_professor p ON CAST(s.professor_id AS VARCHAR) = p.professor_id
        CROSS JOIN LATERAL jsonb_array_elements_text(s.{field}) AS t(term)
        WHERE t.term IS NOT NULL
        {gender_condition}
        GROUP BY t.term
        ORDER BY count DESC
    """
    
    # Execute the raw query
    with connection.cursor() as cursor:
        cursor.execute(query, gender_params)
        results = cursor.fetchall()
    
    # Convert results to list of dicts
    return [{'id': 1, 'term': term, 'count': count} for term, count in results]

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
            # Safely calculate percentages
            total = item['total'] or 1  # Avoid division by zero
            female_percent = round((item['female_count'] * 100.0) / total, 2)
            male_percent = round((item['male_count'] * 100.0) / total, 2)
            
            distributions.append({
                field_name: item[field_name],
                'total': item['total'],
                'female_count': item['female_count'],
                'male_count': item['male_count'],
                'female_percent': female_percent,
                'male_percent': male_percent,
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
        
        # Use raw SQL to fetch the data efficiently
        with connection.cursor() as cursor:
            # Query for disciplines
            cursor.execute("""
                WITH gender_discipline_avg AS (
                    SELECT 
                        p.gender,
                        p.discipline,
                        r.avg_rating,
                        CONCAT(p.gender, ' - ', p.discipline) as gender_discipline
                    FROM api_rating r
                    JOIN api_professor p ON CAST(r.professor_id AS VARCHAR) = p.professor_id
                    WHERE p.gender IN ('Male', 'Female')
                    AND p.discipline IS NOT NULL
                    AND r.avg_rating IS NOT NULL
                )
                SELECT 
                    gender_discipline,
                    gender,
                    discipline,
                    avg_rating
                FROM gender_discipline_avg
                WHERE gender_discipline IN (
                    SELECT gender_discipline
                    FROM gender_discipline_avg
                    GROUP BY gender_discipline
                    HAVING COUNT(*) >= 2
                )
                ORDER BY discipline, gender
            """)
            discipline_results = cursor.fetchall()
            
            # Query for sub-disciplines
            cursor.execute("""
                WITH gender_subdiscipline_avg AS (
                    SELECT 
                        p.gender,
                        p.sub_discipline,
                        r.avg_rating,
                        CONCAT(p.gender, ' - ', p.sub_discipline) as gender_subdiscipline
                    FROM api_rating r
                    JOIN api_professor p ON CAST(r.professor_id AS VARCHAR) = p.professor_id
                    WHERE p.gender IN ('Male', 'Female')
                    AND p.sub_discipline IS NOT NULL
                    AND r.avg_rating IS NOT NULL
                )
                SELECT 
                    gender_subdiscipline,
                    gender,
                    sub_discipline,
                    avg_rating
                FROM gender_subdiscipline_avg
                WHERE gender_subdiscipline IN (
                    SELECT gender_subdiscipline
                    FROM gender_subdiscipline_avg
                    GROUP BY gender_subdiscipline
                    HAVING COUNT(*) >= 2
                )
                ORDER BY sub_discipline, gender
            """)
            subdiscipline_results = cursor.fetchall()
        
        # Process discipline results
        if not discipline_results:
            raise ValueError("No valid ratings data found for discipline analysis")
            
        discipline_df = pd.DataFrame(discipline_results, columns=['gender_discipline', 'gender', 'discipline', 'avg_rating'])
        discipline_tukey = pairwise_tukeyhsd(
            discipline_df['avg_rating'],
            discipline_df['gender_discipline']
        )
        
        # Filter discipline results for same-discipline comparisons
        discipline_comparisons = []
        for i, comp in enumerate(discipline_tukey.summary().data[1:]):
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
        
        # Process sub-discipline results
        subdiscipline_comparisons = []
        if subdiscipline_results:
            subdiscipline_df = pd.DataFrame(subdiscipline_results, columns=['gender_subdiscipline', 'gender', 'sub_discipline', 'avg_rating'])
            subdiscipline_tukey = pairwise_tukeyhsd(
                subdiscipline_df['avg_rating'],
                subdiscipline_df['gender_subdiscipline']
            )
            
            # Filter sub-discipline results
            for i, comp in enumerate(subdiscipline_tukey.summary().data[1:]):
                if comp[0].split(' - ')[1] == comp[1].split(' - ')[1]:
                    subdiscipline_comparisons.append({
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
            'sub_discipline_comparisons': subdiscipline_comparisons
        }
        
    except Exception as e:
        import traceback
        print(f"Error in calculate_tukey_hsd: {str(e)}")
        print(traceback.format_exc())
        raise

def calculate_gender_discipline_heatmap():
    """Calculate average ratings by gender and discipline for heatmap visualization"""
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT 
                p.gender as gender,
                p.discipline as discipline,
                ROUND(AVG(r.avg_rating)::numeric, 2) as avg_rating
            FROM api_rating r
            JOIN api_professor p ON CAST(r.professor_id AS VARCHAR) = p.professor_id
            WHERE p.gender IS NOT NULL 
            AND p.discipline IS NOT NULL
            GROUP BY p.gender, p.discipline
            ORDER BY p.discipline, p.gender
        """)
        results = cursor.fetchall()
    
    # Transform into the format needed for the heatmap
    return [
        {
            'gender': gender,
            'discipline': discipline,
            'avg_rating': float(avg_rating) if avg_rating else 0
        }
        for gender, discipline, avg_rating in results
    ]