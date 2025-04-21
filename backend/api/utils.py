# backend/api/utils.py

from django.db.models import Avg, Count, F, Case, When, FloatField
from django.db.models.functions import Cast
from django.utils import timezone
from django.db import connection
from datetime import timedelta
# Ensure models are imported correctly based on your project structure
try:
    from .models import Rating, Sentiment, Professor
except ImportError:
    # Handle potential circular import or path issues if utils is imported elsewhere
    print("Warning: Could not import models directly in utils.py")
    Rating, Sentiment, Professor = None, None, None

from collections import Counter
from itertools import chain
import pandas as pd # Needed for Tukey HSD
import numpy as np  # Needed for Tukey HSD
# Scipy/Statsmodels are needed for Tukey HSD - ensure they are installed
try:
    from scipy import stats
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
except ImportError:
    print("Warning: statsmodels or scipy not installed. Tukey HSD analysis will not work.")
    pairwise_tukeyhsd = None # Set to None if not available


def calculate_professor_metrics(professor_id):
    """Calculate aggregate metrics for a professor"""
    # Import model here if initial import failed
    global Professor, Rating
    if Rating is None:
        from .models import Rating
    if Professor is None:
        from .models import Professor

    # Ensure the professor exists before querying ratings
    if not Professor.objects.filter(professor_id=professor_id).exists():
        print(f"Warning: Professor {professor_id} not found for metrics calculation.")
        return { # Return default/empty metrics
             'avg_rating': 0, 'avg_helpful': 0, 'avg_clarity': 0,
             'avg_difficulty': 0, 'total_ratings': 0, 'trend': 0
        }

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

    # Ensure results are not None before subtraction
    recent_avg_val = recent_metrics.get('recent_avg') or 0
    current_avg_val = metrics.get('avg_rating') or 0
    metrics['trend'] = recent_avg_val - current_avg_val

    # Ensure all keys exist even if aggregation returns None
    metrics['avg_rating'] = metrics.get('avg_rating') or 0
    metrics['avg_helpful'] = metrics.get('avg_helpful') or 0
    metrics['avg_clarity'] = metrics.get('avg_clarity') or 0
    metrics['avg_difficulty'] = metrics.get('avg_difficulty') or 0
    metrics['total_ratings'] = metrics.get('total_ratings') or 0

    return metrics

def get_sentiment_summary(professor_id=None, institution=False, page=1, page_size=20, include_aggregate_stats=False):
    """Get sentiment analysis summary for a professor or institution-wide, including bias tags."""
    # Import models here if initial import failed
    global Professor, Sentiment
    if Sentiment is None:
        from .models import Sentiment
    if Professor is None:
        from .models import Professor

    if institution:
        sentiments_qs = Sentiment.objects.select_related('professor').all()
        print(f"Fetching institution-wide sentiment summary (Page {page})")
    elif professor_id:
        sentiments_qs = Sentiment.objects.select_related('professor').filter(professor_id=professor_id)
        print(f"Fetching sentiment summary for professor {professor_id} (Page {page})")
    else:
         print("Warning: get_sentiment_summary called without professor_id or institution=True")
         return { # Return empty structure
            'total_comments': 0, 'total_pages': 0, 'sentiment_breakdown': {'positive': 0, 'negative': 0},
            'top_words': {'lexicon': {'positive': [], 'negative': []}, 'vader': {'positive': [], 'negative': []}},
            'gender_analysis': {'lexicon': {'positive_terms': [], 'negative_terms': []}, 'vader': {'positive_terms': [], 'negative_terms': []}},
            'recent_sentiments': [], 'comments': []
        }

    # --- Pagination ---
    total_comments = sentiments_qs.count()
    total_pages = (total_comments + page_size - 1) // page_size if page_size > 0 else 1
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    # --- Fetch Paginated Comments INCLUDING BIAS FIELDS ---
    paginated_comments = list(sentiments_qs.order_by('-created_at')[start_idx:end_idx].values(
        'id', # Good practice for keys
        'comment',
        'processed_comment',
        'sentiment',
        'created_at',
        'confidence',
        # --- NEW BIAS FIELDS ADDED ---
        'bias_tag',
        'bias_interpretation',
        'stereotype_bias_score',
        'objective_focus_percentage'
        # --- END NEW BIAS FIELDS ---
    ))
    print(f"Fetched {len(paginated_comments)} comments for page {page}.")

    # --- Calculate Aggregate Bias Statistics if requested ---
    aggregate_bias_stats = None
    if include_aggregate_stats:
        print("Calculating aggregate bias statistics...")
        
        # Function to calculate bias distribution for a specific sentiment value
        def calculate_bias_distribution(sentiment_value):
            # Get all comments with this sentiment (not just paginated ones)
            sentiment_comments = list(sentiments_qs.filter(sentiment=sentiment_value).values('bias_tag'))
            
            # Count occurrences of each bias tag
            tag_counts = {}
            total_comments = len(sentiment_comments)
            
            for comment in sentiment_comments:
                tag = comment['bias_tag'] or 'UNKNOWN'
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            # Calculate percentages
            bias_distribution = {}
            for tag, count in tag_counts.items():
                percentage = (count / total_comments * 100) if total_comments > 0 else 0
                bias_distribution[tag] = {
                    'count': count,
                    'percentage': f"{percentage:.1f}"
                }
            
            return bias_distribution
        
        # Calculate distribution for both positive and negative sentiments
        aggregate_bias_stats = {
            'positive': {
                'bias_distribution': calculate_bias_distribution(1)  # 1 = positive sentiment
            },
            'negative': {
                'bias_distribution': calculate_bias_distribution(0)  # 0 = negative sentiment
            }
        }
        
        print("Aggregate bias statistics calculation complete.")

    # --- Calculate Sentiment Breakdown ---
    sentiment_breakdown = sentiments_qs.values('sentiment').annotate(count=Count('id'))
    breakdown_dict = {item['sentiment']: item['count'] for item in sentiment_breakdown if item['sentiment'] is not None}

    # --- Calculate Top Words (Lexicon & VADER) ---
    # This part can be slow for large datasets if not optimized or cached.
    # It iterates through *all* sentiments in the queryset, not just the paginated ones.
    # Consider calculating this less frequently or caching if performance is an issue.
    print("Calculating top words...")
    all_pos_lexicon = Counter()
    all_neg_lexicon = Counter()
    all_pos_vader = Counter()
    all_neg_vader = Counter()

    # Fetch relevant fields efficiently
    term_data = sentiments_qs.values(
        'positive_terms_lexicon', 'negative_terms_lexicon',
        'positive_terms_vader', 'negative_terms_vader'
    )
    for s in term_data:
        if s.get('positive_terms_lexicon'): all_pos_lexicon.update(s['positive_terms_lexicon'])
        if s.get('negative_terms_lexicon'): all_neg_lexicon.update(s['negative_terms_lexicon'])
        if s.get('positive_terms_vader'): all_pos_vader.update(s['positive_terms_vader'])
        if s.get('negative_terms_vader'): all_neg_vader.update(s['negative_terms_vader'])
    print("Top word calculation done.")


    # --- Gender Analysis Calculations (Also potentially slow for large datasets) ---
    print("Calculating gender analysis...")
    sentiments_with_gender = sentiments_qs.filter(professor__gender__in=['Male', 'Female'])
    male_sentiments_qs = sentiments_with_gender.filter(professor__gender='Male')
    female_sentiments_qs = sentiments_with_gender.filter(professor__gender='Female')

    def get_term_frequencies_optimized(queryset, terms_field):
        """Optimized term frequency counter using database values_list."""
        term_lists = queryset.values_list(terms_field, flat=True)
        counter = Counter()
        for term_list in term_lists:
            if term_list: # Check if the list is not None or empty
                counter.update(term_list)
        return counter

    male_pos_lexicon_counts = get_term_frequencies_optimized(male_sentiments_qs, 'positive_terms_lexicon')
    male_neg_lexicon_counts = get_term_frequencies_optimized(male_sentiments_qs, 'negative_terms_lexicon')
    female_pos_lexicon_counts = get_term_frequencies_optimized(female_sentiments_qs, 'positive_terms_lexicon')
    female_neg_lexicon_counts = get_term_frequencies_optimized(female_sentiments_qs, 'negative_terms_lexicon')

    male_pos_vader_counts = get_term_frequencies_optimized(male_sentiments_qs, 'positive_terms_vader')
    male_neg_vader_counts = get_term_frequencies_optimized(male_sentiments_qs, 'negative_terms_vader')
    female_pos_vader_counts = get_term_frequencies_optimized(female_sentiments_qs, 'positive_terms_vader')
    female_neg_vader_counts = get_term_frequencies_optimized(female_sentiments_qs, 'negative_terms_vader')


    def calculate_gender_bias(male_counter, female_counter, bias_threshold=1.1):
        """Calculate gender bias in term usage (no change in logic)."""
        male_total = sum(male_counter.values()) or 1
        female_total = sum(female_counter.values()) or 1
        all_terms = set(male_counter.keys()) | set(female_counter.keys())
        terms_data = []
        for term in all_terms:
            male_freq = male_counter.get(term, 0)
            female_freq = female_counter.get(term, 0)
            if male_freq + female_freq < 2: continue
            male_rel_freq = male_freq / male_total
            female_rel_freq = female_freq / female_total
            if male_rel_freq > bias_threshold * female_rel_freq: bias = 'Male'
            elif female_rel_freq > bias_threshold * male_rel_freq: bias = 'Female'
            else: bias = 'Neutral'
            terms_data.append({
                'term': term, 'male_freq': male_freq, 'female_freq': female_freq,
                'male_rel_freq': male_rel_freq, 'female_rel_freq': female_rel_freq,
                'bias': bias, 'total_freq': male_freq + female_freq
            })
        # Sort by total frequency for relevance
        terms_data.sort(key=lambda x: x['total_freq'], reverse=True)
        return terms_data[:50] # Limit results for performance

    print("Gender analysis calculation done.")    # --- Assemble Final Summary ---
    summary = {
        'total_comments': total_comments,
        'total_pages': total_pages,
        'current_page': page, # Good to include current page
        'sentiment_breakdown': {
            'positive': breakdown_dict.get(1, 0),
            'negative': breakdown_dict.get(0, 0)
        },
        'top_words': {
            'lexicon': {
                'positive': [{'word': word, 'count': count} for word, count in all_pos_lexicon.most_common(20)],
                'negative': [{'word': word, 'count': count} for word, count in all_neg_lexicon.most_common(20)]
            },
            'vader': {
                'positive': [{'word': word, 'count': count} for word, count in all_pos_vader.most_common(20)],
                'negative': [{'word': word, 'count': count} for word, count in all_neg_vader.most_common(20)]
            }
        },
        'gender_analysis': { # This structure assumes you want the calculated bias list
            'lexicon': {
                'positive_terms': calculate_gender_bias(male_pos_lexicon_counts, female_pos_lexicon_counts),
                'negative_terms': calculate_gender_bias(male_neg_lexicon_counts, female_neg_lexicon_counts)
            },
            'vader': {
                'positive_terms': calculate_gender_bias(male_pos_vader_counts, female_pos_vader_counts),
                'negative_terms': calculate_gender_bias(male_neg_vader_counts, female_neg_vader_counts)
            }
        },
        # Recent sentiments might not need bias fields unless UI uses them
        # 'recent_sentiments': list(
        #     sentiments_qs.order_by('-created_at')[:5]
        #     .values( # Add bias fields here if needed by UI for recent comments display
        #           'comment', 'processed_comment', 'sentiment', 'created_at', 'bias_tag'
        #      )
        # ),
        'comments': paginated_comments # This list now contains the bias fields
    }
    
    # Include aggregate bias statistics if requested
    if include_aggregate_stats and aggregate_bias_stats:
        summary['aggregate_bias_stats'] = aggregate_bias_stats

    return summary


# --- REST OF UTILS.PY (No changes needed to the functions below for this specific feature) ---

def get_term_frequencies_institutional(sentiment_type='vader', term_type='positive', gender=None):
    """Get term frequencies at institutional level using standard PostgreSQL"""
    # Import model here if initial import failed
    global Professor, Sentiment
    if Sentiment is None: from .models import Sentiment
    if Professor is None: from .models import Professor

    if sentiment_type == 'vader':
        field = 'positive_terms_vader' if term_type == 'positive' else 'negative_terms_vader'
    else:
        field = 'positive_terms_lexicon' if term_type == 'positive' else 'negative_terms_lexicon'

    gender_condition = "AND p.gender = %s" if gender else "AND p.gender IN ('Male', 'Female')"
    gender_params = [gender] if gender else []

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
    with connection.cursor() as cursor:
        cursor.execute(query, gender_params)
        results = cursor.fetchall()
    return [{'id': i+1, 'term': term, 'count': count} for i, (term, count) in enumerate(results)]


def generate_recommendations(professor_id):
    """Generate teaching improvement recommendations based on ratings and comments"""
    # Import model here if initial import failed
    global Professor, Rating
    if Rating is None: from .models import Rating
    if Professor is None: from .models import Professor

    metrics = calculate_professor_metrics(professor_id) # Already handles non-existent prof
    # Need sentiment breakdown, get minimal data
    sentiments_qs = Sentiment.objects.filter(professor_id=professor_id)
    sentiment_breakdown = sentiments_qs.values('sentiment').annotate(count=Count('id'))
    breakdown_dict = {item['sentiment']: item['count'] for item in sentiment_breakdown if item['sentiment'] is not None}
    positive_count = breakdown_dict.get(1, 0)
    negative_count = breakdown_dict.get(0, 0)

    recommendations = {
        'teaching_effectiveness': {'score': metrics['avg_rating'], 'recommendations': []},
        'clarity': {'score': metrics['avg_clarity'], 'recommendations': []},
        'workload': {'score': metrics['avg_difficulty'], 'recommendations': []}
    }

    if metrics['avg_clarity'] < 4.0:
        recommendations['clarity']['recommendations'].append({
            'id': 1, 'text': 'Consider providing more detailed explanations and examples',
            'priority': 'high', 'impact_score': 8.5, 'supporting_ratings': metrics['total_ratings']
        })
    if negative_count > positive_count:
        recommendations['teaching_effectiveness']['recommendations'].append({
            'id': 2, 'text': 'Review and address common themes in negative feedback',
            'priority': 'high', 'impact_score': 9.0, 'supporting_ratings': negative_count
        })
    # Add more recommendations based on other metrics or patterns if desired
    if metrics['avg_helpful'] < 3.5:
         recommendations['teaching_effectiveness']['recommendations'].append({
             'id': 3, 'text': 'Explore ways to enhance student support (e.g., office hours, clearer guidance).',
             'priority': 'medium', 'impact_score': 7.0, 'supporting_ratings': metrics['total_ratings']
         })
    if metrics['avg_difficulty'] > 4.0:
         recommendations['workload']['recommendations'].append({
             'id': 4, 'text': 'Review course workload and difficulty balance to ensure it is challenging but manageable.',
             'priority': 'medium', 'impact_score': 6.5, 'supporting_ratings': metrics['total_ratings']
         })

    return recommendations

def analyze_discipline_ratings():
    """Analyze ratings across disciplines"""
    global Professor, Rating
    if Rating is None: from .models import Rating
    if Professor is None: from .models import Professor

    discipline_stats = Professor.objects.values('discipline', 'sub_discipline').annotate(
        avg_rating=Avg('ratings__avg_rating'),
        total_ratings=Count('ratings'),
        professor_count=Count('id', distinct=True)
    ).exclude(discipline__isnull=True).order_by('discipline', 'sub_discipline')
    return list(discipline_stats)

def analyze_discipline_gender_distribution():
    """Analyze discipline ratings by gender"""
    global Professor, Rating
    if Rating is None: from .models import Rating
    if Professor is None: from .models import Professor

    gender_discipline_stats = Professor.objects.filter(
        gender__in=['Male', 'Female'] # Ensure valid genders
    ).values(
        'discipline', 'sub_discipline', 'gender'
    ).annotate(
        avg_rating=Avg('ratings__avg_rating'),
        total_ratings=Count('ratings')
    ).exclude(
        discipline__isnull=True
    ).order_by('discipline', 'sub_discipline', 'gender')
    return list(gender_discipline_stats)

def calculate_gender_distribution():
    """Calculate gender distribution for top/bottom rated disciplines and sub-disciplines"""
    global Professor, Rating
    if Rating is None: from .models import Rating
    if Professor is None: from .models import Professor

    total_stats = Professor.objects.aggregate(
        total=Count('id'),
        female_count=Count(Case(When(gender='Female', then=1))),
        male_count=Count(Case(When(gender='Male', then=1))),
    )
    total = total_stats['total'] or 1
    total_stats['female_percent'] = round((total_stats['female_count'] * 100.0) / total, 2)
    total_stats['male_percent'] = round((total_stats['male_count'] * 100.0) / total, 2)

    # Calculate average ratings by discipline
    discipline_stats = Professor.objects.values('discipline').annotate(
        avg_rating=Avg('ratings__avg_rating')
    ).exclude(discipline__isnull=True).order_by('-avg_rating')

    top_3_disciplines = list(discipline_stats[:3].values_list('discipline', flat=True))
    bottom_3_disciplines = list(discipline_stats.reverse()[:3].values_list('discipline', flat=True))

    # Calculate average ratings by sub-discipline
    sub_discipline_stats = Professor.objects.values('sub_discipline').annotate(
        avg_rating=Avg('ratings__avg_rating')
    ).exclude(sub_discipline__isnull=True).order_by('-avg_rating')

    top_10_sub_disciplines = list(sub_discipline_stats[:10].values_list('sub_discipline', flat=True))
    bottom_10_sub_disciplines = list(sub_discipline_stats.reverse()[:10].values_list('sub_discipline', flat=True))

    def calculate_dist_list(queryset, field_name):
        distributions = []
        for item in queryset:
            total = item['total'] or 1
            female_percent = round((item['female_count'] * 100.0) / total, 2)
            male_percent = round((item['male_count'] * 100.0) / total, 2)
            distributions.append({
                field_name: item[field_name], 'total': item['total'],
                'female_count': item['female_count'], 'male_count': item['male_count'],
                'female_percent': female_percent, 'male_percent': male_percent,
                'rating': item.get('avg_rating') # Use .get for safety
            })
        return sorted(distributions, key=lambda x: x['rating'] or 0) # Sort by rating

    # Use filter(gender__in=['Male', 'Female']) to be explicit
    base_qs = Professor.objects.filter(gender__in=['Male', 'Female'])

    top_disciplines_dist = base_qs.filter(discipline__in=top_3_disciplines).values('discipline').annotate(
        total=Count('id'), female_count=Count(Case(When(gender='Female', then=1))),
        male_count=Count(Case(When(gender='Male', then=1))), avg_rating=Avg('ratings__avg_rating')
    )
    bottom_disciplines_dist = base_qs.filter(discipline__in=bottom_3_disciplines).values('discipline').annotate(
        total=Count('id'), female_count=Count(Case(When(gender='Female', then=1))),
        male_count=Count(Case(When(gender='Male', then=1))), avg_rating=Avg('ratings__avg_rating')
    )
    top_sub_disciplines_dist = base_qs.filter(sub_discipline__in=top_10_sub_disciplines).values('sub_discipline').annotate(
        total=Count('id'), female_count=Count(Case(When(gender='Female', then=1))),
        male_count=Count(Case(When(gender='Male', then=1))), avg_rating=Avg('ratings__avg_rating')
    )
    bottom_sub_disciplines_dist = base_qs.filter(sub_discipline__in=bottom_10_sub_disciplines).values('sub_discipline').annotate(
        total=Count('id'), female_count=Count(Case(When(gender='Female', then=1))),
        male_count=Count(Case(When(gender='Male', then=1))), avg_rating=Avg('ratings__avg_rating')
    )

    top_disciplines = calculate_dist_list(top_disciplines_dist, 'discipline')
    bottom_disciplines = calculate_dist_list(bottom_disciplines_dist, 'discipline')
    top_sub_disciplines = calculate_dist_list(top_sub_disciplines_dist, 'sub_discipline')
    bottom_sub_disciplines = calculate_dist_list(bottom_sub_disciplines_dist, 'sub_discipline')

    return {
        'total_stats': total_stats,
        'disciplines': {'top': list(reversed(top_disciplines)), 'bottom': bottom_disciplines},
        'sub_disciplines': {'top': list(reversed(top_sub_disciplines)), 'bottom': bottom_sub_disciplines}
    }

def calculate_tukey_hsd():
    """Calculate Tukey's HSD test results for disciplines and sub-disciplines"""
    global Professor, Rating
    if Rating is None: from .models import Rating
    if Professor is None: from .models import Professor
    if pairwise_tukeyhsd is None: # Check if statsmodels is available
        print("ERROR: statsmodels not installed. Cannot perform Tukey HSD.")
        return {'discipline_comparisons': [], 'sub_discipline_comparisons': []}

    try:
        # Fetch data using Django ORM efficiently
        discipline_data = Rating.objects.select_related('professor').filter(
            professor__gender__in=['Male', 'Female'],
            professor__discipline__isnull=False,
            avg_rating__isnull=False
        ).values(
            'professor__gender',
            'professor__discipline',
            'avg_rating'
        )
        discipline_df = pd.DataFrame.from_records(discipline_data)
        if discipline_df.empty:
             print("Warning: No data for Tukey HSD discipline analysis.")
             discipline_comparisons = []
        else:
            discipline_df['group'] = discipline_df['professor__gender'] + ' - ' + discipline_df['professor__discipline']
            # Filter groups with enough data (e.g., >= 2 per group)
            group_counts = discipline_df['group'].value_counts()
            valid_groups = group_counts[group_counts >= 2].index
            filtered_discipline_df = discipline_df[discipline_df['group'].isin(valid_groups)]

            if len(filtered_discipline_df['group'].unique()) < 2:
                print("Warning: Not enough groups with sufficient data for discipline Tukey HSD.")
                discipline_comparisons = []
            else:
                discipline_tukey = pairwise_tukeyhsd(
                    filtered_discipline_df['avg_rating'],
                    filtered_discipline_df['group']
                )
                # Filter for same-discipline comparisons
                discipline_comparisons = []
                summary_df = pd.DataFrame(discipline_tukey.summary().data[1:], columns=discipline_tukey.summary().data[0])
                for _, row in summary_df.iterrows():
                    g1_discipline = row['group1'].split(' - ')[1]
                    g2_discipline = row['group2'].split(' - ')[1]
                    if g1_discipline == g2_discipline:
                        discipline_comparisons.append({
                            'group1': row['group1'], 'group2': row['group2'],
                            'meandiff': float(row['meandiff']), 'lower': float(row['lower']),
                            'upper': float(row['upper']), 'p_adj': float(row['p-adj']),
                            'reject': row['reject'] == True # Explicit bool conversion
                        })

        # Sub-discipline analysis (similar logic)
        subdiscipline_data = Rating.objects.select_related('professor').filter(
            professor__gender__in=['Male', 'Female'],
            professor__sub_discipline__isnull=False,
            avg_rating__isnull=False
        ).values(
            'professor__gender',
            'professor__sub_discipline',
            'avg_rating'
        )
        subdiscipline_df = pd.DataFrame.from_records(subdiscipline_data)
        if subdiscipline_df.empty:
             print("Warning: No data for Tukey HSD sub-discipline analysis.")
             subdiscipline_comparisons = []
        else:
            subdiscipline_df['group'] = subdiscipline_df['professor__gender'] + ' - ' + subdiscipline_df['professor__sub_discipline']
            group_counts_sub = subdiscipline_df['group'].value_counts()
            valid_groups_sub = group_counts_sub[group_counts_sub >= 2].index
            filtered_subdiscipline_df = subdiscipline_df[subdiscipline_df['group'].isin(valid_groups_sub)]

            if len(filtered_subdiscipline_df['group'].unique()) < 2:
                print("Warning: Not enough groups with sufficient data for sub-discipline Tukey HSD.")
                subdiscipline_comparisons = []
            else:
                subdiscipline_tukey = pairwise_tukeyhsd(
                    filtered_subdiscipline_df['avg_rating'],
                    filtered_subdiscipline_df['group']
                )
                # Filter for same-subdiscipline comparisons
                subdiscipline_comparisons = []
                summary_df_sub = pd.DataFrame(subdiscipline_tukey.summary().data[1:], columns=subdiscipline_tukey.summary().data[0])
                for _, row in summary_df_sub.iterrows():
                     g1_subdiscipline = row['group1'].split(' - ')[1]
                     g2_subdiscipline = row['group2'].split(' - ')[1]
                     if g1_subdiscipline == g2_subdiscipline:
                         subdiscipline_comparisons.append({
                             'group1': row['group1'], 'group2': row['group2'],
                             'meandiff': float(row['meandiff']), 'lower': float(row['lower']),
                             'upper': float(row['upper']), 'p_adj': float(row['p-adj']),
                             'reject': row['reject'] == True
                         })

        return {
            'discipline_comparisons': discipline_comparisons,
            'sub_discipline_comparisons': subdiscipline_comparisons
        }

    except Exception as e:
        import traceback
        print(f"Error in calculate_tukey_hsd: {str(e)}")
        print(traceback.format_exc())
        return {'discipline_comparisons': [], 'sub_discipline_comparisons': [], 'error': str(e)}


def calculate_gender_discipline_heatmap():
    """Calculate average ratings by gender and discipline for heatmap visualization"""
    global Professor, Rating
    if Rating is None: from .models import Rating
    if Professor is None: from .models import Professor

    results = Rating.objects.select_related('professor').filter(
        professor__gender__in=['Male', 'Female'],
        professor__discipline__isnull=False
    ).values(
        'professor__gender', 'professor__discipline'
    ).annotate(
        avg_rating_annotated=Avg('avg_rating')
    ).order_by('professor__discipline', 'professor__gender')

    return [
        {
            'gender': item['professor__gender'],
            'discipline': item['professor__discipline'],
            'avg_rating': float(item['avg_rating_annotated']) if item['avg_rating_annotated'] is not None else 0
        }
        for item in results
    ]