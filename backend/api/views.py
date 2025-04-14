# Python standard library imports
import os
import json
from datetime import timedelta

# Django imports
from django.utils import timezone
from django.db.models import Avg, Count
from django.contrib.auth import login, logout, get_user_model, authenticate
from django.shortcuts import get_object_or_404
from django.http import Http404
from django.core.cache import cache
from django.db import connection
from django.conf import settings

# Django REST framework imports
from rest_framework import viewsets, permissions, filters, status, pagination
from rest_framework.decorators import action
from rest_framework.response import Response
from django_filters.rest_framework import DjangoFilterBackend

# Local app imports
from .models import Professor, Rating, Sentiment, UserRole
from .serializers import (
    ProfessorSerializer, RatingSerializer,
    SentimentSerializer, UserSerializer, UserRoleSerializer
)
from .filters import RatingFilter
from .utils import (
    calculate_professor_metrics,
    analyze_discipline_ratings,
    analyze_discipline_gender_distribution,
    calculate_tukey_hsd,
    calculate_gender_discipline_heatmap,
    calculate_gender_distribution,
    get_sentiment_summary
)

# Machine learning imports
import torch
from machine_learning.gender_bias_explainer import GenderBiasExplainer
from machine_learning.ml_model_dev.lstm import CustomSentimentLSTM

User = get_user_model()

# Cache timeouts (in seconds)
CACHE_TIMEOUT_SHORT = 60 * 5  # 5 minutes
CACHE_TIMEOUT_MEDIUM = 60 * 30  # 30 minutes
CACHE_TIMEOUT_LONG = 60 * 60 * 12  # 12 hours

class AuthViewSet(viewsets.GenericViewSet):  # Changed from ViewSet to GenericViewSet
    permission_classes = [permissions.AllowAny]
    
    @action(detail=False, methods=['post'])
    def login(self, request):
        username = request.data.get('username')
        password = request.data.get('password')
        
        if not username or not password:
            return Response(
                {'error': 'Please provide both username and password'},
                status=status.HTTP_400_BAD_REQUEST
            )
            
        user = authenticate(username=username, password=password)
        
        if user is not None:
            login(request, user)
            try:
                user_role = UserRole.objects.get(user=user)
                role_data = {"role": user_role.role, "discipline": user_role.discipline}
                
                # For ACADEMIC users, get their professor ID
                professor_id = None
                if user_role.role == 'ACADEMIC':
                    try:
                        professor = Professor.objects.get(professor_id=user.username)
                        professor_id = professor.professor_id  # Return professor_id instead of database ID
                    except Professor.DoesNotExist:
                        pass
                
            except UserRole.DoesNotExist:
                role_data = None
                professor_id = None
                
            user_data = {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "role": role_data,
                "professor_id": professor_id
            }
            return Response({
                'user': user_data,
                'message': 'Logged in successfully'
            })
        else:
            return Response(
                {'error': 'Invalid credentials'},
                status=status.HTTP_401_UNAUTHORIZED
            )

    @action(detail=False, methods=['post'])
    def logout(self, request):
        logout(request)
        response = Response({'message': 'Logged out successfully'})
        response.delete_cookie('sessionid')
        response.delete_cookie('csrftoken')
        return response

    @action(detail=False, methods=['get'])
    def me(self, request):
        if not request.user.is_authenticated:
            return Response({'authenticated': False}, status=status.HTTP_401_UNAUTHORIZED)
            
        try:
            user_role = UserRole.objects.get(user=request.user)
            role_data = {"role": user_role.role, "discipline": user_role.discipline}
            
            # For ACADEMIC users, get their professor ID
            professor_id = None
            if user_role.role == 'ACADEMIC':
                try:
                    professor = Professor.objects.get(professor_id=request.user.username)
                    professor_id = professor.professor_id  # Return professor_id instead of database ID
                except Professor.DoesNotExist:
                    pass
                    
        except UserRole.DoesNotExist:
            role_data = None
            professor_id = None
            
        user_data = {
            "id": request.user.id,
            "username": request.user.username,
            "email": request.user.email,
            "first_name": request.user.first_name,
            "last_name": request.user.last_name,
            "role": role_data,
            "professor_id": professor_id,
            "authenticated": True
        }
        return Response(user_data)

class IsAdminUser(permissions.BasePermission):
    def has_permission(self, request, view):
        return bool(request.user and request.user.is_authenticated and 
                   hasattr(request.user, 'role') and request.user.role.role == 'ADMIN')

# Base viewsets without role checks for now
class ProfessorViewSet(viewsets.ModelViewSet):
    queryset = Professor.objects.all()
    serializer_class = ProfessorSerializer
    permission_classes = [permissions.IsAuthenticated]
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ['discipline', 'sub_discipline', 'gender']
    search_fields = ['first_name', 'last_name', 'discipline', 'sub_discipline']
    ordering_fields = ['last_name', 'discipline']
    lookup_field = 'professor_id'  # Use professor_id for lookups instead of pk

    # Define all available detail routes
    detail_actions = {
        'get': ['metrics', 'sentiment-analysis', 'sentiment-summary', 'topics', 'recommendations'],
    }

    def get_object(self):
        """Override to use professor_id lookup"""
        queryset = self.filter_queryset(self.get_queryset())
        lookup_url_kwarg = self.lookup_url_kwarg or self.lookup_field
        filter_kwargs = {self.lookup_field: self.kwargs[lookup_url_kwarg]}
        obj = get_object_or_404(queryset, **filter_kwargs)
        self.check_object_permissions(self.request, obj)
        return obj

    def get_queryset(self):
        queryset = Professor.objects.all()
        return queryset.annotate(
            avg_rating=Avg('ratings__avg_rating'),
            avg_helpful=Avg('ratings__helpful_rating'),
            avg_clarity=Avg('ratings__clarity_rating'),
            avg_difficulty=Avg('ratings__difficulty_rating')
        )

    @action(detail=True, methods=['get'])
    def metrics(self, request, professor_id=None):
        """Get detailed metrics for a specific professor"""
        professor = self.get_object()
        metrics = calculate_professor_metrics(professor.professor_id)
        return Response(metrics)

    @action(detail=False, methods=['get'])
    def discipline_stats(self, request):
        """Get statistical analysis of ratings by discipline"""
        # Check cache first
        cache_key = 'discipline_stats'
        cached_data = cache.get(cache_key)
        
        if cached_data:
            return Response(cached_data)
            
        discipline_stats = analyze_discipline_ratings()
        gender_stats = analyze_discipline_gender_distribution()
        
        data = {
            'discipline_ratings': discipline_stats,
            'gender_distribution': gender_stats,
        }
        
        # Cache the results
        cache.set(cache_key, data, CACHE_TIMEOUT_MEDIUM)
        
        return Response(data)

    @action(detail=False, methods=['get'])
    def gender_distribution(self, request):
        """Get gender distribution analysis for top/bottom rated disciplines"""
        # Check cache first
        cache_key = 'gender_distribution'
        cached_data = cache.get(cache_key)
        
        if cached_data:
            return Response(cached_data)
            
        distribution_stats = calculate_gender_distribution()
        
        # Cache the results
        cache.set(cache_key, distribution_stats, CACHE_TIMEOUT_MEDIUM)
        
        return Response(distribution_stats)

    @action(detail=True, methods=['get'])
    def sentiment_analysis(self, request, professor_id=None):
        """Get detailed sentiment analysis for a specific professor's ratings"""
        # Check if user is admin for institution-wide data
        if not request.user.role.role == 'ADMIN':
            return Response({
                "error": "Only administrators can view institution-wide sentiment analysis"
            }, status=status.HTTP_403_FORBIDDEN)

        try:
            if request.user.role.role == 'ADMIN':
                # Get institution-wide data
                summary = {
                    'gender_analysis': {
                        'vader': {
                            'positive_terms': self._get_institution_gender_terms(positive=True, sentiment_type='vader'),
                            'negative_terms': self._get_institution_gender_terms(positive=False, sentiment_type='vader')
                        },
                        'lexicon': {
                            'positive_terms': self._get_institution_gender_terms(positive=True, sentiment_type='lexicon'),
                            'negative_terms': self._get_institution_gender_terms(positive=False, sentiment_type='lexicon')
                        }
                    }
                }
            else:
                professor = self.get_object()
                summary = get_sentiment_summary(professor.professor_id)
            
            if not summary:
                return Response({
                    "error": "No sentiment analysis available"
                }, status=status.HTTP_404_NOT_FOUND)
            
            return Response(summary)
            
        except Http404:
            return Response({
                "error": f"Professor with ID {professor_id} not found"
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            import traceback
            print(f"Error in sentiment analysis endpoint: {str(e)}")
            print(traceback.format_exc())
            return Response({
                "error": f"Error fetching sentiment analysis: {str(e)}",
                "traceback": traceback.format_exc()
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def _get_institution_gender_terms(self, positive=True, sentiment_type='vader'):
        """Helper method to get institution-wide gender-based term analysis"""
        # Create a cache key based on the parameters
        cache_key = f'gender_terms_{sentiment_type}_{"positive" if positive else "negative"}'
        cached_data = cache.get(cache_key)
        
        if cached_data:
            return cached_data
            
        # Determine which field to analyze
        if sentiment_type == 'vader':
            field = 'positive_terms_vader' if positive else 'negative_terms_vader'
        else:
            field = 'positive_terms_lexicon' if positive else 'negative_terms_lexicon'
        
        # Use standard PostgreSQL query with jsonb_array_elements_text
        with connection.cursor() as cursor:
            # Query for male professors
            cursor.execute(f"""
                SELECT t.term, COUNT(*) as count
                FROM api_sentiment s
                JOIN api_professor p ON CAST(s.professor_id AS VARCHAR) = p.professor_id
                CROSS JOIN LATERAL jsonb_array_elements_text(s.{field}) AS t(term)
                WHERE t.term IS NOT NULL
                AND p.gender IN ('Male', 'Female')
                GROUP BY t.term
                ORDER BY count DESC
            """)
            male_terms = cursor.fetchall()
            
            # Query for female professors
            cursor.execute(f"""
                SELECT t.term, COUNT(*) as count
                FROM api_sentiment s
                JOIN api_professor p ON CAST(s.professor_id AS VARCHAR) = p.professor_id
                CROSS JOIN LATERAL jsonb_array_elements_text(s.{field}) AS t(term)
                WHERE t.term IS NOT NULL
                AND p.gender = 'Female'
                GROUP BY t.term
                ORDER BY count DESC
            """)
            female_terms = cursor.fetchall()
        
        # Convert results to dictionaries
        male_counter = {term: count for term, count in male_terms}
        female_counter = {term: count for term, count in female_terms}
        
        # Calculate totals
        male_total = sum(male_counter.values()) or 1
        female_total = sum(female_counter.values()) or 1
        
        # Get all unique terms
        all_terms = set(male_counter.keys()) | set(female_counter.keys())
        
        # Calculate relative frequencies and bias
        terms_data = []
        bias_threshold = 1.1
        
        for term in all_terms:
            male_freq = male_counter.get(term, 0)
            female_freq = female_counter.get(term, 0)
            
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
        
        # Sort by total frequency and get top 10 for each bias
        male_biased = sorted(
            [term for term in terms_data if term['bias'] == 'Male'],
            key=lambda x: x['male_freq'],
            reverse=True
        )[:10]
        
        female_biased = sorted(
            [term for term in terms_data if term['bias'] == 'Female'],
            key=lambda x: x['female_freq'],
            reverse=True
        )[:10]
        
        # Combine results
        results = male_biased + female_biased
        
        # Cache results
        cache.set(cache_key, results, CACHE_TIMEOUT_LONG)
        
        return results

    @action(detail=True, methods=['get'], url_path='sentiment-analysis')
    def sentiment_analysis_kebab(self, request, professor_id=None):
        """Alias for sentiment_analysis with kebab-case URL"""
        return self.sentiment_analysis(request, professor_id)

    @action(detail=True, methods=['get'])
    def sentiment_summary(self, request, professor_id=None):
        """Get summarized sentiment statistics for a professor with paginated comments"""
        try:
            page = request.query_params.get('page', 1)
            # For institution-wide data, use caching
            if request.user.role.role == 'ADMIN':   
                # Check cache for institution data
                cache_key = f'institution_sentiment_summary_page_{page}'
                cached_data = cache.get(cache_key)
                
                if cached_data:
                    return Response(cached_data)
                
                # Process institution-wide data
                sentiment_data = get_sentiment_summary(institution=True, page=int(page))
                
                # Cache the results
                cache.set(cache_key, sentiment_data, CACHE_TIMEOUT_SHORT)
                
                return Response(sentiment_data)
            else:
                professor = self.get_object()
                sentiment_data = get_sentiment_summary(professor_id=professor.professor_id, page=int(page))
                return Response(sentiment_data)
                
        except Http404:
            return Response({
                "error": f"Professor with ID {professor_id} not found"
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                "error": f"Error fetching sentiment summary: {str(e)}"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=True, methods=['get'], url_path='sentiment-summary')
    def sentiment_summary_kebab(self, request, professor_id=None):
        """Alias for sentiment_summary with kebab-case URL"""
        return self.sentiment_summary(request, professor_id)

    @action(detail=True, methods=['get'])
    def topics(self, request, professor_id=None):
        """Get common topics from professor's ratings"""
        professor = self.get_object()
        topics = Sentiment.objects.filter(
            professor=professor,
            comment_topic__isnull=False
        ).values('comment_topic').annotate(
            count=Count('comment_topic')
        ).order_by('-count')
        return Response({'topics': list(topics)})

    @action(detail=True, methods=['get'])
    def recommendations(self, request, professor_id=None):
        """Get personalized recommendations based on professor's performance"""
        professor = self.get_object()
        metrics = calculate_professor_metrics(professor.professor_id)
        
        if not metrics:
            return Response({
                "error": "No metrics available for this professor"
            }, status=404)

        categories = {
            "teaching_effectiveness": {
                "score": metrics.get('avg_rating', 0) or 0,
                "recommendations": []
            },
            "clarity": {
                "score": metrics.get('avg_clarity', 0) or 0,
                "recommendations": []
            },
            "workload": {
                "score": metrics.get('avg_difficulty', 0) or 0,
                "recommendations": []
            }
        }

        # Generate categorized recommendations
        if (metrics.get('avg_rating') or 0) < 3.0:
            categories["teaching_effectiveness"]["recommendations"].append({
                "id": 1,
                "text": "Consider reviewing teaching methods for improvement",
                "priority": "high",
                "impact_score": 8.5,
                "supporting_ratings": metrics.get('total_ratings', 0),
                "category": "teaching_effectiveness"
            })

        if metrics.get('avg_clarity', 0) < 3.0:
            categories["clarity"]["recommendations"].append({
                "id": 2,
                "text": "Focus on improving clarity in lectures",
                "priority": "high",
                "impact_score": 8.0,
                "supporting_ratings": metrics.get('total_ratings', 0),
                "category": "clarity"
            })

        if metrics.get('avg_helpful', 0) < 3.0:
            categories["teaching_effectiveness"]["recommendations"].append({
                "id": 3,
                "text": "Consider increasing office hours availability",
                "priority": "medium",
                "impact_score": 6.5,
                "supporting_ratings": metrics.get('total_ratings', 0),
                "category": "teaching_effectiveness"
            })

        if metrics.get('avg_difficulty', 0) > 4.0:
            categories["workload"]["recommendations"].append({
                "id": 4,
                "text": "Review course material complexity",
                "priority": "medium",
                "impact_score": 7.0,
                "supporting_ratings": metrics.get('total_ratings', 0),
                "category": "workload"
            })
            
        return Response({
            "categories": categories,
            "overall_metrics": {
                "avg_rating": metrics.get('avg_rating', 0),
                "helpful_rating": metrics.get('avg_helpful', 0),
                "clarity_rating": metrics.get('avg_clarity', 0),
                "difficulty_rating": metrics.get('avg_difficulty', 0)
            },
            "total_ratings": metrics.get('total_ratings', 0),
            "last_updated": timezone.now().isoformat()
        })

    @action(detail=True, methods=['get'])
    def word_clouds(self, request, professor_id=None):
        """Get word cloud data for both VADER and Lexicon analyses"""
        # Check if user is admin for institution-wide data
        if not request.user.role.role == 'ADMIN':
            return Response({
                "error": "Only administrators can view institution-wide sentiment analysis"
            }, status=status.HTTP_403_FORBIDDEN)

        try:
            # Use raw SQL to handle JSONB arrays properly
            with connection.cursor() as cursor:
                # Query for VADER positive terms
                cursor.execute("""
                    SELECT t.term, COUNT(*) as count
                    FROM api_sentiment s
                    CROSS JOIN LATERAL jsonb_array_elements_text(s.positive_terms_vader) AS t(term)
                    WHERE t.term IS NOT NULL
                    GROUP BY t.term
                    ORDER BY count DESC
                    LIMIT 50
                """)
                vader_pos_terms = cursor.fetchall()
                
                # Query for VADER negative terms
                cursor.execute("""
                    SELECT t.term, COUNT(*) as count
                    FROM api_sentiment s
                    CROSS JOIN LATERAL jsonb_array_elements_text(s.negative_terms_vader) AS t(term)
                    WHERE t.term IS NOT NULL
                    GROUP BY t.term
                    ORDER BY count DESC
                    LIMIT 50
                """)
                vader_neg_terms = cursor.fetchall()
                
                # Query for Lexicon positive terms
                cursor.execute("""
                    SELECT t.term, COUNT(*) as count
                    FROM api_sentiment s
                    CROSS JOIN LATERAL jsonb_array_elements_text(s.positive_terms_lexicon) AS t(term)
                    WHERE t.term IS NOT NULL
                    GROUP BY t.term
                    ORDER BY count DESC
                    LIMIT 50
                """)
                lexicon_pos_terms = cursor.fetchall()
                
                # Query for Lexicon negative terms
                cursor.execute("""
                    SELECT t.term, COUNT(*) as count
                    FROM api_sentiment s
                    CROSS JOIN LATERAL jsonb_array_elements_text(s.negative_terms_lexicon) AS t(term)
                    WHERE t.term IS NOT NULL
                    GROUP BY t.term
                    ORDER BY count DESC
                    LIMIT 50
                """)
                lexicon_neg_terms = cursor.fetchall()
                
            # Create response with filtered terms
            response_data = {
                'vader': {
                    'positive': [{'word': word, 'count': count} 
                               for word, count in vader_pos_terms 
                               if word and word.strip()],
                    'negative': [{'word': word, 'count': count} 
                               for word, count in vader_neg_terms 
                               if word and word.strip()]
                },
                'lexicon': {
                    'positive': [{'word': word, 'count': count} 
                               for word, count in lexicon_pos_terms 
                               if word and word.strip()],
                    'negative': [{'word': word, 'count': count} 
                               for word, count in lexicon_neg_terms 
                               if word and word.strip()]
                }
            }
            
            return Response(response_data)
            
        except Exception as e:
            import traceback
            print(f"Error in word_clouds endpoint: {str(e)}")
            print(traceback.format_exc())
            return Response({
                "error": f"Error fetching word cloud data: {str(e)}",
                "traceback": traceback.format_exc()
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=False, methods=['get'])
    def tukey_analysis(self, request):
        """Get Tukey's HSD test results for gender comparisons across disciplines"""
        try:
            # Check cache first
            cache_key = 'tukey_analysis'
            cached_data = cache.get(cache_key)
            
            if cached_data:
                return Response(cached_data)
                
            tukey_results = calculate_tukey_hsd()
            
            # Cache the results
            cache.set(cache_key, tukey_results, CACHE_TIMEOUT_LONG)
            
            return Response(tukey_results)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=False, methods=['get'])
    def gender_discipline_heatmap(self, request):
        """Get average ratings by gender and discipline for heatmap visualization"""
        # Check cache first
        cache_key = 'gender_discipline_heatmap'
        cached_data = cache.get(cache_key)
        
        if cached_data:
            return Response(cached_data)
            
        heatmap_data = calculate_gender_discipline_heatmap()
        
        # Cache the results
        cache.set(cache_key, heatmap_data, CACHE_TIMEOUT_MEDIUM)
        
        return Response(heatmap_data)

    @action(detail=False, methods=['get'])
    def gender_term_analysis(self, request):
        """Get gender-based term analysis at institutional level"""
        sentiment_type = request.query_params.get('sentiment_type', 'vader') 
        term_type = request.query_params.get('term_type', 'positive')
        
        # Create a cache key
        cache_key = f'gender_term_{sentiment_type}_{term_type}'
        cached_data = cache.get(cache_key)
        
        if cached_data:
            return Response(cached_data)
        
        try:
            # Determine which field to analyze
            if sentiment_type == 'vader':
                field = 'positive_terms_vader' if term_type == 'positive' else 'negative_terms_vader'
            else:
                field = 'positive_terms_lexicon' if term_type == 'positive' else 'negative_terms_lexicon'
            
            # Use raw SQL for proper JSONB array handling
            with connection.cursor() as cursor:
                # Get male terms
                cursor.execute(f"""
                    SELECT t.term, COUNT(*) as count
                    FROM api_sentiment s
                    JOIN api_professor p ON CAST(s.professor_id AS VARCHAR) = p.professor_id
                    CROSS JOIN LATERAL jsonb_array_elements_text(s.{field}) AS t(term)
                    WHERE t.term IS NOT NULL
                    AND p.gender = 'Male'
                    GROUP BY t.term
                    ORDER BY count DESC
                """)
                male_terms = cursor.fetchall()
                
                # Get female terms
                cursor.execute(f"""
                    SELECT t.term, COUNT(*) as count
                    FROM api_sentiment s
                    JOIN api_professor p ON CAST(s.professor_id AS VARCHAR) = p.professor_id
                    CROSS JOIN LATERAL jsonb_array_elements_text(s.{field}) AS t(term)
                    WHERE t.term IS NOT NULL
                    AND p.gender = 'Female'
                    GROUP BY t.term
                    ORDER BY count DESC
                """)
                female_terms = cursor.fetchall()
            
            # Convert to dictionaries for easier manipulation
            male_dict = {term: count for term, count in male_terms}
            female_dict = {term: count for term, count in female_terms}
            
            # Calculate totals
            male_total = sum(male_dict.values()) or 1
            female_total = sum(female_dict.values()) or 1
            
            # Calculate bias and relative frequencies
            all_terms = set(male_dict.keys()) | set(female_dict.keys())
            results = []
            
            for term in all_terms:
                male_freq = male_dict.get(term, 0)
                female_freq = female_dict.get(term, 0)
                
                # Skip rare terms
                if male_freq + female_freq < 3:
                    continue
                
                # Calculate relative frequencies
                male_rel_freq = male_freq / male_total
                female_rel_freq = female_freq / female_total
                
                # Determine bias
                bias = 'neutral'
                if male_rel_freq > 1.5 * female_rel_freq:
                    bias = 'male'
                elif female_rel_freq > 1.5 * male_rel_freq:
                    bias = 'female'
                
                results.append({
                    'term': term,
                    'male_count': male_freq,
                    'female_count': female_freq,
                    'male_relative': round(male_rel_freq, 4),
                    'female_relative': round(female_rel_freq, 4),
                    'total': male_freq + female_freq,
                    'bias': bias
                })
            
            # Sort by total frequency
            results.sort(key=lambda x: x['total'], reverse=True)
            
            response_data = {
                'results': results[:100],  # Limit to top 100 terms
                'totals': {
                    'male_terms': male_total,
                    'female_terms': female_total
                },
                'sentiment_type': sentiment_type,
                'term_type': term_type
            }
            
            # Cache the results
            cache.set(cache_key, response_data, CACHE_TIMEOUT_MEDIUM)
            
            return Response(response_data)
            
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=False, methods=['get'])
    def gender_ratings_comparison(self, request):
        """Compare ratings across genders at institutional level"""
        # Check cache first
        cache_key = 'gender_ratings_comparison'
        cached_data = cache.get(cache_key)
        
        if cached_data:
            return Response(cached_data)
        
        # Use database aggregation to get gender rating statistics
        gender_stats = Rating.objects.select_related('professor').values(
            'professor__gender'
        ).annotate(
            avg_rating=Avg('avg_rating'),
            avg_clarity=Avg('clarity_rating'),
            avg_helpfulness=Avg('helpful_rating'),
            avg_difficulty=Avg('difficulty_rating'),
            count=Count('id')
        ).filter(
            professor__gender__in=['Male', 'Female']
        )
        
        # Format response
        stats = {item['professor__gender']: {
            'avg_rating': round(item['avg_rating'], 2),
            'avg_clarity': round(item['avg_clarity'], 2),
            'avg_helpfulness': round(item['avg_helpfulness'], 2),
            'avg_difficulty': round(item['avg_difficulty'], 2),
            'count': item['count']
        } for item in gender_stats}
        
        # Get discipline-level breakdown
        discipline_gender_stats = Rating.objects.select_related('professor').values(
            'professor__gender', 'professor__discipline'
        ).annotate(
            avg_rating=Avg('avg_rating'),
            count=Count('id')
        ).filter(
            professor__gender__in=['Male', 'Female'],
            professor__discipline__isnull=False
        ).order_by('professor__discipline')
        
        disciplines = {}
        for item in discipline_gender_stats:
            discipline = item['professor__discipline']
            gender = item['professor__gender']
            
            if discipline not in disciplines:
                disciplines[discipline] = {'Male': {}, 'Female': {}}
                
            disciplines[discipline][gender] = {
                'avg_rating': round(item['avg_rating'], 2),
                'count': item['count']
            }
        
        results = {
            'overall': stats,
            'disciplines': disciplines
        }
        
        # Cache the results
        cache.set(cache_key, results, CACHE_TIMEOUT_LONG)
        
        return Response(results)

class RatingViewSet(viewsets.ModelViewSet):
    queryset = Rating.objects.all()
    serializer_class = RatingSerializer
    permission_classes = [permissions.AllowAny]
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_class = RatingFilter
    search_fields = ['professor__first_name', 'professor__last_name', 'professor__discipline', 'professor__sub_discipline']
    ordering_fields = ['created_at', 'avg_rating', 'helpful_rating', 'clarity_rating', 'difficulty_rating']
    pagination_class = pagination.PageNumberPagination
    page_size = 50

    def get_queryset(self):
        queryset = Rating.objects.select_related('professor').all()
        return queryset
        
    def destroy(self, request, *args, **kwargs):
        """Override destroy method to delete course evaluation"""
        try:
            instance = self.get_object()
            self.perform_destroy(instance)
            return Response({"message": "Evaluation deleted successfully"}, status=status.HTTP_204_NO_CONTENT)
        except Exception as e:
            import traceback
            print(f"Error in destroy method: {str(e)}")
            print(traceback.format_exc())
            return Response({"error": f"Error deleting evaluation: {str(e)}"}, 
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    @action(detail=False, methods=['post', 'delete'])
    def bulk_delete(self, request):
        """Bulk delete multiple ratings"""
        try:
            # Handle both POST and DELETE requests
            if request.method == 'DELETE':
                # For DELETE requests, data is in request.data or request.query_params
                ids = request.data.get('ids', []) if request.data else request.query_params.getlist('ids')
            else:  # POST
                ids = request.data.get('ids', [])
                
            if not ids:
                return Response({"error": "No rating IDs provided"}, 
                                status=status.HTTP_400_BAD_REQUEST)
                                
            # Get the ratings to delete
            ratings_to_delete = Rating.objects.filter(id__in=ids)
            if not ratings_to_delete.exists():
                return Response({"error": "No ratings found with the provided IDs"}, 
                                status=status.HTTP_404_NOT_FOUND)
                                
            # Count how many were found
            found_count = ratings_to_delete.count()
            
            # Perform the delete
            ratings_to_delete.delete()
            
            return Response({
                "message": f"Successfully deleted {found_count} ratings",
                "deleted_count": found_count
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            import traceback
            print(f"Error in bulk_delete method: {str(e)}")
            print(traceback.format_exc())
            return Response({"error": f"Error performing bulk delete: {str(e)}"}, 
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=False, methods=['get'])
    def stats(self, request):
        """Get institution-wide rating statistics"""
        # Check cache first
        cache_key = 'rating_stats'
        cached_data = cache.get(cache_key)
        
        if cached_data:
            return Response(cached_data)
        
        # Get total evaluation count using count() instead of loading all records
        evaluation_count = Rating.objects.count()
        
        # Calculate overall metrics
        metrics = Rating.objects.aggregate(
            avg_rating=Avg('avg_rating'),
            avg_helpful=Avg('helpful_rating'),
            avg_clarity=Avg('clarity_rating'),
            avg_difficulty=Avg('difficulty_rating')
        )
        
        # Calculate trend using a single query
        thirty_days_ago = timezone.now() - timedelta(days=30)
        recent_avg = Rating.objects.filter(
            created_at__gte=thirty_days_ago
        ).aggregate(
            recent_avg=Avg('avg_rating')
        )['recent_avg'] or 0
        
        metrics['trend'] = recent_avg - (metrics['avg_rating'] or 0)
        
        data = {
            'evaluationCount': evaluation_count,
            'metrics': metrics
        }
        
        # Cache the results
        cache.set(cache_key, data, CACHE_TIMEOUT_SHORT)
        
        return Response(data)
        
    @action(detail=False, methods=['get', 'post'], permission_classes=[permissions.AllowAny])
    def upload(self, request):
        """
        Upload and process evaluation data CSV file.
        GET: Returns a status or form
        POST: Processes the uploaded file
        """
        
        # Handle GET request - Just return a simple response for now
        if request.method == 'GET':
            return Response({
                "message": "Upload evaluation data (CSV) using a POST request to this endpoint", 
                "allowed_methods": ["POST"],
                "required_fields": ["file (CSV file)"]
            })
            
        # Handle POST request for file upload
        if 'file' not in request.FILES:
            return Response({
                "error": "No file was uploaded",
                "help": "Please attach a CSV file with the 'file' key"
            }, status=status.HTTP_400_BAD_REQUEST)
            
        file = request.FILES['file']
        
        # Validate file extension
        if not file.name.endswith('.csv'):
            return Response({
                "error": "Uploaded file must be a CSV file"
            }, status=status.HTTP_400_BAD_REQUEST)
            
        # Save the uploaded file temporarily
        import tempfile
        import os
        from django.conf import settings
        from django.core.files.storage import FileSystemStorage
        
        # Create temp directory if it doesn't exist
        temp_dir = os.path.join(settings.BASE_DIR, 'tmp')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save the file
        fs = FileSystemStorage(location=temp_dir)
        filename = fs.save(file.name, file)
        file_path = os.path.join(temp_dir, filename)
        
        try:
            # Configure database connection
            db_config = {
                'dbname': settings.DATABASES['default']['NAME'],
                'user': settings.DATABASES['default']['USER'],
                'password': settings.DATABASES['default']['PASSWORD'],
                'host': settings.DATABASES['default']['HOST'],
                'port': settings.DATABASES['default']['PORT']
            }
            
            from data_processing.pipeline import run_full_pipeline
            from api.tasks import process_evaluation_data_task
            
            # Run the data processing task asynchronously
            task = process_evaluation_data_task.delay(file_path, db_config)
            
            return Response({
                "message": "File uploaded successfully. Processing has begun.",
                "task_id": task.id
            }, status=status.HTTP_202_ACCEPTED)
            
        except Exception as e:
            # Clean up the file
            if os.path.exists(file_path):
                os.remove(file_path)
                
            import traceback
            return Response({
                "error": f"Error processing file: {str(e)}",
                "details": traceback.format_exc()
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
    @action(detail=False, methods=['get'])
    def upload_status(self, request):
        """Check the status of an evaluation data upload task"""
        task_id = request.query_params.get('task_id')
        
        if not task_id:
            return Response({
                "error": "No task_id provided"
            }, status=status.HTTP_400_BAD_REQUEST)
            
        from api.tasks import process_evaluation_data_task
        from celery.result import AsyncResult
        
        task_result = AsyncResult(task_id)
        
        result = {
            "task_id": task_id,
            "status": task_result.status,
        }
        
        if task_result.successful():
            result["result"] = task_result.result
        elif task_result.failed():
            result["error"] = str(task_result.result)
            
        return Response(result)

class SentimentViewSet(viewsets.ModelViewSet):
    queryset = Sentiment.objects.all()
    serializer_class = SentimentSerializer
    permission_classes = [permissions.IsAuthenticated]
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_fields = ['professor', 'sentiment']
    ordering_fields = ['created_at', 'sentiment']

class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAdminUser]

class UserRoleViewSet(viewsets.ModelViewSet):
    queryset = UserRole.objects.all()
    serializer_class = UserRoleSerializer 
    permission_classes = [permissions.IsAdminUser]

class TopicViewSet(viewsets.ViewSet):
    permission_classes = [permissions.IsAuthenticated]
    
    def list(self, request):
        # Get unique topics from ratings and sentiments
        topics = Sentiment.objects.values('comment_topic').distinct()
        return Response({'topics': [t['comment_topic'] for t in topics if t['comment_topic']]})

class SentimentExplainabilityViewSet(viewsets.ViewSet):
    """
    API endpoints for LSTM model explainability and gender bias analysis
    """
    lookup_field = 'professor_id'
    permission_classes = [permissions.IsAuthenticated] # Keep authentication

    @action(detail=False, methods=['post'])
    def explain_comment(self, request):
        """Analyze a comment with attention-based gender bias explanation"""
        comment = request.data.get('comment', '')
        discipline = request.data.get('discipline', None)
        # --- GET SELECTED GENDER ---
        selected_gender = request.data.get('gender', None)

        if not comment:
            return Response({"error": "Comment text is required"}, status=status.HTTP_400_BAD_REQUEST)
        # --- VALIDATE GENDER ---
        if not selected_gender or selected_gender not in ['Male', 'Female']:
            return Response({"error": "Valid gender ('Male' or 'Female') selection is required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # --- LOAD MODEL AND VOCAB (same as before) ---
            model_path = os.path.join(settings.BASE_DIR, 'machine_learning/ml_models_trained/lstm_sentiment.pt')
            vocab_path = os.path.join(settings.BASE_DIR, 'machine_learning/ml_models_trained/vocab.json')
            if not os.path.exists(model_path) or not os.path.exists(vocab_path):
                return Response({"error": "Model files not found"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = CustomSentimentLSTM(
                vocab_size=len(vocab), embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.5
            ).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            explainer = GenderBiasExplainer(model, vocab)

            # --- PASS selected_gender TO EXPLAINER ---
            explanation = explainer.explain_prediction(comment, selected_gender, discipline)

            # --- The explanation structure is already updated by the explainer ---
            return Response(explanation)

        except Exception as e:
            import traceback
            return Response({
                "error": f"Error analyzing comment: {str(e)}",
                "traceback": traceback.format_exc()
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=True, methods=['get'])
    def professor_bias_analysis(self, request, professor_id=None):
        """Analyze gender bias patterns in a professor's comments"""
        try:
            # Permission check (no change)
            user_role = getattr(request.user, 'role', None)
            if not user_role or (user_role.role != 'ADMIN' and request.user.username != professor_id):
                 return Response({"error": "You don't have permission to access this data"}, status=status.HTTP_403_FORBIDDEN)

            from api.models import Sentiment, Professor
            try:
                professor = Professor.objects.get(professor_id=professor_id)
            except Professor.DoesNotExist:
                return Response({"error": f"Professor with ID {professor_id} not found"}, status=status.HTTP_404_NOT_FOUND)

            # --- GET PROFESSOR'S GENDER ---
            selected_gender = professor.gender
            if not selected_gender or selected_gender not in ['Male', 'Female']:
                 # Handle cases where professor gender might be missing or invalid
                 return Response({"error": f"Professor {professor_id} has an invalid or missing gender designation. Analysis cannot proceed."}, status=status.HTTP_400_BAD_REQUEST)

            comments = Sentiment.objects.filter(
                professor_id=professor_id, comment__isnull=False
            ).values_list('comment', flat=True)

            if not comments:
                return Response({"error": "No comments found for this professor"}, status=status.HTTP_404_NOT_FOUND)

            # Load model and vocab (same as before)
            model_path = os.path.join(settings.BASE_DIR, 'machine_learning/ml_models_trained/lstm_sentiment.pt')
            vocab_path = os.path.join(settings.BASE_DIR, 'machine_learning/ml_models_trained/vocab.json')
            if not os.path.exists(model_path) or not os.path.exists(vocab_path):
                return Response({"error": "Model files not found"}, status=500)
            with open(vocab_path, 'r', encoding='utf-8') as f: vocab = json.load(f)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = CustomSentimentLSTM(len(vocab), 128, 256, 2, 0.5).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            explainer = GenderBiasExplainer(model, vocab)

            discipline = professor.discipline
            comments_list = list(comments[:100]) # Limit batch size

            # --- PASS selected_gender TO BATCH ANALYZER ---
            batch_results = explainer.analyze_comments_batch(comments_list, selected_gender, [discipline] * len(comments_list))

            metrics = calculate_professor_metrics(professor_id)
            recommendations = self._generate_bias_recommendations(batch_results, metrics, selected_gender) # Pass gender

            # --- NO NEED TO EXTRACT top_male/female_terms anymore, structure changed ---

            response = {
                'professor_id': professor_id,
                'discipline': discipline,
                'professor_gender': selected_gender, # Add professor's gender for context
                'analysis_results': batch_results, # Structure already updated by explainer
                'recommendations': recommendations
            }
            return Response(response)

        except Exception as e:
            import traceback
            return Response({
                "error": f"Error analyzing professor bias: {str(e)}",
                "traceback": traceback.format_exc()
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # --- REVISED RECOMMENDATIONS BASED ON NEW STRUCTURE ---
    def _generate_bias_recommendations(self, analysis_results, metrics, professor_gender):
        """Generate teaching recommendations based on new bias analysis structure"""
        recommendations = []
        insights = analysis_results.get('insights', [])
        interpretations = analysis_results.get('interpretation_summary', {})
        descriptor_bias = analysis_results.get('descriptor_bias_score', 0)

        # Helper to check for specific insight keywords
        def has_insight(keywords):
            return any(keyword.lower() in insight.lower() for insight in insights for keyword in keywords)

        # 1. Recommendation based on predominant negative bias interpretation
        neg_bias_interp = None
        for interp, count in interpretations.items():
             if "negative bias" in interp.lower():
                 # Check if this is the most common or very frequent
                 if count / analysis_results['comment_count'] > 0.2: # If > 20% comments show negative bias
                     neg_bias_interp = interp
                     break

        if neg_bias_interp:
            recommendations.append({
                'text': f"A significant portion of feedback indicates potential negative gender bias ('{neg_bias_interp}'). Review feedback for patterns of gendered criticism and consider workshops on equitable evaluation.",
                'priority': 'high',
                'impact_score': 9.0,
                'supporting_evidence': ["See 'Predominant Finding' insight and negative term frequencies."]
            })

        # 2. Recommendation based on stereotypical positive praise
        pos_bias_interp = None
        for interp, count in interpretations.items():
            if "stereotypical praise" in interp.lower():
                 if count / analysis_results['comment_count'] > 0.25: # If > 25% comments show stereotypical praise
                     pos_bias_interp = interp
                     break
        if pos_bias_interp:
             recommendations.append({
                'text': f"Feedback often relies on stereotypical praise ('{pos_bias_interp}'). Aim to solicit and acknowledge feedback on a broader range of professional attributes.",
                'priority': 'medium',
                'impact_score': 7.0,
                'supporting_evidence': ["See interpretation summary and descriptor category stats."]
             })

        # 3. Recommendation based on strong descriptor focus skew
        if has_insight(["strong focus skew", "heavily emphasize"]):
            focus = "male-associated (intellect/entertainment)" if descriptor_bias > 0 else "female-associated (competence/warmth)"
            recommendations.append({
                'text': f"Evaluations show a strong skew towards focusing on {focus} descriptors. This might reflect student bias. Ensure course materials and assessments highlight diverse strengths.",
                'priority': 'medium',
                'impact_score': 6.5,
                'supporting_evidence': ["See 'Strong Focus Skew' insight."]
            })

        # 4. Recommendation if low ratings correlate with bias indicators
        avg_rating = metrics.get('avg_rating', 5.0) # Default high if no metrics
        if avg_rating < 3.5 and (neg_bias_interp or pos_bias_interp or abs(descriptor_bias) > 0.2):
             recommendations.append({
                'text': "Lower overall ratings coincide with indicators of potential gender bias in comments. Explore inclusive teaching strategies to address potential underlying issues.",
                'priority': 'high',
                'impact_score': 8.0,
                'supporting_evidence': [f"Avg Rating: {avg_rating:.2f}", "Review bias interpretations and insights."]
             })

        # 5. General recommendation if bias patterns are present but not extreme
        if not recommendations and (abs(descriptor_bias) > 0.15 or any("bias" in interp.lower() for interp in interpretations)):
             recommendations.append({
                 'text': f"Some language patterns suggest potential gender bias. Regularly reflect on feedback through a bias lens and seek diverse perspectives on teaching effectiveness.",
                 'priority': 'low',
                 'impact_score': 5.0,
                 'supporting_evidence': ["Review descriptor focus and interpretation summary."]
             })

        return recommendations

    @action(detail=False, methods=['post'], url_path='upload-evaluation-data', permission_classes=[permissions.AllowAny])
    def upload_evaluation_data(self, request):
        """
        Upload and process evaluation data CSV file.
        This endpoint is accessible to all users.
        """
        # Check if the file was uploaded
        if 'file' not in request.FILES:
            return Response({
                "error": "No file was uploaded"
            }, status=status.HTTP_400_BAD_REQUEST)
            
        # Validate file extension
        file = request.FILES['file']
        if not file.name.endswith('.csv'):
            return Response({
                "error": "Uploaded file must be a CSV file"
            }, status=status.HTTP_400_BAD_REQUEST)
            
        # Save the uploaded file temporarily
        import tempfile
        import os
        from django.conf import settings
        from django.core.files.storage import FileSystemStorage
        
        # Create temp directory if it doesn't exist
        temp_dir = os.path.join(settings.BASE_DIR, 'tmp')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save the file
        fs = FileSystemStorage(location=temp_dir)
        filename = fs.save(file.name, file)
        file_path = os.path.join(temp_dir, filename)
        
        try:
            # Configure database connection
            db_config = {
                'dbname': settings.DATABASES['default']['NAME'],
                'user': settings.DATABASES['default']['USER'],
                'password': settings.DATABASES['default']['PASSWORD'],
                'host': settings.DATABASES['default']['HOST'],
                'port': settings.DATABASES['default']['PORT']
            }
            
            from data_processing.pipeline import run_full_pipeline
            from api.tasks import process_evaluation_data_task
            
            # Run the data processing task asynchronously
            task = process_evaluation_data_task.delay(file_path, db_config)
            
            return Response({
                "message": "File uploaded successfully. Processing has begun.",
                "task_id": task.id
            }, status=status.HTTP_202_ACCEPTED)
            
        except Exception as e:
            # Clean up the file
            if os.path.exists(file_path):
                os.remove(file_path)
                
            return Response({
                "error": f"Error processing file: {str(e)}"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
    @action(detail=False, methods=['get'])
    def upload_status(self, request):
        """Check the status of an evaluation data upload task"""
        task_id = request.query_params.get('task_id')
        
        if not task_id:
            return Response({
                "error": "No task_id provided"
            }, status=status.HTTP_400_BAD_REQUEST)
            
        from api.tasks import process_evaluation_data_task
        from celery.result import AsyncResult
        
        task_result = AsyncResult(task_id)
        
        result = {
            "task_id": task_id,
            "status": task_result.status,
        }
        
        if task_result.successful():
            result["result"] = task_result.result
        elif task_result.failed():
            result["error"] = str(task_result.result)
            
        return Response(result)