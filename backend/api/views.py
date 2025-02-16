from django.utils import timezone
from datetime import timedelta
from rest_framework import viewsets, permissions, filters, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django_filters.rest_framework import DjangoFilterBackend
from django.db.models import Avg, Count, Case, When
from django.db.models.functions import Cast
from django.contrib.auth import login, logout, get_user_model, authenticate
from django.shortcuts import get_object_or_404
from django.http import Http404
from .models import Professor, Rating, Sentiment, UserRole
from .serializers import (
    ProfessorSerializer, RatingSerializer,
    SentimentSerializer, UserSerializer, UserRoleSerializer
)
from .filters import RatingFilter
from .utils import (
    calculate_professor_metrics, analyze_discipline_ratings,
    analyze_discipline_gender_distribution, perform_discipline_tukey_hsd,
    calculate_gender_distribution, get_sentiment_summary
)

User = get_user_model()

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
        discipline_stats = analyze_discipline_ratings()
        gender_stats = analyze_discipline_gender_distribution()
        tukey_results = perform_discipline_tukey_hsd()
        
        return Response({
            'discipline_ratings': discipline_stats,
            'gender_distribution': gender_stats,
            'statistical_tests': tukey_results
        })

    @action(detail=False, methods=['get'])
    def gender_distribution(self, request):
        """Get gender distribution analysis for top/bottom rated disciplines"""
        distribution_stats = calculate_gender_distribution()
        return Response(distribution_stats)

    @action(detail=True, methods=['get'])
    def sentiment_analysis(self, request, professor_id=None):
        """Get detailed sentiment analysis for a specific professor's ratings"""
        try:
            professor = self.get_object()
            summary = get_sentiment_summary(professor.professor_id)
            if not summary['total_comments']:
                return Response({
                    "error": "No sentiment analysis available for this professor"
                }, status=status.HTTP_404_NOT_FOUND)
            
            # Count sentiments
            positive_count = sum(1 for s in summary.get('recent_sentiments', []) if s['sentiment'] == 1)
            negative_count = sum(1 for s in summary.get('recent_sentiments', []) if s['sentiment'] == 0)
            
            # Restructure the data to match frontend expectations
            response_data = {
                'sentiment_counts': {
                    'positive': positive_count,
                    'negative': negative_count
                },
                'vader_scores': summary.get('vader_scores', {
                    'compound': 0,
                    'pos': 0,
                    'neg': 0
                }),
                'top_words': summary.get('top_words', {
                    'positive': [],
                    'negative': []
                }),
                'gender_analysis': {
                    'positive_terms': summary.get('gender_analysis', {}).get('positive_terms', []),
                    'negative_terms': summary.get('gender_analysis', {}).get('negative_terms', [])
                },
                'recent_sentiments': summary.get('recent_sentiments', [])
            }
            
            return Response(response_data)
        except Http404:
            return Response({
                "error": f"Professor with ID {professor_id} not found"
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                "error": f"Error fetching sentiment analysis: {str(e)}"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=True, methods=['get'], url_path='sentiment-analysis')
    def sentiment_analysis_kebab(self, request, professor_id=None):
        """Alias for sentiment_analysis with kebab-case URL"""
        return self.sentiment_analysis(request, professor_id)

    @action(detail=True, methods=['get'])
    def sentiment_summary(self, request, professor_id=None):
        """Get summarized sentiment statistics for a professor"""
        try:
            professor = self.get_object()
            summary = get_sentiment_summary(professor.professor_id)
            if not summary['total_comments']:
                return Response({
                    "error": "No sentiment summary available for this professor"
                }, status=status.HTTP_404_NOT_FOUND)
            return Response({
                'total_comments': summary['total_comments'],
                'sentiment_breakdown': summary['sentiment_breakdown'],
                'recent_sentiments': summary['recent_sentiments']
            })
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

class RatingViewSet(viewsets.ModelViewSet):
    queryset = Rating.objects.all()
    serializer_class = RatingSerializer
    permission_classes = [permissions.IsAuthenticated]
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_class = RatingFilter
    ordering_fields = ['created_at', 'avg_rating', 'helpful_rating', 'clarity_rating', 'difficulty_rating']

    @action(detail=False, methods=['get'])
    def stats(self, request):
        """Get institution-wide rating statistics"""
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
        
        return Response({
            'evaluationCount': evaluation_count,
            'metrics': metrics
        })

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