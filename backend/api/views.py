from django.utils import timezone
from datetime import timedelta
from rest_framework import viewsets, permissions, filters, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django_filters.rest_framework import DjangoFilterBackend
from django.db.models import Avg, Count, Case, When, Value, CharField, F, Q
from django.db.models.functions import Cast, ExtractYear, ExtractMonth, Concat
from django.contrib.auth import login, logout, get_user_model, authenticate
from .models import UserRole, Department, Professor, Rating, Sentiment
from .serializers import (
    DepartmentSerializer, ProfessorSerializer, RatingSerializer,
    SentimentSerializer, UserSerializer, UserRoleSerializer
)
from .utils import calculate_professor_metrics, analyze_department_bias

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
            except UserRole.DoesNotExist:
                role_data = None
                
            user_data = {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "role": role_data
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
        return Response({'message': 'Logged out successfully'})

    @action(detail=False, methods=['get'])
    def me(self, request):
        if request.user.is_authenticated:
            try:
                user_role = UserRole.objects.get(user=request.user)
                role_data = {"role": user_role.role, "discipline": user_role.discipline}
            except UserRole.DoesNotExist:
                role_data = None
                
            user_data = {
                "id": request.user.id,
                "username": request.user.username,
                "email": request.user.email,
                "first_name": request.user.first_name,
                "last_name": request.user.last_name,
                "role": role_data
            }
            return Response(user_data)
        return Response(status=status.HTTP_401_UNAUTHORIZED)

class IsAdminUser(permissions.BasePermission):
    def has_permission(self, request, view):
        return bool(request.user and request.user.is_authenticated and 
                   hasattr(request.user, 'role') and request.user.role.role == 'ADMIN')

# Base viewsets without role checks for now
class DepartmentViewSet(viewsets.ModelViewSet):
    queryset = Department.objects.all().order_by('name')
    serializer_class = DepartmentSerializer
    permission_classes = [permissions.IsAuthenticated]
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ['name', 'discipline', 'sub_discipline']
    ordering_fields = ['name', 'discipline']

    @action(detail=False, methods=['get'])
    def stats(self, request):
        """Get statistics for all departments"""
        departments = Department.objects.all()
        stats = {}
        
        for dept in departments:
            dept_stats = {
                'professorCount': Professor.objects.filter(department=dept).count(),
                'avgRating': Rating.objects.filter(
                    professor__department=dept
                ).aggregate(avg=Avg('avg_rating'))['avg'] or 0
            }
            stats[dept.id] = dept_stats
            
        return Response(stats)

    @action(detail=False, methods=['get'], url_path='bias/gender')
    def gender_bias(self, request):
        departments = Department.objects.all()
        results = []
        
        for dept in departments:
            bias_data = analyze_department_bias(dept.id)
            if bias_data['male_stats']['total_ratings'] > 0 and bias_data['female_stats']['total_ratings'] > 0:
                results.append({
                    'department_name': dept.name,
                    'male_avg_rating': bias_data['male_stats']['avg_rating'],
                    'female_avg_rating': bias_data['female_stats']['avg_rating'],
                    'rating_difference': bias_data['male_stats']['avg_rating'] - bias_data['female_stats']['avg_rating'],
                    'sample_size_male': bias_data['male_stats']['total_ratings'],
                    'sample_size_female': bias_data['female_stats']['total_ratings']
                })
        
        return Response(results)

class ProfessorViewSet(viewsets.ModelViewSet):
    queryset = Professor.objects.all()
    serializer_class = ProfessorSerializer
    permission_classes = [permissions.IsAuthenticated]
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ['department', 'gender']
    search_fields = ['first_name', 'last_name']
    ordering_fields = ['last_name', 'would_take_again_percent']

    def get_queryset(self):
        queryset = Professor.objects.all()
        return queryset.annotate(
            avg_rating=Avg('ratings__avg_rating'),
            avg_helpful=Avg('ratings__helpful_rating'),
            avg_clarity=Avg('ratings__clarity_rating'),
            avg_difficulty=Avg('ratings__difficulty_rating')
        )

    @action(detail=True, methods=['get'])
    def metrics(self, request, pk=None):
        """Get detailed metrics for a specific professor"""
        metrics = calculate_professor_metrics(pk)
        return Response(metrics)

class RatingViewSet(viewsets.ModelViewSet):
    queryset = Rating.objects.all()
    serializer_class = RatingSerializer
    permission_classes = [permissions.IsAuthenticated]
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_fields = ['professor', 'is_online', 'is_for_credit']
    ordering_fields = ['created_at', 'avg_rating', 'helpful_rating', 'clarity_rating', 'difficulty_rating']

    @action(detail=False, methods=['get'])
    def stats(self, request):
        """Get institution-wide rating statistics"""
        from django.db.models.functions import ExtractYear, ExtractMonth, Concat
        from django.db.models import Value, CharField
        
        # Get total evaluation count using count() instead of loading all records
        evaluation_count = Rating.objects.count()
        
        # Calculate semester stats using database aggregation
        semester_stats = Rating.objects.annotate(
            year=ExtractYear('created_at'),
            month=ExtractMonth('created_at'),
            semester=Case(
                When(month__lt=6, then=Concat(
                    Value('Spring '), Cast('year', CharField())
                )),
                default=Concat(
                    Value('Fall '), Cast('year', CharField())
                ),
                output_field=CharField(),
            )
        ).values('semester').annotate(
            score=Avg('avg_rating'),
            total_evaluations=Count('id')
        ).order_by('-semester')[:10]  # Limit to last 10 semesters for performance
        
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
            'averageScores': list(semester_stats),
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
        topics = Rating.objects.values('comment_topic').distinct()
        return Response({'topics': [t['comment_topic'] for t in topics if t['comment_topic']]})