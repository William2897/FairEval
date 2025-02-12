from rest_framework import viewsets, permissions, filters, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django_filters.rest_framework import DjangoFilterBackend
from django.db.models import Avg
from django.contrib.auth import login, logout, get_user_model, authenticate
from .models import UserRole, Department, Professor, Rating, Sentiment
from .serializers import (
    DepartmentSerializer, ProfessorSerializer, RatingSerializer,
    SentimentSerializer, UserSerializer, UserRoleSerializer
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
    queryset = Department.objects.all()
    serializer_class = DepartmentSerializer
    permission_classes = [permissions.IsAuthenticated]
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ['name', 'discipline', 'sub_discipline']
    ordering_fields = ['name', 'discipline']

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

class RatingViewSet(viewsets.ModelViewSet):
    queryset = Rating.objects.all()
    serializer_class = RatingSerializer
    permission_classes = [permissions.IsAuthenticated]
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_fields = ['professor', 'class_name', 'is_online', 'is_for_credit']
    ordering_fields = ['created_at', 'avg_rating', 'helpful_rating', 'clarity_rating']

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