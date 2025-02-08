from rest_framework import viewsets, permissions, filters
from django_filters.rest_framework import DjangoFilterBackend
from .models import Department, Course, Evaluation, Comment
from .serializers import (
    DepartmentSerializer, CourseSerializer,
    EvaluationSerializer, CommentSerializer
)

class DepartmentViewSet(viewsets.ModelViewSet):
    queryset = Department.objects.all()
    serializer_class = DepartmentSerializer
    permission_classes = [permissions.IsAuthenticated]
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ['name', 'code']
    ordering_fields = ['name', 'code', 'created_at']

class CourseViewSet(viewsets.ModelViewSet):
    queryset = Course.objects.all()
    serializer_class = CourseSerializer
    permission_classes = [permissions.IsAuthenticated]
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ['department']
    search_fields = ['code', 'title', 'description']
    ordering_fields = ['code', 'title', 'created_at']

class EvaluationViewSet(viewsets.ModelViewSet):
    serializer_class = EvaluationSerializer
    permission_classes = [permissions.IsAuthenticated]
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ['course', 'professor', 'semester', 'year']
    search_fields = ['course__title', 'professor__username']
    ordering_fields = ['year', 'semester', 'teaching_effectiveness', 'created_at']

    def get_queryset(self):
        user = self.request.user
        if user.is_staff:
            return Evaluation.objects.all()
        return Evaluation.objects.filter(professor=user)

class CommentViewSet(viewsets.ModelViewSet):
    queryset = Comment.objects.all()
    serializer_class = CommentSerializer
    permission_classes = [permissions.IsAuthenticated]
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_fields = ['evaluation']
    ordering_fields = ['created_at', 'sentiment_score']