from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import DepartmentViewSet, CourseViewSet, EvaluationViewSet, CommentViewSet

router = DefaultRouter()
router.register(r'departments', DepartmentViewSet)
router.register(r'courses', CourseViewSet)
router.register(r'evaluations', EvaluationViewSet, basename='evaluation')
router.register(r'comments', CommentViewSet)

urlpatterns = [
    path('', include(router.urls)),
]