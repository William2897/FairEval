from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    DepartmentViewSet, ProfessorViewSet, 
    RatingViewSet, SentimentViewSet, AuthViewSet,
    UserViewSet, UserRoleViewSet  # Add these
)

router = DefaultRouter()
router.register(r'auth', AuthViewSet, basename='auth')
router.register(r'departments', DepartmentViewSet)
router.register(r'professors', ProfessorViewSet)
router.register(r'ratings', RatingViewSet)
router.register(r'sentiments', SentimentViewSet)
router.register(r'users', UserViewSet)  # Add this
router.register(r'roles', UserRoleViewSet)  # Add this

urlpatterns = [
    path('', include(router.urls)),
]