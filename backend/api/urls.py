from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'professors', views.ProfessorViewSet)
router.register(r'ratings', views.RatingViewSet)
router.register(r'sentiments', views.SentimentViewSet)
router.register(r'users', views.UserViewSet)
router.register(r'user-roles', views.UserRoleViewSet)
router.register(r'auth', views.AuthViewSet, basename='auth')
router.register(r'topics', views.TopicViewSet, basename='topics')
router.register(r'sentiment-explainability', views.SentimentExplainabilityViewSet, basename='sentiment-explainability')


urlpatterns = [
    path('', include(router.urls)),
]