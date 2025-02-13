from rest_framework import serializers
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from .models import Professor, Rating, Sentiment, UserRole

class UserRoleSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserRole
        fields = ['role', 'discipline']

class LoginSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField(write_only=True)

    def validate(self, data):
        user = authenticate(**data)
        if user and user.is_active:
            return user
        raise serializers.ValidationError("Incorrect credentials.")

class UserSerializer(serializers.ModelSerializer):
    role = UserRoleSerializer(read_only=True)
    
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 'role']
        read_only_fields = ['email']

class ProfessorSerializer(serializers.ModelSerializer):
    ratings_count = serializers.IntegerField(read_only=True)
    avg_rating = serializers.FloatField(read_only=True)
    avg_helpful = serializers.FloatField(read_only=True)
    avg_clarity = serializers.FloatField(read_only=True)
    avg_difficulty = serializers.FloatField(read_only=True)

    class Meta:
        model = Professor
        fields = ['id', 'professor_id', 'first_name', 'last_name', 'gender', 
                 'discipline', 'sub_discipline', 'ratings_count', 'avg_rating',
                 'avg_helpful', 'avg_clarity', 'avg_difficulty']

class RatingSerializer(serializers.ModelSerializer):
    professor_name = serializers.CharField(source='professor.first_name', read_only=True)

    class Meta:
        model = Rating
        fields = '__all__'

class SentimentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Sentiment
        fields = '__all__'