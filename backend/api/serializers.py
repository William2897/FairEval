from rest_framework import serializers
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from .models import Department, Professor, Rating, Sentiment, UserRole

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

class DepartmentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Department
        fields = '__all__'

class ProfessorSerializer(serializers.ModelSerializer):
    department_name = serializers.CharField(source='department.name', read_only=True)
    ratings_count = serializers.IntegerField(source='ratings.count', read_only=True)
    avg_helpful_rating = serializers.FloatField(source='ratings.aggregate.avg_helpful_rating', read_only=True)
    avg_clarity_rating = serializers.FloatField(source='ratings.aggregate.avg_clarity_rating', read_only=True)

    class Meta:
        model = Professor
        fields = '__all__'

class RatingSerializer(serializers.ModelSerializer):
    professor_name = serializers.CharField(source='professor.first_name', read_only=True)

    class Meta:
        model = Rating
        fields = '__all__'

class SentimentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Sentiment
        fields = '__all__'