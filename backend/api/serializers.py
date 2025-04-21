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
    professor_name = serializers.SerializerMethodField()
    discipline = serializers.CharField(source='professor.discipline', read_only=True)
    sub_discipline = serializers.CharField(source='professor.sub_discipline', read_only=True)
    comment = serializers.SerializerMethodField()

    def get_professor_name(self, obj):
        return f"{obj.professor.first_name} {obj.professor.last_name}"
    
    def get_comment(self, obj):
        # Get associated comment from Sentiment model based on professor_id and rating ID to ensure uniqueness
        from .models import Sentiment
        
        try:
            # Use the rating's ID to create a deterministic offset for fetching comments
            rating_index = obj.id % 1000  # Use modulo to keep numbers manageable
            
            # Get all sentiments for this professor
            sentiments = Sentiment.objects.filter(
                professor_id=obj.professor.professor_id
            ).order_by('id')
            
            # If we have sentiments, select one based on the rating's ID
            if sentiments.exists():
                count = sentiments.count()
                # Use the rating ID to determine which sentiment to show
                index = rating_index % count
                sentiment = sentiments[index]
                return sentiment.comment if sentiment.comment else ""
            return ""
        except Exception:
            return ""

    class Meta:
        model = Rating
        fields = ['id', 'professor', 'professor_name', 'discipline', 'sub_discipline', 
                 'avg_rating', 'flag_status', 'helpful_rating', 'clarity_rating', 
                 'difficulty_rating', 'is_online', 'is_for_credit', 'created_at', 'comment']

class SentimentSerializer(serializers.ModelSerializer):
    # Add the new fields here, mark as read_only
    bias_tag = serializers.CharField(read_only=True)
    bias_interpretation = serializers.CharField(read_only=True)
    stereotype_bias_score = serializers.FloatField(read_only=True)
    objective_focus_percentage = serializers.FloatField(read_only=True)
    class Meta:
        model = Sentiment
        fields = '__all__'
