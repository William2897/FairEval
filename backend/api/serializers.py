from rest_framework import serializers
from django.contrib.auth.models import User
from .models import Department, Course, Evaluation, Comment

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name']

class DepartmentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Department
        fields = '__all__'

class CourseSerializer(serializers.ModelSerializer):
    department_name = serializers.CharField(source='department.name', read_only=True)

    class Meta:
        model = Course
        fields = '__all__'

class CommentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Comment
        fields = '__all__'

class EvaluationSerializer(serializers.ModelSerializer):
    comments = CommentSerializer(many=True, read_only=True)
    course_title = serializers.CharField(source='course.title', read_only=True)
    professor_name = serializers.SerializerMethodField()

    class Meta:
        model = Evaluation
        fields = '__all__'

    def get_professor_name(self, obj):
        return f"{obj.professor.first_name} {obj.professor.last_name}"