from django.db import models
from django.contrib.auth.models import User

class Department(models.Model):
    name = models.CharField(max_length=100)
    code = models.CharField(max_length=10, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.code} - {self.name}"

class Course(models.Model):
    department = models.ForeignKey(Department, on_delete=models.CASCADE, related_name='courses')
    code = models.CharField(max_length=20)
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ['department', 'code']

    def __str__(self):
        return f"{self.department.code} {self.code} - {self.title}"

class Evaluation(models.Model):
    SEMESTER_CHOICES = [
        ('FALL', 'Fall'),
        ('SPRING', 'Spring'),
        ('SUMMER', 'Summer'),
    ]

    course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name='evaluations')
    professor = models.ForeignKey(User, on_delete=models.CASCADE, related_name='evaluations')
    semester = models.CharField(max_length=10, choices=SEMESTER_CHOICES)
    year = models.IntegerField()
    response_count = models.IntegerField(default=0)
    
    # Numerical metrics
    teaching_effectiveness = models.DecimalField(max_digits=3, decimal_places=2, null=True)
    course_content = models.DecimalField(max_digits=3, decimal_places=2, null=True)
    workload_fairness = models.DecimalField(max_digits=3, decimal_places=2, null=True)
    
    # Processed metrics
    sentiment_score = models.DecimalField(max_digits=4, decimal_places=3, null=True)
    bias_score = models.DecimalField(max_digits=4, decimal_places=3, null=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ['course', 'professor', 'semester', 'year']

    def __str__(self):
        return f"{self.course} - {self.semester} {self.year}"

class Comment(models.Model):
    evaluation = models.ForeignKey(Evaluation, on_delete=models.CASCADE, related_name='comments')
    text = models.TextField()
    sentiment_score = models.DecimalField(max_digits=4, decimal_places=3, null=True)
    topics = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Comment for {self.evaluation} - {self.created_at.date()}"