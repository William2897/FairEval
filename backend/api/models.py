from django.db import models
from django.contrib.auth.models import User

class Professor(models.Model):
    professor_id = models.CharField(max_length=50, unique=True)
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    gender = models.CharField(max_length=10, null=True)
    discipline = models.CharField(max_length=100, db_index=True)
    sub_discipline = models.CharField(max_length=100, null=True, db_index=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['discipline', 'sub_discipline']),
            models.Index(fields=['gender', 'discipline']),
        ]
        ordering = ['last_name', 'first_name']

    def __str__(self):
        return f"{self.first_name} {self.last_name} ({self.discipline})"

class Rating(models.Model):
    professor = models.ForeignKey(
        Professor, 
        to_field='professor_id',  # Reference the business key
        db_column='professor_id', # Use consistent column name  
        on_delete=models.CASCADE, 
        related_name='ratings'
    )
    avg_rating = models.FloatField(default=0.0, db_index=True)
    flag_status = models.CharField(max_length=50, null=True, db_index=True)
    helpful_rating = models.FloatField(null=True, db_index=True)
    clarity_rating = models.FloatField(null=True, db_index=True)
    difficulty_rating = models.FloatField(null=True, db_index=True)
    is_online = models.BooleanField(default=False, db_index=True)
    is_for_credit = models.BooleanField(default=False, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        indexes = [
            models.Index(fields=['professor', 'created_at']),
            models.Index(fields=['avg_rating', 'helpful_rating', 'clarity_rating', 'difficulty_rating']),
        ]
        ordering = ['-created_at']  # Default ordering by newest first

    def __str__(self):
        return f"{self.professor} - {self.avg_rating} - {self.flag_status}"

class Sentiment(models.Model):
    professor = models.ForeignKey(
        Professor,
        to_field='professor_id',  # Reference the business key
        db_column='professor_id', # Use consistent column name
        on_delete=models.CASCADE,
        related_name='sentiments'
    )
    comment = models.TextField(null=True)
    processed_comment = models.TextField(null=True)
    sentiment = models.FloatField(null=True, db_index=True)
    confidence = models.FloatField(null=True)
    comment_topic = models.CharField(max_length=100, null=True, db_index=True)
    vader_compound = models.FloatField(null=True)
    vader_positive = models.FloatField(null=True)
    vader_negative = models.FloatField(null=True)  
    vader_neutral = models.FloatField(null=True)
    positive_terms = models.JSONField(null=True)
    negative_terms = models.JSONField(null=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        indexes = [
            models.Index(fields=['professor', 'sentiment']),
            models.Index(fields=['professor', 'created_at']),
        ]

    def __str__(self):
        return f"{self.professor} - {self.sentiment}"

class UserRole(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='role')
    role = models.CharField(
        max_length=10,
        choices=[('ADMIN', 'Administrator'), ('ACADEMIC', 'Academic Staff')],
        default='ACADEMIC'
    )
    discipline = models.CharField(  # Changed from department ForeignKey
        max_length=200,
        null=True,
        blank=True
    )

    class Meta:
        verbose_name = 'User Role'
        verbose_name_plural = 'User Roles'

    def __str__(self):
        return f"{self.user.username} - {self.role}"