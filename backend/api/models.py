from django.db import models
from django.contrib.auth.models import User

class Department(models.Model):
    name = models.CharField(max_length=200, unique=True, db_index=True)
    discipline = models.CharField(max_length=200, db_index=True)
    sub_discipline = models.CharField(max_length=200, null=True, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=['discipline', 'sub_discipline']),
        ]

    def __str__(self):
        return f"{self.name} - {self.discipline} - {self.sub_discipline}"

class Professor(models.Model):
    professor_id = models.CharField(max_length=50, unique=True, null=True, blank=True)
    first_name = models.CharField(max_length=100, db_index=True)
    last_name = models.CharField(max_length=100, db_index=True)
    gender = models.CharField(max_length=50, null=True, db_index=True)
    department = models.ForeignKey(Department, on_delete=models.SET_NULL, null=True)
    would_take_again_percent = models.FloatField(null=True)

    class Meta:
        indexes = [
            models.Index(fields=['department', 'gender']),
            models.Index(fields=['last_name', 'first_name']),
        ]

    def __str__(self):
        return f"{self.first_name} - {self.last_name} - {self.gender} - {self.department}"

class Rating(models.Model):
    professor = models.ForeignKey(Professor, on_delete=models.CASCADE, related_name='ratings')
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
            models.Index(fields=['avg_rating', 'helpful_rating', 'clarity_rating']),
        ]

    def __str__(self):
        return f"{self.professor} - {self.avg_rating} - {self.flag_status}"

class Sentiment(models.Model):
    professor = models.ForeignKey(User, on_delete=models.CASCADE, related_name='sentiment')
    comment = models.TextField(null=True)
    processed_comment = models.TextField(null=True)
    sentiment = models.IntegerField(null=True, db_index=True)
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
    department = models.ForeignKey(
        Department, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True
    )

    class Meta:
        verbose_name = 'User Role'
        verbose_name_plural = 'User Roles'

    def __str__(self):
        return f"{self.user.username} - {self.role}"