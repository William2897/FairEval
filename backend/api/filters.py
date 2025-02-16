from django_filters import rest_framework as filters
from django.db.models import Avg, Q
from .models import Rating, Professor

class RatingFilter(filters.FilterSet):
    start_date = filters.DateTimeFilter(field_name='created_at', lookup_expr='gte')
    end_date = filters.DateTimeFilter(field_name='created_at', lookup_expr='lte')
    min_rating = filters.NumberFilter(field_name='avg_rating', lookup_expr='gte')
    max_rating = filters.NumberFilter(field_name='avg_rating', lookup_expr='lte')
    professor = filters.NumberFilter(field_name='professor__id')
    search = filters.CharFilter(method='filter_search')

    def filter_search(self, queryset, name, value):
        return queryset.filter(
            Q(professor__first_name__icontains=value) |
            Q(professor__last_name__icontains=value) |
            Q(professor__discipline__icontains=value) |
            Q(professor__sub_discipline__icontains=value)
        )

    class Meta:
        model = Rating
        fields = ['professor', 'is_online', 'is_for_credit', 
                 'start_date', 'end_date', 'min_rating', 'max_rating']

class ProfessorFilter(filters.FilterSet):
    min_rating = filters.NumberFilter(method='filter_avg_rating')
    max_rating = filters.NumberFilter(method='filter_avg_rating')
    discipline_name = filters.CharFilter(field_name='discipline__name', lookup_expr='icontains')

    class Meta:
        model = Professor
        fields = ['gender', 'discipline', 'discipline_name']

    def filter_avg_rating(self, queryset, name, value):
        lookup = '__gte' if name == 'min_rating' else '__lte'
        return queryset.annotate(
            avg_rating=Avg('ratings__avg_rating')
        ).filter(**{f'avg_rating{lookup}': value})