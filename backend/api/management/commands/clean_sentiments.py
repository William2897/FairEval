# Save this as backend/api/management/commands/clean_sentiments.py
from django.core.management.base import BaseCommand
from api.models import Rating, Sentiment
from django.db.models import F, Subquery, OuterRef

class Command(BaseCommand):
    help = 'Clean up orphaned sentiment records'

    def handle(self, *args, **options):
        # Find sentiments that don't have a corresponding rating with the same professor and timestamp
        orphaned_sentiments = Sentiment.objects.exclude(
            professor_id__in=Subquery(
                Rating.objects.filter(
                    professor__professor_id=OuterRef('professor_id'),
                    created_at=OuterRef('created_at')
                ).values('professor__professor_id').distinct()
            )
        )
        
        count = orphaned_sentiments.count()
        orphaned_sentiments.delete()
        
        self.stdout.write(self.style.SUCCESS(f'Successfully deleted {count} orphaned sentiments'))