from django.db import migrations, models

class Migration(migrations.Migration):
    dependencies = [
        ('api', '0005_sentiment_negative_terms_sentiment_positive_terms_and_more'),
    ]

    operations = [
        # First update gender field to handle longer values
        migrations.AlterField(
            model_name='professor',
            name='gender',
            field=models.CharField(max_length=10, null=True),
        ),
        # Add new discipline fields
        migrations.AddField(
            model_name='professor',
            name='discipline',
            field=models.CharField(default='Interdisciplinary Studies', max_length=100, db_index=True),
        ),
        migrations.AddField(
            model_name='professor',
            name='sub_discipline',
            field=models.CharField(max_length=100, null=True, db_index=True),
        ),
        # Remove old fields and update indexes
        migrations.RemoveField(
            model_name='professor',
            name='department',
        ),
        migrations.RemoveField(
            model_name='professor',
            name='would_take_again_percent',
        ),
        migrations.AddIndex(
            model_name='professor',
            index=models.Index(fields=['discipline', 'sub_discipline'], name='api_profess_discipl_a763ff_idx'),
        ),
        migrations.AddIndex(
            model_name='professor',
            index=models.Index(fields=['gender', 'discipline'], name='api_profess_gender_4edddf_idx'),
        ),
    ]