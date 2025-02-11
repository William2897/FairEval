# backend/fair_eval_backend/run_pipeline.py

import os
import django

# Set up Django environment before importing settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'faireval.settings')
django.setup()

# settings after Django is configured
from django.conf import settings
from data_processing.pipeline import run_full_pipeline

db_config = {
    'dbname': settings.DATABASES['default']['NAME'],
    'user': settings.DATABASES['default']['USER'],
    'password': settings.DATABASES['default']['PASSWORD'],
    'host': settings.DATABASES['default']['HOST'],
    'port': settings.DATABASES['default']['PORT']
}

# Remove extra quotes from the CSV path
csv_file = "professors_75346.csv"

if __name__ == "__main__":
    run_full_pipeline(csv_file, db_config)