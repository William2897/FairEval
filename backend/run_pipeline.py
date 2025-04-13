# backend/run_pipeline.py

import os
import django
from data_processing.pipeline import run_db_population

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

csv_file = "professors_75346.csv"

if __name__ == "__main__":
    run_full_pipeline(csv_file, db_config)

# processed_csv = "professors_75346_processed.csv"
# run_db_population(processed_csv, db_config)