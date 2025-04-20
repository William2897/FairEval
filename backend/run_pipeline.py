# backend/run_pipeline.py

import os
import django
from data_processing.pipeline import run_initial_processing, run_bias_analysis_only, run_db_population

# Set up Django environment before importing settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'faireval.settings')
django.setup()

# settings after Django is configured
from django.conf import settings

db_config = {
    'dbname': settings.DATABASES['default']['NAME'],
    'user': settings.DATABASES['default']['USER'],
    'password': settings.DATABASES['default']['PASSWORD'],
    'host': settings.DATABASES['default']['HOST'],
    'port': settings.DATABASES['default']['PORT']
}

# Step 1 and 2: Initial preprocessing of the CSV file
# raw_csv = "professors_75346.csv"
# processed_csv = run_initial_processing(raw_csv)
# if processed_csv:
#     print(f"Initial processing complete. Result in: {processed_csv}")

if __name__ == "__main__":
    processed_csv = "professors_75346_processed.csv" # Use the output from step 1 and
    output_bias_csv = "professors_75346_with_bias.csv"
    #final_csv = run_bias_analysis_only(processed_csv, output_bias_csv, chunk_size=1000)
    run_db_population(output_bias_csv, db_config)

# processed_csv = "professors_75346_processed.csv"
# run_db_population(processed_csv, db_config)