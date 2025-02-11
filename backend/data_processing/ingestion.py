# backend/data_processing/ingestion.py

import pandas as pd

# these are the actual CSV columns → DB columns mapping
CSV_COLUMNS = [
    'professor_id', 'department', 'first_name', 'last_name',
    'would_take_again_percent', 'avg_rating', 'rating_comment',
    'rating_flagStatus', 'rating_class', 'rating_helpfulRating',
    'rating_clarityRating', 'rating_isForOnlineClass',
    'rating_difficultyRating', 'rating_isForCredit'
]

def ingest_csv_to_df(csv_path, chunksize=50000):
    """
    Reads CSV file and returns DataFrame with properly typed columns
    """
    # Read CSV
    df = pd.read_csv(csv_path, encoding='utf-8', dtype=str)
    
    # Convert numeric columns
    numeric_cols = [
        'would_take_again_percent', 'avg_rating', 'rating_helpfulRating',
        'rating_clarityRating', 'rating_difficultyRating'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert boolean columns
    bool_cols = ['rating_isForOnlineClass', 'rating_isForCredit']
    for col in bool_cols:
        if col in bool_cols:
            df[col] = df[col].map({'True': True, 'False': False, '1': True, '0': False})
    
    print(f"Ingested {len(df)} rows from CSV")
    return df


# Example usage:
# from ingestion import ingest_csv_to_df
#
# df = ingest_csv_to_df('/path/to/raw_rmp_data.csv')
