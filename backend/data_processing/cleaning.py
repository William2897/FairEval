# backend/data_processing/cleaning.py
def clean_data(df):
    """
    Applies cleaning operations on the input DataFrame and returns cleaned DataFrame
    """
    print(f"Initial DataFrame shape: {df.shape}")
    print("Available columns:", df.columns.tolist())

    # Create explicit copy to avoid chaining issues
    keep_columns = [
        'professor_id', 'department', 'first_name', 'last_name',
        'would_take_again_percent', 'avg_rating', 'rating_comment',
        'rating_flagStatus', 'rating_class', 'rating_helpfulRating',
        'rating_clarityRating', 'rating_isForOnlineClass', 
        'rating_difficultyRating', 'rating_isForCredit'
    ]
    
    # Verify columns exist
    missing_cols = [col for col in keep_columns if col not in df.columns]
    if missing_cols:
        print("Warning: Missing columns:", missing_cols)
        keep_columns = [col for col in keep_columns if col in df.columns]
    
    # Create a clean copy with selected columns
    cleaned_df = df[keep_columns].copy()

    # Fill missing values in rating_class
    if cleaned_df['rating_class'].notnull().sum() > 0:
        mode_rating_class = cleaned_df['rating_class'].mode()[0]
        cleaned_df['rating_class'] = cleaned_df['rating_class'].fillna(mode_rating_class)

    # Remove rows with missing comments
    cleaned_df = cleaned_df.dropna(subset=['rating_comment'])

    # Remove duplicate comments
    cleaned_df = cleaned_df.drop_duplicates(subset='rating_comment')

    # Convert professor_id to string
    cleaned_df['professor_id'] = cleaned_df['professor_id'].astype(str)

    # Normalize column names
    column_mapping = {
        'rating_flagStatus': 'flag_status',
        'rating_helpfulRating': 'helpful_rating',
        'rating_clarityRating': 'clarity_rating',
        'rating_isForOnlineClass': 'is_online',
        'rating_difficultyRating': 'difficulty_rating',
        'rating_isForCredit': 'is_for_credit'
    }
    cleaned_df = cleaned_df.rename(columns=column_mapping)

    print(f"Shape after cleaning: {cleaned_df.shape}")
    return cleaned_df

# from cleaning import clean_data
# df = pd.read_csv('path_to_your_csv_file.csv')
# cleaned_df = clean_data(df)
# cleaned_df.to_csv('path_to_cleaned_csv_file.csv', index=False)
