# backend//data_processing/gender_assignment.py

import re
import gender_guesser.detector as gender
from functools import lru_cache

def compile_patterns(keywords):
    escaped = [re.escape(w) for w in keywords]
    pattern = r'\b(' + '|'.join(escaped) + r')\b'
    return re.compile(pattern, re.IGNORECASE)

MALE_KEYWORDS = [
    'he', "he's", 'him', 'his', 'himself',
    'father', 'son', 'brother', 'uncle', 'grandfather', 'husband',
    'mr', 'sir', 'gentleman', 'male', 'man', 'gent',
]
FEMALE_KEYWORDS = [
    'she', "she's", 'her', 'hers', 'herself',
    'mother', 'daughter', 'sister', 'aunt', 'grandmother', 'wife',
    'mrs', 'ms', 'miss', 'lady', 'queen',
    'female', 'woman', 'gentlewoman',
]

male_pattern = compile_patterns(MALE_KEYWORDS)
female_pattern = compile_patterns(FEMALE_KEYWORDS)

detector = gender.Detector(case_sensitive=False)

def preprocess_comment_for_gender(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # remove punctuation except hyphens
    text = re.sub(r'[^\w\s-]', '', text)
    return text

def count_gender_words_regex(text, pattern):
    return len(pattern.findall(text))

def assign_gender_from_keywords(df):
    # Add columns for processed text
    df['processed_comment'] = df['rating_comment'].apply(preprocess_comment_for_gender)
    # Count occurrences
    df['male_count'] = df['processed_comment'].apply(lambda x: count_gender_words_regex(x, male_pattern))
    df['female_count'] = df['processed_comment'].apply(lambda x: count_gender_words_regex(x, female_pattern))
    
    # Aggregate counts per professor
    agg = df.groupby('professor_id').agg({'male_count':'sum', 'female_count':'sum'}).reset_index()

    def gender_logic(row):
        if row['male_count'] > row['female_count']:
            return 'Male'
        elif row['female_count'] > row['male_count']:
            return 'Female'
        elif row['male_count']>0 or row['female_count']>0:
            return 'Ambiguous'
        else:
            return 'Unknown'

    agg['gender_keyword'] = agg.apply(gender_logic, axis=1)
    return agg

@lru_cache(maxsize=None)
def cached_guess_gender(first_name):
    if not first_name:
        return 'Unknown'
    first_name = first_name.split()[0]
    guess = detector.get_gender(first_name)
    if guess in ['male','mostly_male']:
        return 'Male'
    elif guess in ['female','mostly_female']:
        return 'Female'
    return 'Unknown'

def guess_gender_by_name(row):
    # If row['first_name'] is missing, return 'Unknown'
    fname = row['first_name'] if isinstance(row['first_name'], str) else ''
    return cached_guess_gender(fname)

def combine_gender(row):
    """
    If the keyword-based gender is 'Unknown' or 'Ambiguous',
    we use the name-based guess. Otherwise keep keyword-based.
    """
    if row['gender_keyword'] in ['Unknown','Ambiguous']:
        return row['name_based_gender']
    else:
        return row['gender_keyword']

def engineer_gender(df):
    """Process gender assignment in memory"""
    # Step 1: Keyword-based
    agg_df = assign_gender_from_keywords(df)

    # Merge with original
    df = df.merge(agg_df[['professor_id','gender_keyword']], on='professor_id', how='left')

    # Step 2: Name-based guess
    df['name_based_gender'] = df.apply(guess_gender_by_name, axis=1)

    # Step 3: Final gender
    df['gender'] = df.apply(combine_gender, axis=1)

    # Drop intermediate columns and unknown gender
    df = df[df['gender']!='Unknown']
    drop_cols = ['processed_comment','male_count','female_count','gender_keyword','name_based_gender']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    print("Gender distribution:\n", df['gender'].value_counts())
    return df
