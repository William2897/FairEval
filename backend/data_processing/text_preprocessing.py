import re
import gc
import unicodedata
import pandas as pd
import spacy
import torch
from nltk.corpus import stopwords, opinion_lexicon
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import contractions
from tqdm import tqdm
from collections import Counter

# Initialize VADER
sia = SentimentIntensityAnalyzer()

# Initialize opinion lexicon sets
positive_words = set(opinion_lexicon.positive())
negative_words = set(opinion_lexicon.negative())

# Optimize spaCy pipeline
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "textcat"])
nlp.max_length = 1000000
nlp.add_pipe('sentencizer')

# Domain stopwords
DOMAIN_STOP_WORDS = {
    'professor','prof','teacher','lecturer','student','class','course','lecture','dr',
    'section','ratemyprofessor','rmp','review','rating','comment','feedback'
}

STOP_WORDS = set(stopwords.words('english')).union(DOMAIN_STOP_WORDS)
CLEAN_PATTERN = re.compile(r'[^a-zA-Z\s]')

def clean_text(text):
    """Memory efficient text cleaning"""
    if not isinstance(text, str) or pd.isna(text):
        return ""
    text = text.lower().strip()
    if len(text) < 3:
        return ""
    text = contractions.fix(text)
    text = unicodedata.normalize('NFKD', text).encode('ascii','ignore').decode('utf-8')
    cleaned = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return ' '.join(w for w in cleaned.split() if w not in STOP_WORDS)

def convert_to_list(text):
    """Convert text to list if it's not already"""
    if isinstance(text, list):
        return text
    elif isinstance(text, str):
        return text.split()
    return []

def extract_opinion_terms(tokens):
    """Extract positive and negative terms from tokens using NLTK opinion lexicon"""
    tokens = convert_to_list(tokens)
    pos_terms = [word for word in tokens if word in positive_words]
    neg_terms = [word for word in tokens if word in negative_words]
    return pos_terms, neg_terms

def extract_vader_terms(text):
    """Extract positive and negative terms based on VADER scores"""
    if isinstance(text, list):
        text = ' '.join(text)
    tokens = text.split()
    negative_words = [word for word in tokens if sia.polarity_scores(word)['compound'] <= -0.5]
    positive_words = [word for word in tokens if sia.polarity_scores(word)['compound'] >= 0.5]
    return positive_words, negative_words

def process_texts(texts, batch_size=10000):
    """Process texts in small batches to manage memory"""
    results = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing texts"):
        batch = texts[i:i + batch_size]
        cleaned_batch = [clean_text(text) for text in batch]
        processed_batch = []
        
        for text in cleaned_batch:
            if not text:
                processed_batch.append("")
                continue
                
            try:
                doc = nlp(text)
                processed = ' '.join(token.lemma_ for token in doc)
                processed_batch.append(processed)
            except Exception as e:
                print(f"Error processing text: {str(e)}")
                processed_batch.append("")
        
        results.extend(processed_batch)
        
        if i % 10000 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return results

def preprocess_comments(df):
    """Process comments and extract sentiment terms"""
    try:
        print(f"Processing {len(df)} comments...")
        
        # Debug: Check for missing comments before processing
        print(f"[DEBUG] Missing comments: {df['rating_comment'].isna().sum()} out of {len(df)}")
        print(f"[DEBUG] Empty comments: {(df['rating_comment'] == '').sum()} out of {len(df)}")
        print(f"[DEBUG] Comment sample: {df['rating_comment'].head(3).tolist()}")
        
        # Process comments
        comments = df['rating_comment'].fillna('').tolist()
        print(f"[DEBUG] Comments to process: {len(comments)}")
        processed_comments = process_texts(comments)
        print(f"[DEBUG] Processed comments returned: {len(processed_comments)}")
        df['processed_comment'] = processed_comments
        
        # Extract sentiment terms
        df['processed_tokens'] = df['processed_comment'].apply(convert_to_list)
        
        # Extract lexicon terms
        lexicon_terms = df['processed_tokens'].apply(extract_opinion_terms)
        df['positive_terms_lexicon'] = [terms[0] for terms in lexicon_terms]
        df['negative_terms_lexicon'] = [terms[1] for terms in lexicon_terms]
        
        # Extract VADER terms
        vader_terms = df['processed_comment'].apply(extract_vader_terms)
        df['positive_terms_vader'] = [terms[0] for terms in vader_terms]
        df['negative_terms_vader'] = [terms[1] for terms in vader_terms]
        
        # Calculate term frequencies
        positive_counter = Counter()
        negative_counter = Counter()
        vader_pos_counter = Counter()
        vader_neg_counter = Counter()
        
        # Update counters with progress bar
        for idx in tqdm(range(len(df)), desc="Calculating term frequencies"):
            positive_counter.update(df['positive_terms_lexicon'].iloc[idx])
            negative_counter.update(df['negative_terms_lexicon'].iloc[idx])
            vader_pos_counter.update(df['positive_terms_vader'].iloc[idx])
            vader_neg_counter.update(df['negative_terms_vader'].iloc[idx])
        
        return df, positive_counter, negative_counter, vader_pos_counter, vader_neg_counter
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"Number of positive words in lexicon: {len(positive_words)}")
    print(f"Number of negative words in lexicon: {len(negative_words)}")
