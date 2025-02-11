import re
import gc
import unicodedata
import pandas as pd
import spacy
import torch
from nltk.corpus import stopwords
import contractions
from tqdm import tqdm

# Simple memory-efficient cache
text_cache = {}

# Optimize spaCy pipeline - keep tagger for better lemmatization
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

def process_texts(texts, batch_size=10000):
    """Process texts in small batches to manage memory"""
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        cleaned_batch = [clean_text(text) for text in batch]
        batch_results = []
        
        # Group texts for processing
        to_process = []
        for idx, text in enumerate(cleaned_batch):
            if not text:
                batch_results.append("")
            else:
                text_hash = hash(text)
                if text_hash in text_cache:
                    batch_results.append(text_cache[text_hash])
                else:
                    batch_results.append(None)
                    to_process.append((idx, text))
        
        # Process uncached texts
        if to_process:
            try:
                indices, process_texts = zip(*to_process)
                docs = list(nlp.pipe(process_texts))  # Convert to list for safe iteration
                
                for idx, doc in zip(indices, docs):
                    processed = ' '.join(token.lemma_ for token in doc)
                    text_hash = hash(process_texts[indices.index(idx)])  # Fix indexing
                    text_cache[text_hash] = processed
                    batch_results[idx] = processed
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                # Fill any remaining None values with empty strings
                batch_results = [result if result is not None else "" for result in batch_results]
        
        results.extend(batch_results)
        
        # Memory management
        if i % 10000 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return results

def preprocess_comments(df):
    """Process comments in chunks to manage memory"""
    try:
        print(f"Processing {len(df)} comments...")
        chunk_size = 10000  # Reduced chunk size for better memory management
        all_processed = []
        
        for start in tqdm(range(0, len(df), chunk_size)):
            end = min(start + chunk_size, len(df))  # Ensure we don't exceed df length
            chunk = df.iloc[start:end]
            
            try:
                comments = chunk['rating_comment'].fillna('').tolist()
                processed = process_texts(comments)
                all_processed.extend(processed)
            except Exception as e:
                print(f"Error processing chunk {start}-{end}: {str(e)}")
                # Add empty strings for failed chunk
                all_processed.extend([""] * len(chunk))
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        df['processed_comment'] = all_processed
        return df
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise
    finally:
        text_cache.clear()
        gc.collect()

if __name__ == '__main__':
    # Clear GPU memory if using CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
