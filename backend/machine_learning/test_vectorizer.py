from joblib import load, dump
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def test_and_repair_vectorizer(vectorizer_path, repair_suffix="_repaired"):
    print(f"\nTesting vectorizer: {vectorizer_path}")
    vectorizer = load(vectorizer_path)

    print("\nVectorizer attributes:")
    print("Has idf_:", hasattr(vectorizer, 'idf_'))
    if hasattr(vectorizer, 'idf_'):
        print("idf_ shape:", vectorizer.idf_.shape)
        print("idf_ type:", type(vectorizer.idf_))
        print("idf_ sample:", vectorizer.idf_[:5])

    print("\nTrying to check internal _tfidf attribute:")
    if hasattr(vectorizer, '_tfidf'):
        print("Has _tfidf:", True)
        print("_tfidf attributes:", dir(vectorizer._tfidf))
        if hasattr(vectorizer._tfidf, 'idf_'):
            print("_tfidf has idf_:", True)
            print("_tfidf.idf_ shape:", vectorizer._tfidf.idf_.shape)
        else:
            print("_tfidf does not have idf_ attribute")
    else:
        print("No _tfidf attribute found")

    print("\nOther important attributes:")
    print("vocabulary size:", len(vectorizer.vocabulary_) if hasattr(vectorizer, 'vocabulary_') else "No vocabulary")
    print("stop_words:", "Yes" if hasattr(vectorizer, 'stop_words_') else "No")

    # Try to transform a simple text
    test_text = ["This is a test sentence"]
    try:
        transformed = vectorizer.transform(test_text)
        print("\nSuccessfully transformed test text!")
        print("Transformed shape:", transformed.shape)
        needs_repair = False
    except Exception as e:
        print("\nError transforming text:", str(e))
        needs_repair = True
    
    # Try to re-save the vectorizer with explicitly set idf_
    if needs_repair and hasattr(vectorizer, 'vocabulary_'):
        print("\nAttempting to repair vectorizer...")
        # Create a new vectorizer and fit it on a dummy text to initialize idf_
        new_vectorizer = TfidfVectorizer(vocabulary=vectorizer.vocabulary_)
        new_vectorizer.fit(["dummy text to initialize idf"])
        
        # Save the repaired vectorizer
        repair_path = os.path.splitext(vectorizer_path)[0] + repair_suffix + os.path.splitext(vectorizer_path)[1]
        dump(new_vectorizer, repair_path)
        print(f"Saved repaired vectorizer to: {repair_path}")
        return repair_path
    return None

# Get the absolute path to the vectorizer files
current_dir = os.path.dirname(os.path.abspath(__file__))
rf_vectorizer_path = os.path.join(current_dir, 'ml_models_trained', 'rf_tfidf_vectorizer.joblib')
svc_vectorizer_path = os.path.join(current_dir, 'ml_models_trained', 'svc_tfidf_vectorizer.joblib')

# Test and repair both vectorizers
rf_repaired_path = test_and_repair_vectorizer(rf_vectorizer_path)
svc_repaired_path = test_and_repair_vectorizer(svc_vectorizer_path)