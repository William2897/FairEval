import os
import sys
import django
from django.conf import settings
import json
import numpy as np


import torch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'faireval.settings') # Ensure 'faireval' is your Django project name
try:
    django.setup()
    print("Django setup successful.")
except Exception as e:
    print(f"Django setup failed: {e}. Some features like DB access might not work, but explainer should load.")

# Use absolute import instead of relative import
from machine_learning.gender_bias_explainer import GenderBiasExplainer
from machine_learning.ml_model_dev.lstm import CustomSentimentLSTM
from data_processing.text_preprocessing import clean_text # Import the actual clean_text

TARGET_DEBUG_COMMENT_SUBSTRING_WARSI = "Professor Warsi"
TARGET_DEBUG_COMMENT_SUBSTRING_KNOWLEDGEABLE = "clearly knowledgeable"

def print_analysis_details(method_name, comment_text_identifier, result_dict, original_raw_text="N/A"):
    """Helper function to print detailed analysis results."""
    # Use a more robust check for the target comment if needed
    # For now, using the identifier passed (which could be the raw text or a substring)
    should_print = False
    if TARGET_DEBUG_COMMENT_SUBSTRING_WARSI in comment_text_identifier or \
       TARGET_DEBUG_COMMENT_SUBSTRING_KNOWLEDGEABLE in comment_text_identifier:
        should_print = True

    if not should_print:
        return

    print(f"\n--- DETAILED ANALYSIS from {method_name} for comment containing '{comment_text_identifier}' ---")
    if original_raw_text != "N/A":
        print(f"Original Raw Text: '{original_raw_text}'")
    print(f"Prediction: {result_dict.get('prediction')}, Confidence: {result_dict.get('confidence')}")
    
    tokens = result_dict.get('tokens', [])
    print(f"Tokens ({len(tokens)}): {tokens}")
    
    attention_weights = result_dict.get('attention', [])
    if isinstance(attention_weights, np.ndarray): # If it's still numpy array
        attention_weights_list = attention_weights.tolist()
    else:
        attention_weights_list = attention_weights

    print(f"Attention Weights ({len(attention_weights_list)}, sum: {np.sum(attention_weights_list):.4f}): {attention_weights_list}")

    gender_bias_info = result_dict.get('gender_bias', {})
    print(f"Interpretation: {gender_bias_info.get('interpretation')}")
    print(f"Stereotype Bias Score: {gender_bias_info.get('stereotype_bias_score')}")
    
    category_attention_pct = gender_bias_info.get('category_attention_pct', {})
    print(f"Category Attention Pct: {category_attention_pct}")
    print(f"  Objective Pct: {category_attention_pct.get('objective_pedagogical', 0):.1f}%")
    print(f"  Intellect Pct: {category_attention_pct.get('intellect_achievement', 0):.1f}%")
    print(f"  Entertainment Pct: {category_attention_pct.get('entertainment_authority', 0):.1f}%")
    print(f"  Competence Pct: {category_attention_pct.get('competence_organization', 0):.1f}%")
    print(f"  Warmth Pct: {category_attention_pct.get('warmth_nurturing', 0):.1f}%")
    print(f"  Male Negative Pct: {category_attention_pct.get('male_negative', 0):.1f}%")
    print(f"  Female Negative Pct: {category_attention_pct.get('female_negative', 0):.1f}%")
    print("--- END DETAILED ANALYSIS ---")


def run_debug_test():
    print("Attempting to load actual trained model and vocab.")
    # ---Paths to your trained model and vocab ---
    model_path = os.path.join(settings.BASE_DIR, 'machine_learning/ml_models_trained/lstm_sentiment.pt')
    vocab_path = os.path.join(settings.BASE_DIR, 'machine_learning/ml_models_trained/vocab.json')

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return
    if not os.path.exists(vocab_path):
        print(f"ERROR: Vocab file not found at {vocab_path}")
        return

    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        vocab_size = len(vocab)
        print(f"Vocabulary loaded successfully. Size: {vocab_size}")

        # --- Initialize model with parameters matching your saved model's architecture ---
        # Adjust these if your trained model used different parameters
        model = CustomSentimentLSTM(
            vocab_size=vocab_size,
            embed_dim=128,     # Ensure this matches your training
            hidden_dim=256,    # Ensure this matches your training
            num_layers=2,      # Ensure this matches your training
            dropout=0.5        # Ensure this matches your training
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Trained model loaded successfully onto {device}.")

    except Exception as e:
        print(f"ERROR loading trained model/vocab: {e}")
        import traceback
        print(traceback.format_exc())
        return

    explainer = GenderBiasExplainer(model, vocab)

    # --- Define Test Comments ---
    comment1_raw = "Professor Warsi is brilliant! His lectures are not only informative but also incredibly entertaining and funny. Made English enjoyable. Truly awesome."
    comment2_raw = "He's clearly knowledgeable, but sometimes came across as a bit distant or too theoretical for an intro English class. Could be more engaging."
    comment3_raw = "This professor is average at best, and lectures were often boring." # Another example


    # --- Test explain_prediction (uses internal clean_text) ---
    print("\n\n--- Testing explain_prediction ---")

    # Test comment 1
    print(f"\nProcessing single (Comment 1 - Warsi): '{comment1_raw}'")
    # The `explain_prediction` now internally calls `clean_text(comment1_raw)`
    # and then `_tokenize_and_pad_batch` uses the result of that.
    result_single1 = explainer.explain_prediction(
        text=comment1_raw,
        selected_gender="Male",
        discipline="English"
    )
    print_analysis_details("explain_prediction", TARGET_DEBUG_COMMENT_SUBSTRING_WARSI, result_single1, original_raw_text=comment1_raw)

    # Test comment 2
    print(f"\nProcessing single (Comment 2 - Knowledgeable): '{comment2_raw}'")
    result_single2 = explainer.explain_prediction(
        text=comment2_raw,
        selected_gender="Male",
        discipline="English"
    )
    print_analysis_details("explain_prediction", TARGET_DEBUG_COMMENT_SUBSTRING_KNOWLEDGEABLE, result_single2, original_raw_text=comment2_raw)


    # --- Test explain_batch (simulate pipeline's pre-cleaned input) ---
    print("\n\n--- Testing explain_batch ---")

    # Pre-clean comments for batch, similar to how pipeline would prepare 'processed_comment'
    comment1_cleaned_for_batch = clean_text(comment1_raw)
    comment2_cleaned_for_batch = clean_text(comment2_raw)
    comment3_cleaned_for_batch = clean_text(comment3_raw)

    print(f"\nInput to _tokenize_and_pad_batch (via explain_batch) for Comment 1 (Warsi): '{comment1_cleaned_for_batch}'")
    print(f"Input to _tokenize_and_pad_batch (via explain_batch) for Comment 2 (Knowledgeable): '{comment2_cleaned_for_batch}'")

    batch_texts = [comment1_cleaned_for_batch, comment2_cleaned_for_batch, comment3_cleaned_for_batch]
    batch_genders = ["Male", "Male", "Female"]
    batch_disciplines = ["English", "English", "Mathematics"]

    batch_results = explainer.explain_batch(
        texts=batch_texts,
        selected_genders=batch_genders,
        disciplines=batch_disciplines
    )

    print("\nOverall results from explain_batch (first 3 comments):")
    # Print details for the comments we're interested in
    if len(batch_results) > 0:
        # The text passed to print_analysis_details for matching is the raw form,
        # as batch_results[0] corresponds to comment1_raw
        print_analysis_details("explain_batch", TARGET_DEBUG_COMMENT_SUBSTRING_WARSI, batch_results[0], original_raw_text=comment1_raw)
    if len(batch_results) > 1:
        print_analysis_details("explain_batch", TARGET_DEBUG_COMMENT_SUBSTRING_KNOWLEDGEABLE, batch_results[1], original_raw_text=comment2_raw)
    if len(batch_results) > 2: # Just to show the third one if needed
        print(f"\nFull result for 3rd batch comment (raw: '{comment3_raw}'):")
        print(batch_results[2])


if __name__ == "__main__":
    run_debug_test()