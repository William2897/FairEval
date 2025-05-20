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
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'faireval.settings')
try:
    django.setup()
    print("Django setup successful.")
except Exception as e:
    print(f"Django setup failed: {e}")

from machine_learning.gender_bias_explainer import GenderBiasExplainer
from machine_learning.ml_model_dev.lstm import CustomSentimentLSTM
from data_processing.text_preprocessing import get_fully_processed_text_for_explainer # NEW

TARGET_DEBUG_COMMENT_SUBSTRING_WARSI = "Professor Warsi"

# Using the same helper function from your previous test script
def print_analysis_details(method_name, comment_text_identifier, result_dict, original_raw_text="N/A", processed_text_for_batch="N/A"):
    should_print = False
    if TARGET_DEBUG_COMMENT_SUBSTRING_WARSI in comment_text_identifier:
        should_print = True

    if not should_print:
        return

    print(f"\n--- DETAILED ANALYSIS from {method_name} for comment containing '{comment_text_identifier}' ---")
    if original_raw_text != "N/A":
        print(f"  Original Raw Text: '{original_raw_text}'")
    if processed_text_for_batch != "N/A":
        print(f"  Text fed to explainer's _tokenize_and_pad_batch: '{processed_text_for_batch}'")

    print(f"  Prediction: {result_dict.get('prediction')}, Confidence: {result_dict.get('confidence')}")
    tokens = result_dict.get('tokens', [])
    print(f"  Tokens ({len(tokens)}): {tokens}")
    attention_weights = result_dict.get('attention', [])
    if isinstance(attention_weights, np.ndarray):
        attention_weights_list = attention_weights.tolist()
    else:
        attention_weights_list = attention_weights
    print(f"  Attention Weights ({len(attention_weights_list)}, sum: {np.sum(attention_weights_list):.4f}): {attention_weights_list}")

    gender_bias_info = result_dict.get('gender_bias', {})
    print(f"  Interpretation: {gender_bias_info.get('interpretation')}")
    print(f"  Stereotype Bias Score: {gender_bias_info.get('stereotype_bias_score')}")
    category_attention_pct = gender_bias_info.get('category_attention_pct', {})
    print(f"  Category Attention Pct: {category_attention_pct}")
    print(f"    Objective Pct: {category_attention_pct.get('objective_pedagogical', 0):.1f}%")
    print(f"    Intellect Pct: {category_attention_pct.get('intellect_achievement', 0):.1f}%")
    print(f"    Entertainment Pct: {category_attention_pct.get('entertainment_authority', 0):.1f}%")
    print("--- END DETAILED ANALYSIS ---")


def run_pipeline_simulation_test():
    print("Attempting to load actual trained model and vocab.")
    model_path = os.path.join(settings.BASE_DIR, 'machine_learning/ml_models_trained/lstm_sentiment.pt')
    vocab_path = os.path.join(settings.BASE_DIR, 'machine_learning/ml_models_trained/vocab.json')

    if not (os.path.exists(model_path) and os.path.exists(vocab_path)):
        print(f"ERROR: Model or Vocab file not found. Check paths:\nModel: {model_path}\nVocab: {vocab_path}")
        return

    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        vocab_size = len(vocab)
        model = CustomSentimentLSTM(
            vocab_size=vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.5
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Trained model and vocab loaded successfully onto {device}.")
    except Exception as e:
        print(f"ERROR loading model/vocab: {e}")
        import traceback
        print(traceback.format_exc())
        return

    explainer = GenderBiasExplainer(model, vocab)

    # --- Test Case: "Professor Warsi" comment ---
    warsi_raw = "Professor Warsi is brilliant! His lectures are not only informative but also incredibly entertaining and funny. Made English enjoyable. Truly awesome."

    # Scenario 1: Input as `explain_prediction` would process it (via internal clean_text)
    print("\n\n--- SCENARIO 1: Simulating `explain_prediction` for Warsi comment ---")
    # `explain_prediction` calls clean_text internally. We are replicating its behavior.
    # The text_for_tokenizer within explain_prediction would be clean_text(warsi_raw)
    warsi_cleaned_by_explainer_internal = get_fully_processed_text_for_explainer(warsi_raw) # This is what _tokenize_and_pad_batch gets inside explain_prediction
    result_single_warsi = explainer.explain_prediction(
        text=warsi_raw, selected_gender="Male", discipline="English"
    )
    print_analysis_details(
        "explain_prediction",
        TARGET_DEBUG_COMMENT_SUBSTRING_WARSI,
        result_single_warsi,
        original_raw_text=warsi_raw,
        processed_text_for_batch=warsi_cleaned_by_explainer_internal # Show what it was tokenized from
    )

    # Scenario 2: Input as `explain_batch` might receive it from a "problematic" processed_comment
    # HYPOTHESIS: The 'processed_comment' in your CSV for Warsi is NOT 'warsi brilliant ...'
    # but something slightly different, e.g., includes "Professor" or has different punctuation/spacing.

    # Create a few hypothetical "problematic" versions of processed_comment for Warsi
    # These are what you suspect might be in your _processed.csv for this comment
    # (and thus get passed to explain_batch in the pipeline)
    problematic_versions = {
        "With 'Professor' title (lowercase)": "professor warsi is brilliant his lectures are not only informative but also incredibly entertaining and funny made english enjoyable truly awesome",
        "With 'Professor' title (original case)": "Professor warsi is brilliant! his lectures are not only informative but also incredibly entertaining and funny. made english enjoyable. truly awesome.", # With some punctuation
        "Slightly different spacing/punctuation": "professor warsi is brilliant! his lectures are not only informative but also incredibly entertaining and funny. made english enjoyable. truly awesome.",
        "Ideal clean_text output (for control)": get_fully_processed_text_for_explainer(warsi_raw) # This should match Scenario 1
    }

    print("\n\n--- SCENARIO 2: Simulating `explain_batch` with various hypothetical 'processed_comment' inputs for Warsi ---")
    for desc, warsi_processed_hypothetical in problematic_versions.items():
        print(f"\nTesting `explain_batch` with: {desc}")
        # This `warsi_processed_hypothetical` simulates a single entry from `chunk_df['processed_comment']`
        # that `explain_batch` would receive in its `texts` list.
        # `_tokenize_and_pad_batch` inside `explain_batch` will then do .lower().split() on this.
        batch_result_warsi = explainer.explain_batch(
            texts=[warsi_processed_hypothetical], # Batch of one
            selected_genders=["Male"],
            disciplines=["English"]
        )
        if batch_result_warsi: # It returns a list
            print_analysis_details(
                f"explain_batch ({desc})",
                TARGET_DEBUG_COMMENT_SUBSTRING_WARSI, # For matching to print
                batch_result_warsi[0],
                original_raw_text=warsi_raw, # To show what it originally was
                processed_text_for_batch=warsi_processed_hypothetical # Show what was fed
            )
        else:
            print(f"  explain_batch returned empty for: {desc}")

if __name__ == "__main__":
    run_pipeline_simulation_test()