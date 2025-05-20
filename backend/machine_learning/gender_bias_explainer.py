# machine_learning/gender_bias_explainer.py

import torch
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import os
from tqdm import tqdm

# --- Define LEXICONS (Updated as per previous refinement) ---

# ** Potentially Biased / Stereotypical Descriptors **
INTELLECT_ACHIEVEMENT_DESCRIPTORS = {
    'brilliant', 'smart', 'intelligent', 'genius', 'intellectual', 'knowledgeable', 'expert',
    'outstanding', 'excellent', 'exceptional', 'remarkable', 'phenomenal', 'best', 'talented',
    'impressive', 'fascinating', 'thought-provoking', 'thought', 'deep', 'analytical', 'original',
    'innovative', 'creative', 'critical', 'sharp', 'astute', 'wisdom', 'masterful',
    'theoretical', 'philosophical', 'scientific', 'technical', 'research', 'scholar', 'scholarly',
    'academic', 'rational', 'logical', 'rigorous', 'demanding'
}
ENTERTAINMENT_AUTHORITY_DESCRIPTORS = {
    'funny', 'cool', 'entertaining', 'hilarious', 'humor', 'witty', 'laugh', 'funniest',
    'enjoyable', 'awesome', 'fun', 'interesting', 'engaging', 'enthusiastic',
    'charismatic', 'energetic', 'charming', 'dynamic', 'lively', 'animated', 'captivating',
    'confident', 'authoritative', 'powerful', 'strong', 'commanding', 'bold', 'direct',
    'straightforward', 'assertive', 'decisive', 'fair', 'objective', 'tough', 'challenging',
    'respect', 'respectable', 'no-nonsense', 'leader', 'leadership', 'mentor',
    'amazing' # ADDED
}
COMPETENCE_ORGANIZATION_DESCRIPTORS = {
    'prepared', 'thorough', 'detailed', 'precise', 'effective', 'efficient',
    'methodical', 'structured',
    'timely', 'punctual', 'consistent', 'reliable', 'dependable', 'professional',
    'diligent', 'meticulous', 'careful', 'attentive', 'focused',
    'flexible', 'adaptable', 'accessible', 'available', 'responsive',
    'correct', 'improve', 'improvement'
}
WARMTH_NURTURING_DESCRIPTORS = {
    'personable', 'caring', 'friendly', 'nice', 'sweet', 'kind', 'relatable', 'approachable',
    'easygoing', 'understanding', 'relaxed', 'cheerful', 'excited', 'passionate', 'warm',
    'delightful', 'joy', 'pleasant', 'lovely', 'wonderful', 'sweetheart', 'positive',
    'supportive', 'encouraging', 'nurturing', 'empathetic', 'compassionate', 'sympathetic',
    'thoughtful', 'considerate', 'gentle', 'patient',
    'comfort', 'comfortable', 'safe', 'welcoming', 'inclusive',
    'help', 'helpful',
    'dedicated', 'devoted', 'committed', 'accommodating', 'gracious',
    'loved' # ADDED
}
MALE_NEGATIVE_DESCRIPTORS = {
    'boring', 'bored', 'bore', 'tedious', 'dull', 'monotonous',
    'harsh', 'brutal', 'intimidating', 'arrogant', 'condescending', 'dismissive',
    'egotistical', 'pompous', 'aggressive', 'rude', 'insensitive', 'aloof', 'cold',
    'distant', 'detached', 'unapproachable',
    'terrible', 'horrible', 'awful', 'incompetent', 'lazy',
    'disorganized', 'unprepared', 'confusing', 'unclear',
    'difficult', 'unfair', 'biased', 'hard',
    'useless' # ADDED
}
FEMALE_NEGATIVE_DESCRIPTORS = {
    'unprofessional', 'emotional', 'moody', 'sensitive', 'defensive', 'dramatic',
    'scattered', 'disorganized', 'unclear', 'confusing', 'vague', 'rambling',
    'chatty', 'talkative', 'loud', 'shrill', 'annoying', 'irritating', 'frustrating',
    'difficult', 'picky', 'fussy', 'demanding', 'strict', 'harsh', 'mean', 'nasty',
    'rude', 'unfriendly', 'cold', 'uptight', 'rigid', 'inflexible', 'unreasonable',
    'unforgiving', 'stressed', 'stressful', 'anxious', 'nervous', 'worried', 'frazzled',
    'overwhelmed', 'uncaring','judgemental',
    'condescend', 'condescending', 'patronizing', 'belittling',
    'insane' # ADDED
}
OBJECTIVE_PEDAGOGICAL_DESCRIPTORS = {
    'clear', 'clarity', 'organized', 'structure', 'structured', 'well-structured', 'systematic', 'orderly',
    'material', 'content', 'topics', 'concepts', 'theories', 'readings', 'relevant', 'informative', 'comprehensive',
    'examples', 'illustrate', 'objectives', 'connections', 'progression', 'align', 'systematically', 'covered',
    'feedback', 'assignments', 'grading', 'evaluation', 'assessment', 'criteria', 'suggestions', 'tested',
    'helpful', 'communication', 'provided', 'needed', 'presented',
    'course', 'class', 'lecture', 'module', 'learning', 'teaching', 'professor', 'instructor', 'student',
    'practical', 'pragmatic', 'reasonable', 'specific', 'delayed', 'performance', 'quality', 'effectiveness'
}

# Add debug target substring at the top of the class
TARGET_DEBUG_COMMENT_SUBSTRING = "Professor Warsi is brilliant"  # Or "clearly knowledgeable"

class GenderBiasExplainer:    
    def __init__(self, model, vocab, max_len=100):
        """Initializes the explainer with the model, vocabulary, and max length."""
        self.model = model
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.device = next(model.parameters()).device
        self.max_len = max_len
        self.pad_idx = self.vocab.get('<pad>', 1) # Define pad index once
        self.discipline_gender_gaps = self._load_discipline_gender_gaps()
        self.thresholds = {
            'objective_focus': 35.0,
            'negative_focus': 10.0,
            'stereotype_focus': 15.0,
            'stereotype_bias': 0.20,
            'moderate_stereotype_bias': 0.15
        }
        
    def _normalize_attention_weights(self, attention_weights):
        """Normalize attention weights to ensure they sum to 1.0."""
        if len(attention_weights) == 0:
            return np.array([])
        
        # Normalize to sum to 1.0
        weight_sum = np.sum(attention_weights)
        if weight_sum > 0:
            return attention_weights / weight_sum
        return attention_weights

    def _load_discipline_gender_gaps(self):
        """Loads discipline gender rating gaps (requires Django setup or alternative data source)."""
        try:
            import django
            # Ensure Django is configured before importing models/utils
            if not django.apps.apps.ready:
                 # Attempt to configure Django. Adjust 'faireval.settings' if needed.
                 os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'faireval.settings')
                 django.setup()
            from api.utils import calculate_gender_discipline_heatmap # Assuming utils is in api app
            heatmap_data = calculate_gender_discipline_heatmap()
            discipline_gaps = {}
            for item in heatmap_data:
                discipline = item['discipline']
                gender = item['gender']
                rating = item['avg_rating']
                if discipline not in discipline_gaps:
                    discipline_gaps[discipline] = {'male_rating': 0, 'female_rating': 0, 'gap': 0}
                if gender == 'Male': discipline_gaps[discipline]['male_rating'] = rating
                elif gender == 'Female': discipline_gaps[discipline]['female_rating'] = rating
            for discipline, data in discipline_gaps.items():
                if data['male_rating'] and data['female_rating']:
                    data['gap'] = round(data['male_rating'] - data['female_rating'], 2)
            return discipline_gaps
        except Exception as e:
            # Catch a wider range of potential errors (ImportError, ImproperlyConfigured, etc.)
            print(f"Warning: Cannot load discipline gender gaps. Django environment might not be available or configured correctly. Error: {e}")
            return {} # Return empty dict on failure

    def _tokenize_and_pad_batch(self, texts: list):
        """Tokenizes and pads a batch of texts to the specified max_len."""
        batch_indices = []
        original_tokens_batch = []
        for text in texts:
            # Ensure text is a string and handle potential None/NaN
            text_str = str(text) if text and isinstance(text, (str, bytes)) else ""
            tokens = text_str.lower().split()
            original_tokens_batch.append(tokens[:self.max_len])
            indices = [self.vocab.get(t, 0) for t in tokens] # 0 for <unk>
            # Truncate or pad
            current_len = len(indices)
            if current_len > self.max_len:
                indices = indices[:self.max_len]
            else:
                padding = [self.pad_idx] * (self.max_len - current_len)
                indices += padding
            batch_indices.append(indices)
        # Move tensor creation outside the loop for efficiency
        input_tensor = torch.tensor(batch_indices, dtype=torch.long).to(self.device)
        return input_tensor, original_tokens_batch

    def explain_batch(self, texts: list, selected_genders: list, disciplines: list = None):
        """Processes a batch of comments for bias explanation using batched inference."""
        if not texts: return []
        batch_size = len(texts)
        # Ensure disciplines list matches batch size if provided
        if disciplines is None: disciplines = [None] * batch_size
        if len(selected_genders) != batch_size or len(disciplines) != batch_size:
             raise ValueError("Length mismatch: texts, selected_genders, and disciplines must have the same length.")

        input_tensor, original_tokens_batch = self._tokenize_and_pad_batch(texts)

        # Batched inference
        with torch.no_grad():
            try:
                # model output: batch_pred shape [batch_size, 1] (for sentiment), batch_attention shape [batch_size, seq_len]
                batch_pred_tensor, batch_attention_tensor = self.model(input_tensor, return_attention=True)
            except Exception as model_err:
                print(f"ERROR during batch model inference: {model_err}")
                return [{'prediction': 'Error', 'gender_bias': {'interpretation': [f'Model Inference Error: {model_err}']}} for _ in texts]

        # Process results individually after batch inference
        # Squeeze might reduce (1,1) to (1,) or even to 0-dim if batch_size is also 1
        # It's safer to handle based on batch_size
        
        batch_pred_cpu = batch_pred_tensor.cpu().numpy() # Shape will be (batch_size, 1) or (batch_size,)
        batch_attention_cpu = batch_attention_tensor.cpu().numpy() # Shape will be (batch_size, seq_len)
        
        results = []

        for i in range(batch_size):
            original_tokens = original_tokens_batch[i]
            num_original_tokens = len(original_tokens)
            
            # Safely slice attention weights
            # batch_attention_cpu is expected to be [batch_size, seq_len_padded]
            raw_attn_weights = batch_attention_cpu[i, :num_original_tokens] if num_original_tokens > 0 else np.array([])
            attn_weights = self._normalize_attention_weights(raw_attn_weights)

            # Handle pred_prob extraction carefully
            # batch_pred_cpu is expected to be [batch_size, 1] or [batch_size,]
            # Accessing batch_pred_cpu[i] should give a single value or a single-element array
            current_pred_val = batch_pred_cpu[i]
            if isinstance(current_pred_val, np.ndarray) and current_pred_val.ndim > 0:
                # If it's like array([0.9]), get the first element
                pred_prob = float(current_pred_val[0]) 
            else:
                # If it's already a scalar (e.g. from a squeeze if batch_size was 1, or if model output is [batch_size])
                pred_prob = float(current_pred_val)

            prediction = "Positive" if pred_prob >= 0.5 else "Negative"
            confidence_score = pred_prob if pred_prob >= 0.5 else 1.0 - pred_prob

            # ... (rest of the loop for gender_bias_data, explanation, etc.)
            current_gender = selected_genders[i]
            if current_gender not in ['Male', 'Female']:
                 print(f"Warning: Invalid gender ('{current_gender}') for comment index {i}. Skipping detailed analysis.")
                 gender_bias_data = {
                     'interpretation': ['Error: Invalid gender provided'],
                     'stereotype_bias_score': 0.0, 'category_attention_pct': {}
                 }
            else:
                gender_bias_data = self._analyze_gender_patterns(
                    texts[i], original_tokens, attn_weights, prediction, current_gender, confidence_score
                )

            explanation = {
                'prediction': prediction,
                'confidence': confidence_score,
                'tokens': original_tokens,
                'attention': attn_weights.tolist(),
                'gender_bias': gender_bias_data,
            }

            discipline = disciplines[i]
            if discipline and discipline in self.discipline_gender_gaps:
                 gap_data = self.discipline_gender_gaps[discipline]
                 explanation['discipline_context'] = {
                     'discipline': discipline,
                     'gender_rating_gap': gap_data.get('gap', 0),
                     'male_avg_rating': gap_data.get('male_rating', 0),
                     'female_avg_rating': gap_data.get('female_rating', 0),
                     'correlation': self._calculate_attention_gap_correlation(
                         gender_bias_data.get('stereotype_bias_score', 0.0),
                         gap_data.get('gap', 0)
                     )
                 }
            results.append(explanation)

        return results

    # --- Single prediction (for interactive use, kept separate from batch) ---
    def explain_prediction(self, text, selected_gender, discipline=None, return_attention_only=False):
        """Explain a single model prediction with attention and bias analysis."""
        if selected_gender not in ['Male', 'Female']:
             raise ValueError("Invalid selected_gender provided. Must be 'Male' or 'Female'.")
        from data_processing.text_preprocessing import get_fully_processed_text_for_explainer

        # Apply the same preprocessing as in pipeline.py before tokenization
        processed_text_for_model = get_fully_processed_text_for_explainer(text)

        if not processed_text_for_model.strip():
            print(f"Warning: Full preprocessing made the input '{str(text)[:50]}...' empty. Analysis might be trivial.")
            pass

        # Prepare single input
        input_tensor, original_tokens_batch = self._tokenize_and_pad_batch([processed_text_for_model])
        original_tokens = original_tokens_batch[0]
        num_original_tokens = len(original_tokens)        # Single inference
        with torch.no_grad():
            pred, attention = self.model(input_tensor, return_attention=True)

        # Extract and normalize attention weights
        raw_attn_weights = attention.squeeze().cpu().numpy()[:num_original_tokens] if num_original_tokens > 0 else np.array([])
        # Renormalize the weights after truncation to ensure they sum to 1.0
        attn_weights = self._normalize_attention_weights(raw_attn_weights)
        
        # Debug information for target comment
        if TARGET_DEBUG_COMMENT_SUBSTRING in processed_text_for_model:
            print(f"\n--- DEBUG: explain_prediction for '{TARGET_DEBUG_COMMENT_SUBSTRING}' ---")
            print(f"Raw text input: '{text}'")
            print(f"Input text to _tokenize_and_pad_batch: '{processed_text_for_model}'")
            print(f"Tokens sent to _analyze_gender_patterns: {original_tokens}")
            print(f"Attention weights (sum: {np.sum(attn_weights):.4f}): {attn_weights.tolist()}")
        
        pred_prob = float(pred.item())
        prediction = "Positive" if pred_prob >= 0.5 else "Negative"
        confidence_score = pred_prob if pred_prob >= 0.5 else 1.0 - pred_prob

        if return_attention_only:
            return attn_weights.tolist() # Return as list if only attention needed

        # Analyze patterns
        gender_bias_data = self._analyze_gender_patterns(
            processed_text_for_model, original_tokens, attn_weights, prediction, selected_gender, confidence_score
        )

        explanation = {
            'prediction': prediction,
            'confidence': confidence_score,
            'tokens': original_tokens,
            'attention': attn_weights.tolist(),
            'gender_bias': gender_bias_data,
        }

        # Add discipline context
        if discipline and discipline in self.discipline_gender_gaps:
             gap_data = self.discipline_gender_gaps[discipline]
             explanation['discipline_context'] = {
                 'discipline': discipline,
                 'gender_rating_gap': gap_data.get('gap', 0),
                 'male_avg_rating': gap_data.get('male_rating', 0),
                 'female_avg_rating': gap_data.get('female_rating', 0),
                 'correlation': self._calculate_attention_gap_correlation(
                     gender_bias_data.get('stereotype_bias_score', 0.0),
                     gap_data.get('gap', 0)
                 )
             }
        return explanation

    def _analyze_gender_patterns(self, text, tokens, attention_weights, prediction, selected_gender, confidence):
        """Analyze gendered patterns based on descriptors and selected gender."""
        # Ensure attention_weights has same length as tokens
        if len(attention_weights) != len(tokens):
            min_len = min(len(attention_weights), len(tokens))
            tokens = tokens[:min_len]
            attention_weights = attention_weights[:min_len]
            print(f"Warning: Mismatch in token/attention length for comment: '{str(text)[:50]}...' - Truncating.")
            if min_len == 0:
                 return { # Return structure expected by caller
                     'descriptor_categories': {}, 'category_attention': {}, 'category_attention_pct': {},
                     'stereotype_bias_score': 0.0, 'top_terms_by_category': {},
                     'interpretation': ["Error: Token/Attention Length Mismatch (0)"]
                 }

        descriptor_categories = defaultdict(list)
        # Associate tokens with categories (Objective First)
        for i, token in enumerate(tokens):
            token_lower = token.lower()
            assigned = False
            if token_lower in OBJECTIVE_PEDAGOGICAL_DESCRIPTORS: descriptor_categories['objective_pedagogical'].append((i, token, attention_weights[i])); assigned = True
            elif token_lower in INTELLECT_ACHIEVEMENT_DESCRIPTORS: descriptor_categories['intellect_achievement'].append((i, token, attention_weights[i])); assigned = True
            elif token_lower in ENTERTAINMENT_AUTHORITY_DESCRIPTORS: descriptor_categories['entertainment_authority'].append((i, token, attention_weights[i])); assigned = True
            elif token_lower in COMPETENCE_ORGANIZATION_DESCRIPTORS: descriptor_categories['competence_organization'].append((i, token, attention_weights[i])); assigned = True
            elif token_lower in WARMTH_NURTURING_DESCRIPTORS: descriptor_categories['warmth_nurturing'].append((i, token, attention_weights[i])); assigned = True
            elif token_lower in MALE_NEGATIVE_DESCRIPTORS: descriptor_categories['male_negative'].append((i, token, attention_weights[i])); assigned = True
            elif token_lower in FEMALE_NEGATIVE_DESCRIPTORS: descriptor_categories['female_negative'].append((i, token, attention_weights[i])); assigned = True
            if not assigned: descriptor_categories['other'].append((i, token, attention_weights[i]))

        category_attention = {cat: sum(w for _, _, w in items) for cat, items in descriptor_categories.items()}
        total_meaningful_attention = sum(v for k, v in category_attention.items() if k != 'other')
        if total_meaningful_attention <= 1e-6: total_meaningful_attention = 1.0
        category_attention_pct = {
            k: (v / total_meaningful_attention) * 100
            for k, v in category_attention.items() if k != 'other'
        }
        total_all_attention = sum(category_attention.values()) or 1.0
        category_attention_pct['other'] = (category_attention.get('other', 0) / total_all_attention) * 100

        male_stereotype_weight = category_attention.get('intellect_achievement', 0) + category_attention.get('entertainment_authority', 0)
        female_stereotype_weight = category_attention.get('competence_organization', 0) + category_attention.get('warmth_nurturing', 0)
        stereotype_bias_score = 0.0
        denominator = male_stereotype_weight + female_stereotype_weight
        if denominator > 1e-6:
            stereotype_bias_score = (male_stereotype_weight - female_stereotype_weight) / denominator

        top_terms_by_category = {
            cat: sorted(items, key=lambda x: x[2], reverse=True)[:5]
            for cat, items in descriptor_categories.items() if items
        }

        # Add this debug block after calculating category_attention_pct and stereotype_bias_score
        if TARGET_DEBUG_COMMENT_SUBSTRING in text:
            print(f"--- DEBUG: _analyze_gender_patterns for '{TARGET_DEBUG_COMMENT_SUBSTRING}' ---")
            print(f"  Text received: '{text}'")
            print(f"  Tokens received: {tokens}")  # Should match the ones printed above
            print(f"  Category Attention Pct: {category_attention_pct}")
            print(f"  Stereotype Bias Score: {stereotype_bias_score}")
            # You can also print individual category percentages
            obj_pct = category_attention_pct.get('objective_pedagogical', 0)
            print(f"  Calculated obj_pct: {obj_pct:.1f}%")
            entertain_pct = category_attention_pct.get('entertainment_authority', 0)
            print(f"  Calculated entertain_pct: {entertain_pct:.1f}%")
            intellect_pct = category_attention_pct.get('intellect_achievement', 0)
            print(f"  Calculated intellect_pct: {intellect_pct:.1f}%")
        
        interpretation = self._interpret_gender_bias(
            stereotype_bias_score, category_attention_pct, prediction, selected_gender, confidence
        )

        return {
            'descriptor_categories': dict(descriptor_categories), # Convert back if needed
            'category_attention': category_attention,
            'category_attention_pct': category_attention_pct,
            'stereotype_bias_score': stereotype_bias_score,
            'top_terms_by_category': top_terms_by_category,
            'interpretation': interpretation
        }

    def _interpret_gender_bias(self, stereotype_bias, attention_pct, sentiment, selected_gender, confidence):
        """Generate interpretation prioritizing objective focus."""
        interpretation_list = []
        obj_pct = attention_pct.get('objective_pedagogical', 0)
        f_neg_pct = attention_pct.get('female_negative', 0)
        m_neg_pct = attention_pct.get('male_negative', 0)
        warmth_pct = attention_pct.get('warmth_nurturing', 0)
        entertain_pct = attention_pct.get('entertainment_authority', 0)
        intellect_pct = attention_pct.get('intellect_achievement', 0)
        comp_org_pct = attention_pct.get('competence_organization', 0)
        primary_interpretation = None

        # Step 1: Objective Focus
        if obj_pct > self.thresholds['objective_focus']:
            high_neg = (selected_gender == 'Female' and f_neg_pct > self.thresholds['negative_focus']) or \
                       (selected_gender == 'Male' and m_neg_pct > self.thresholds['negative_focus'])
            if not high_neg:
                base_interp = f"({obj_pct:.1f}%): Focus on objective aspects (e.g., structure, clarity) without strong gendered patterns."
                primary_interpretation = f"{sentiment} Sentiment - Objective Focus {base_interp}"
                if abs(stereotype_bias) > self.thresholds['moderate_stereotype_bias']:
                    bias_lean = "male-associated" if stereotype_bias > 0 else "female-associated"
                    primary_interpretation += f" (Note: Underlying language leans slightly towards {bias_lean} terms.)"

        # Step 2: Strong Negative Gendered Criticism
        if not primary_interpretation and sentiment == "Negative":
            if selected_gender == 'Female' and f_neg_pct > self.thresholds['negative_focus']:
                 primary_interpretation = f"Strong Negative Bias towards Female: Focus ({f_neg_pct:.1f}%) on negative terms typically associated with criticism of women."
            elif selected_gender == 'Male' and m_neg_pct > self.thresholds['negative_focus']:
                 primary_interpretation = f"Strong Negative Bias towards Male: Focus ({m_neg_pct:.1f}%) on negative terms typically associated with criticism of men."

        # Step 3: Positive Stereotypical Praise
        if not primary_interpretation and sentiment == "Positive":
            if selected_gender == 'Female':
                if warmth_pct > self.thresholds['stereotype_focus'] and stereotype_bias < -self.thresholds['stereotype_bias']:
                    primary_interpretation = f"Moderate Positive Bias towards Female (Stereotypical Praise): Focus ({warmth_pct:.1f}%) on warmth/nurturing, potentially reinforcing stereotypes."
            elif selected_gender == 'Male':
                if entertain_pct > self.thresholds['stereotype_focus'] and stereotype_bias > self.thresholds['stereotype_bias']:
                    primary_interpretation = f"Moderate Positive Bias towards Male (Stereotypical Praise): Focus ({entertain_pct:.1f}%) on entertainment/authority, potentially reinforcing stereotypes."

        # Step 4: General Stereotype Balance
        if not primary_interpretation:
            abs_stereotype_bias = abs(stereotype_bias)
            if abs_stereotype_bias > self.thresholds['stereotype_bias']:
                if stereotype_bias > 0:
                    bias_type = "male-associated (intellect, entertainment)"
                    other_type = "female-associated (competence, warmth)"
                    focus_detail = f"(Intellect: {intellect_pct:.1f}%, Entertain: {entertain_pct:.1f}%)"
                    pattern_context = "Aligns with common patterns for men." if selected_gender == 'Male' else "Less typical pattern for female evaluations."
                    primary_interpretation = f"Focus on Male-Associated Stereotypes: Language emphasizes {bias_type} {focus_detail} over {other_type}. {pattern_context}"
                else:
                    bias_type = "female-associated (competence, warmth)"
                    other_type = "male-associated (intellect, entertainment)"
                    focus_detail = f"(Competence: {comp_org_pct:.1f}%, Warmth: {warmth_pct:.1f}%)"
                    pattern_context = "Aligns with common patterns for women." if selected_gender == 'Female' else "Less typical pattern for male evaluations."
                    primary_interpretation = f"Focus on Female-Associated Stereotypes: Language emphasizes {bias_type} {focus_detail} over {other_type}. {pattern_context}"

        # Step 5: Final Fallback - Neutral
        if not primary_interpretation:
            base_interp = "No strong alignment with common gendered evaluation patterns or heavy focus on objective terms."
            primary_interpretation = f"{sentiment} Sentiment - Neutral Language Pattern: {base_interp}"

        interpretation_list.append(primary_interpretation)
        return interpretation_list # Return list (without confidence note)


    def _determine_bias_tag(self, interpretation_list, stereotype_bias_score, objective_percentage):
        """Determines a concise bias tag based on the primary interpretation."""
        # (Keep the implementation from previous step - it uses the refined interpretation)
        if not interpretation_list: return 'UNKNOWN'
        main_interpretation = interpretation_list[-1].lower()
        if "objective focus" in main_interpretation:
            if "leans slightly towards male-associated" in main_interpretation: return 'OBJECTIVE_M_LEAN'
            if "leans slightly towards female-associated" in main_interpretation: return 'OBJECTIVE_F_LEAN'
            return 'OBJECTIVE'
        elif "strong negative bias towards female" in main_interpretation: return 'NEG_BIAS_F'
        elif "strong negative bias towards male" in main_interpretation: return 'NEG_BIAS_M'
        elif "positive bias towards female (stereotypical praise)" in main_interpretation: return 'POS_BIAS_F'
        elif "positive bias towards male (stereotypical praise)" in main_interpretation: return 'POS_BIAS_M'
        elif "focus on male-associated stereotypes" in main_interpretation:
             return 'POS_BIAS_M' if stereotype_bias_score > self.thresholds['stereotype_bias'] else 'NEUTRAL'
        elif "focus on female-associated stereotypes" in main_interpretation:
             return 'POS_BIAS_F' if stereotype_bias_score < -self.thresholds['stereotype_bias'] else 'NEUTRAL'
        elif "neutral language pattern" in main_interpretation: return 'NEUTRAL'
        else: return 'UNKNOWN'

    def _calculate_attention_gap_correlation(self, stereotype_bias_score, rating_gap):
        """Calculate if stereotype attention bias aligns with rating gap"""
        # (Keep implementation as is)
        if pd.isna(stereotype_bias_score) or pd.isna(rating_gap): return {'alignment': 'unknown', 'explanation': 'Cannot determine alignment due to missing data.'}
        if (stereotype_bias_score > 0 and rating_gap > 0) or \
           (stereotype_bias_score < 0 and rating_gap < 0) or \
           (abs(stereotype_bias_score) < 0.1 and abs(rating_gap) < 0.1):
            alignment = "aligned"
        else:
            alignment = "contrary"
        return {
            'alignment': alignment,
            'explanation': self._generate_alignment_explanation(stereotype_bias_score, rating_gap, alignment)
        }

    def _generate_alignment_explanation(self, stereotype_bias_score, rating_gap, alignment):
        """Generate explanation of alignment between stereotype attention and ratings"""
        # (Keep implementation as is)
        if pd.isna(stereotype_bias_score) or pd.isna(rating_gap): return 'Alignment explanation unavailable.'
        abs_attention = abs(stereotype_bias_score)
        attention_focus = "strong" if abs_attention > 0.4 else "moderate" if abs_attention > 0.15 else "weak"
        attention_dir = "male-associated stereotypes" if stereotype_bias_score > 0 else "female-associated stereotypes"
        abs_gap = abs(rating_gap)
        gap_strength = "significant" if abs_gap > 0.4 else "moderate" if abs_gap > 0.15 else "minor"
        gender_ratings = "male" if rating_gap >= 0 else "female"
        if alignment == "aligned": return (f"The {attention_focus} focus on {attention_dir} aligns with the {gap_strength} rating gap favoring {gender_ratings} professors in this discipline.")
        else: return (f"The {attention_focus} focus on {attention_dir} is contrary to the {gap_strength} rating gap favoring {gender_ratings} professors in this discipline.")

    def visualize_attention(self, text, selected_gender, discipline=None, save_path=None):
        """Generate attention visualization with descriptor highlighting"""
        # (Keep implementation as is - it uses the output of explain_prediction)
        pass # Replace with actual plotting code if needed

    def analyze_comments_batch(self, comments, selected_genders, disciplines=None):
        """Analyze a batch of comments for aggregation (uses single explanation calls)."""
        # IMPORTANT: This method iterates and calls explain_prediction individually.
        # It does NOT leverage the batched GPU inference of explain_batch.
        # It's suitable for API endpoints handling moderate numbers of comments for display,
        # but the main pipeline should use explain_batch internally if possible.
        print("Warning: analyze_comments_batch uses single-comment explanation logic sequentially.")
        results = []
        if disciplines is None: disciplines = [None] * len(comments)
        if len(selected_genders) != len(comments): # Add check
            raise ValueError("Length of selected_genders must match length of comments.")

        for i, comment in enumerate(tqdm(comments, desc="Analyzing Batch (Individual Explanations)")):
            discipline = disciplines[i] if disciplines and i < len(disciplines) else None
            current_gender = selected_genders[i]
            if current_gender not in ['Male', 'Female']:
                 print(f"Skipping comment {i} due to invalid gender: {current_gender}")
                 results.append({'prediction': 'Skipped', 'gender_bias': {'interpretation': ['Invalid Gender']}})
                 continue
            try:
                explanation = self.explain_prediction(comment, current_gender, discipline)
                results.append(explanation)
            except Exception as e:
                print(f"Error explaining comment {i}: {e}")
                results.append({'prediction': 'Error', 'gender_bias': {'interpretation': [f'Error: {e}']}})

        # --- Aggregation logic (Refined) ---
        valid_results = [r for r in results if r['prediction'] not in ['Error', 'Skipped']]
        if not valid_results:
            return {
                'selected_gender_context': selected_genders[0] if selected_genders else 'Unknown', # Provide context gender if possible
                'error': 'No comments could be successfully analyzed in this batch.',
                'comment_count': len(results), 'valid_comment_count': 0,
                # Include empty structures expected by caller
                'descriptor_bias_score': 0.0, 'positive_descriptor_bias': 0.0, 'negative_descriptor_bias': 0.0,
                'positive_count': 0, 'negative_count': 0, 'category_stats': {},
                'interpretation_summary': {}, 'insights': []
            }

        # Calculate aggregate scores from valid results
        stereotype_bias_scores = [r['gender_bias']['stereotype_bias_score'] for r in valid_results if pd.notna(r['gender_bias']['stereotype_bias_score'])]
        descriptor_bias_score = np.mean(stereotype_bias_scores) if stereotype_bias_scores else 0.0

        pos_results = [r for r in valid_results if r['prediction'] == 'Positive']
        neg_results = [r for r in valid_results if r['prediction'] == 'Negative']

        pos_stereotype_scores = [r['gender_bias']['stereotype_bias_score'] for r in pos_results if pd.notna(r['gender_bias']['stereotype_bias_score'])]
        neg_stereotype_scores = [r['gender_bias']['stereotype_bias_score'] for r in neg_results if pd.notna(r['gender_bias']['stereotype_bias_score'])]

        pos_descriptor_bias = np.mean(pos_stereotype_scores) if pos_stereotype_scores else 0.0
        neg_descriptor_bias = np.mean(neg_stereotype_scores) if neg_stereotype_scores else 0.0

        # Collect interpretations and category stats (same logic as before)
        all_interpretations = [interp for r in valid_results for interp in r['gender_bias'].get('interpretation', [])]
        interpretation_counts = Counter(all_interpretations)

        category_terms = defaultdict(list)
        for r in valid_results:
             for category, items in r['gender_bias'].get('descriptor_categories', {}).items():
                 category_terms[category].extend([(term, weight) for _, term, weight in items])

        category_stats = {}
        for category, terms in category_terms.items():
            if not terms: continue
            term_counter = Counter([term.lower() for term, _ in terms])
            term_avg_attn = defaultdict(list)
            for term, weight in terms: term_avg_attn[term.lower()].append(weight)
            term_avg = {term: np.mean(weights) for term, weights in term_avg_attn.items()} # Use numpy mean
            term_relevance = {term: count * term_avg.get(term, 0) for term, count in term_counter.items()}
            top_terms = sorted(term_relevance.items(), key=lambda x: x[1], reverse=True)[:10]
            category_stats[category] = {
                'top_terms': top_terms, 'total_occurrences': len(terms), 'unique_terms': len(term_counter)
            }

        insights = self._generate_batch_insights(
            selected_genders[0] if selected_genders else 'Unknown', # Use first gender for context
            descriptor_bias_score, pos_descriptor_bias, neg_descriptor_bias,
            category_stats, interpretation_counts, len(pos_results), len(neg_results)
        )

        return {
            'selected_gender_context': selected_genders[0] if selected_genders else 'Unknown',
            'descriptor_bias_score': descriptor_bias_score, # Keep name consistent
            'positive_descriptor_bias': pos_descriptor_bias,
            'negative_descriptor_bias': neg_descriptor_bias,
            'comment_count': len(results),
            'valid_comment_count': len(valid_results),
            'positive_count': len(pos_results),
            'negative_count': len(neg_results),
            'category_stats': category_stats,
            'interpretation_summary': dict(interpretation_counts.most_common(5)),
            'insights': insights
        }

    def _generate_batch_insights(self, selected_gender, descriptor_bias, pos_descriptor_bias,
                               neg_descriptor_bias, category_stats, interpretation_counts,
                               pos_count, neg_count):
        """Generate insights about bias patterns based on new interpretation."""
        # (Keep implementation as is - uses refined scores/interpretations)
        insights = []
        total_comments = pos_count + neg_count
        if total_comments == 0: return ["No valid comments to analyze for insights."]

        # 1. Dominant Bias Interpretation
        if interpretation_counts:
            most_common_interp, count = interpretation_counts.most_common(1)[0]
            if count / total_comments > 0.3:
                 insights.append(f"Predominant Finding: A significant portion ({count}/{total_comments}) of comments were assessed as: '{most_common_interp}'.")

        # 2. Overall Descriptor Focus Pattern
        if abs(descriptor_bias) > self.thresholds['moderate_stereotype_bias']: # Use moderate threshold here
            focus = "male-associated (intellect/entertainment)" if descriptor_bias > 0 else "female-associated (competence/warmth)"
            insights.append(
                f"Overall Descriptor Focus: Comments tend to emphasize {focus} descriptors, which may reflect biased expectations for a {selected_gender} individual."
            )

        # 3. Descriptor Focus differences between Positive and Negative comments
        if abs(pos_descriptor_bias - neg_descriptor_bias) > self.thresholds['stereotype_bias']: # Use bias threshold
            if pos_descriptor_bias > neg_descriptor_bias:
                 insights.append("Praise vs. Criticism Focus: Positive comments emphasize intellect/entertainment more, while negative comments focus more on competence/warmth.")
            else:
                 insights.append("Praise vs. Criticism Focus: Negative comments emphasize intellect/entertainment more, while positive comments focus more on competence/warmth.")

        # 4. Highlighting specific category dominance
        male_assoc_occur = category_stats.get('intellect_achievement', {}).get('total_occurrences', 0) + \
                           category_stats.get('entertainment_authority', {}).get('total_occurrences', 0)
        female_assoc_occur = category_stats.get('competence_organization', {}).get('total_occurrences', 0) + \
                             category_stats.get('warmth_nurturing', {}).get('total_occurrences', 0)
        total_stereotype_occur = male_assoc_occur + female_assoc_occur

        if total_stereotype_occur > 0:
             if male_assoc_occur / total_stereotype_occur > 0.65:
                 insights.append("Strong Focus Skew: Comments heavily emphasize male-associated descriptors (intellect, entertainment) over female-associated ones.")
             elif female_assoc_occur / total_stereotype_occur > 0.65:
                 insights.append("Strong Focus Skew: Comments heavily emphasize female-associated descriptors (competence, warmth) over male-associated ones.")

        # 5. Mention presence of gendered negative terms if significant
        neg_male_terms_count = category_stats.get('male_negative', {}).get('total_occurrences', 0)
        neg_female_terms_count = category_stats.get('female_negative', {}).get('total_occurrences', 0)
        if neg_male_terms_count / total_comments > 0.1:
             insights.append("Note: Male-associated negative terms appear frequently, suggesting potential patterns in criticism.")
        if neg_female_terms_count / total_comments > 0.1:
             insights.append("Note: Female-associated negative terms appear frequently, suggesting potential patterns in criticism.")

        if not insights:
            insights.append("No strong overarching bias patterns detected across the comments in this batch for the selected gender context.")

        return insights