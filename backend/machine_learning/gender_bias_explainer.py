# gender_bias_explainer.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import os

# Define lexicons for different types of descriptors commonly used in evaluations
# Research shows these patterns differ by gender in evaluations

# --- Define LEXICONS ---

# ** Potentially Biased / Stereotypical Descriptors **

# Male-Associated (Positive)
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
    'enjoyable', 'awesome', 'amazing', 'fun', 'interesting', 'engaging', 'enthusiastic',
    'charismatic', 'energetic', 'charming', 'dynamic', 'lively', 'animated', 'captivating',
    'confident', 'authoritative', 'powerful', 'strong', 'commanding', 'bold', 'direct',
    'straightforward', 'assertive', 'decisive', 'fair', 'objective', 'tough', 'challenging', # Objective/fair could be debated, keep here for now
    'respect', 'respectable', 'no-nonsense', 'leader', 'leadership', 'mentor'
}

# Female-Associated (Positive - often Stereotypical or specific focus areas)
COMPETENCE_ORGANIZATION_DESCRIPTORS = {
    # Keep terms leaning towards effort/process rather than pure objectivity
    'prepared', 'thorough', 'detailed', 'precise', 'effective', 'efficient',
    'methodical', 'structured', #'orderly', 'systematic', # Moved to objective
    'timely', 'punctual', 'consistent', 'reliable', 'dependable', 'professional',
    'diligent', 'meticulous', 'careful', 'attentive', 'focused',
    'flexible', 'adaptable', 'accessible', 'available', 'responsive',
    'correct', 'improve', 'improvement' # Keep improvement-related here
    # 'organized', 'clear', 'concise', 'informative', 'helpful', 'comprehensive', 'practical', 'pragmatic', 'reasonable', 'feedback', 'clarity', 'communication' -> Moved to OBJECTIVE
}
WARMTH_NURTURING_DESCRIPTORS = {
    'personable', 'caring', 'friendly', 'nice', 'sweet', 'kind', 'relatable', 'approachable',
    'easygoing', 'understanding', 'relaxed', 'cheerful', 'excited', 'passionate', 'warm',
    'delightful', 'joy', 'pleasant', 'lovely', 'wonderful', 'sweetheart', 'positive',
    'supportive', 'encouraging', 'nurturing', 'empathetic', 'compassionate', 'sympathetic',
    'thoughtful', 'considerate', 'gentle', 'patient',
    'comfort', 'comfortable', 'safe', 'welcoming', 'inclusive',
    'help', 'helpful', # Duplicated in objective, but strong warmth signal too
    'dedicated', 'devoted', 'committed', 'accommodating', 'gracious'
}

# Male-Associated (Negative)
MALE_NEGATIVE_DESCRIPTORS = {
    'boring', 'bored', 'bore', 'tedious', 'dull', 'monotonous',
    'harsh', 'brutal', 'intimidating', 'arrogant', 'condescending', 'dismissive',
    'egotistical', 'pompous', 'aggressive', 'rude', 'insensitive', 'aloof', 'cold',
    'distant', 'detached', 'unapproachable',
    'terrible', 'horrible', 'awful', 'incompetent', 'lazy',
    'disorganized', 'unprepared', 'confusing', 'unclear',
    'difficult', 'unfair', 'biased', 'hard'
}

# Female-Associated (Negative) 
FEMALE_NEGATIVE_DESCRIPTORS = {
    'unprofessional', 'emotional', 'moody', 'sensitive', 'defensive', 'dramatic',
    'scattered', 'disorganized', 'unclear', 'confusing', 'vague', 'rambling',
    'chatty', 'talkative', 'loud', 'shrill', 'annoying', 'irritating', 'frustrating',
    'difficult', 'picky', 'fussy', 'demanding', 'strict', 'harsh', 'mean', 'nasty',
    'rude', 'unfriendly', 'cold', 'uptight', 'rigid', 'inflexible', 'unreasonable',
    'unforgiving', 'stressed', 'stressful', 'anxious', 'nervous', 'worried', 'frazzled',
    'overwhelmed', 'uncaring','judgemental',
    'condescend', 'condescending', 'patronizing', 'belittling'
}


# --- NEW CATEGORY ---
OBJECTIVE_PEDAGOGICAL_DESCRIPTORS = {
    # Structure & Clarity
    'clear', 'clarity', 'organized', 'structure', 'structured', 'well-structured', 'systematic', 'orderly',
    # Content & Relevance
    'material', 'content', 'topics', 'concepts', 'theories', 'readings', 'relevant', 'informative', 'comprehensive',
    'examples', 'illustrate', 'objectives', 'connections', 'progression', 'align', 'systematically', 'covered',
    # Feedback & Assessment
    'feedback', 'assignments', 'grading', 'evaluation', 'assessment', 'criteria', 'suggestions', 'tested',
    # Interaction & Helpfulness (Objective aspects)
    'helpful', 'communication', 'provided', 'needed', 'presented',
    # General Course Terms (often neutral)
    'course', 'class', 'lecture', 'module', 'learning', 'teaching', 'professor', 'instructor', 'student',
    'practical', 'pragmatic', 'reasonable', 'specific', 'delayed', 'performance', 'quality', 'effectiveness'
}

# --- REMOVED EXPLICIT_GENDERED_TERMS ---

class GenderBiasExplainer:
    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.device = next(model.parameters()).device
        self.discipline_gender_gaps = self._load_discipline_gender_gaps()

    def _load_discipline_gender_gaps(self):
        # (Keep this function as is)
        try:
            import django
            os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'faireval.settings')
            django.setup()
            from api.utils import calculate_gender_discipline_heatmap
            heatmap_data = calculate_gender_discipline_heatmap()
            discipline_gaps = {}
            for item in heatmap_data:
                discipline = item['discipline']
                gender = item['gender']
                rating = item['avg_rating']
                if discipline not in discipline_gaps:
                    discipline_gaps[discipline] = {'male_rating': 0, 'female_rating': 0, 'gap': 0}
                if gender == 'Male':
                    discipline_gaps[discipline]['male_rating'] = rating
                elif gender == 'Female':
                    discipline_gaps[discipline]['female_rating'] = rating
            for discipline, data in discipline_gaps.items():
                if data['male_rating'] and data['female_rating']:
                    data['gap'] = round(data['male_rating'] - data['female_rating'], 2)
            return discipline_gaps
        except (ImportError, ModuleNotFoundError):
            print("Warning: Cannot load discipline gender gaps - Django environment not available")
            return {}
        except Exception as e:
            print(f"Error loading discipline gender gaps: {str(e)}")
            return {}

    # --- UPDATED SIGNATURE TO ACCEPT selected_gender ---
    def explain_prediction(self, text, selected_gender, discipline=None, return_attention_only=False):
        """
        Explain model prediction with attention weights and gender bias analysis based on selected gender.
        """
        # Tokenization and input prep (no change)
        tokens = text.lower().split()
        indices = [self.vocab.get(t.lower(), 0) for t in tokens]
        max_len = 100
        if len(indices) > max_len:
            indices = indices[:max_len]
            tokens = tokens[:max_len]
        else:
            padding = [1] * (max_len - len(indices))
            indices += padding
        input_tensor = torch.tensor([indices], dtype=torch.long).to(self.device)

        # Get model prediction and attention (no change)
        with torch.no_grad():
            pred, attention = self.model(input_tensor, return_attention=True)
        attn_weights = attention.squeeze().cpu().numpy()[:len(tokens)]
        prediction = "Positive" if pred.item() >= 0.5 else "Negative"
        confidence = pred.item() if pred.item() >= 0.5 else 1 - pred.item()

        if return_attention_only:
            return attn_weights

        # --- PASS selected_gender AND prediction TO _analyze_gender_patterns ---
        gender_bias_data = self._analyze_gender_patterns(text, tokens, attn_weights, prediction, selected_gender)

        explanation = {
            'prediction': prediction,
            'confidence': confidence,
            'tokens': tokens,
            'attention': attn_weights.tolist(),
            'gender_bias': gender_bias_data,
        }

        # Add discipline context (no change, but uses updated bias score)
        if discipline and discipline in self.discipline_gender_gaps:
            explanation['discipline_context'] = {
                'discipline': discipline,
                'gender_rating_gap': self.discipline_gender_gaps[discipline]['gap'],
                'male_avg_rating': self.discipline_gender_gaps[discipline]['male_rating'],
                'female_avg_rating': self.discipline_gender_gaps[discipline]['female_rating'],
                'correlation': self._calculate_attention_gap_correlation(
                    # Use descriptor_bias_score for correlation now
                    gender_bias_data['descriptor_bias_score'],
                    self.discipline_gender_gaps[discipline]['gap']
                )
            }

        return explanation

    # --- ADDED prediction and selected_gender PARAMETERS ---
    def _analyze_gender_patterns(self, text, tokens, attention_weights, prediction, selected_gender):
        """
        Analyze gendered patterns based on descriptors and selected gender.
        """
        # --- REMOVED explicit_male/female from categories ---
        descriptor_categories = {
            'intellect_achievement': [],
            'entertainment_authority': [],
            'competence_organization': [],
            'warmth_nurturing': [],
            'male_negative': [],
            'female_negative': [],
            'objective_pedagogical': [], # NEW
            'other': []
        }

        # Associate tokens with categories
        for i, token in enumerate(tokens):
            token_lower = token.lower()
            # --- ADD check for OBJECTIVE_PEDAGOGICAL first ---
            if token_lower in OBJECTIVE_PEDAGOGICAL_DESCRIPTORS:
                 descriptor_categories['objective_pedagogical'].append((i, token, attention_weights[i]))
            # (Checks for other descriptors remain, order might matter slightly if overlap exists)
            elif token_lower in INTELLECT_ACHIEVEMENT_DESCRIPTORS:
                descriptor_categories['intellect_achievement'].append((i, token, attention_weights[i]))
            elif token_lower in ENTERTAINMENT_AUTHORITY_DESCRIPTORS:
                descriptor_categories['entertainment_authority'].append((i, token, attention_weights[i]))
            elif token_lower in COMPETENCE_ORGANIZATION_DESCRIPTORS:
                descriptor_categories['competence_organization'].append((i, token, attention_weights[i]))
            elif token_lower in WARMTH_NURTURING_DESCRIPTORS:
                descriptor_categories['warmth_nurturing'].append((i, token, attention_weights[i]))
            elif token_lower in MALE_NEGATIVE_DESCRIPTORS:
                descriptor_categories['male_negative'].append((i, token, attention_weights[i]))
            elif token_lower in FEMALE_NEGATIVE_DESCRIPTORS:
                descriptor_categories['female_negative'].append((i, token, attention_weights[i]))
            else:
                descriptor_categories['other'].append((i, token, attention_weights[i]))

        # Calculate attention by category (includes new category)
        category_attention = {}
        for category, items in descriptor_categories.items():
            category_attention[category] = sum(w for _, _, w in items) if items else 0

        # Calculate total attention and percentages (includes new category)
        total_attention = sum(v for k, v in category_attention.items() if k != 'other') # Exclude 'other' for clearer bias %
        if total_attention == 0: total_attention = 1
        category_attention_pct = {
            k: (v / total_attention) * 100 if k != 'other' else category_attention.get('other',0) # Keep 'other' raw or handle differently
            for k, v in category_attention.items()
        }
        # Ensure 'other' doesn't skew percentages if needed, recalc total_attention without 'other' if interpreting percentages directly.
        # Let's refine percentage calculation slightly for clarity in interpretation
        total_meaningful_attention = sum(v for k, v in category_attention.items() if k != 'other')
        if total_meaningful_attention == 0: total_meaningful_attention = 1
        category_attention_pct = {
             k: (v / total_meaningful_attention) * 100
             for k, v in category_attention.items() if k != 'other'
        }
        category_attention_pct['other'] = category_attention.get('other', 0) # Keep other separate


        # --- RECALCULATE BIAS SCORE BASED ON STEREOTYPICAL TERMS ---
        # Option 1: Male Positive vs Female Positive (Stereotypical)
        male_stereotype_weight = (category_attention['intellect_achievement'] +
                                  category_attention['entertainment_authority'])
        female_stereotype_weight = (category_attention['competence_organization'] + # Now has fewer objective terms
                                    category_attention['warmth_nurturing'])
        stereotype_bias_score = 0
        if (male_stereotype_weight + female_stereotype_weight) > 0:
            stereotype_bias_score = (male_stereotype_weight - female_stereotype_weight) / (male_stereotype_weight + female_stereotype_weight)

        # Keep the old descriptor_bias_score calculation as well for potential secondary analysis/display
        # but the primary interpretation will use stereotype_bias_score and objective_pct
        male_pattern_weight = male_stereotype_weight + category_attention['male_negative']
        female_pattern_weight = female_stereotype_weight + category_attention['female_negative']
        original_descriptor_bias_score = 0 # Renamed
        if (male_pattern_weight + female_pattern_weight) > 0:
            original_descriptor_bias_score = (male_pattern_weight - female_pattern_weight) / (male_pattern_weight + female_pattern_weight)

        # Find top attended terms by category (includes new category)
        top_terms_by_category = {}
        for category, items in descriptor_categories.items():
            top_terms_by_category[category] = sorted(items, key=lambda x: x[2], reverse=True)[:5]

        # --- CALL NEW interpretation function ---
        interpretation = self._interpret_gender_bias(
            stereotype_bias_score, # Use the score based on stereotypes
            category_attention_pct, # Pass all percentages
            prediction,
            selected_gender
        )

        # --- UPDATED RETURNED DICTIONARY ---
        return {
            'descriptor_categories': descriptor_categories,
            'category_attention': category_attention,
            'category_attention_pct': category_attention_pct,
            'stereotype_bias_score': stereotype_bias_score, # Renamed bias score
            'original_descriptor_bias_score': original_descriptor_bias_score, # Keep for reference/other uses if needed
            'top_terms_by_category': top_terms_by_category,
            'interpretation': interpretation
        }

    # --- COMPLETELY REWRITTEN INTERPRETATION LOGIC ---
# --- REVISED INTERPRETATION LOGIC using Objective Category ---
    def _interpret_gender_bias(self, stereotype_bias, attention_pct, sentiment, selected_gender):
        """Generate interpretation prioritizing objective focus."""
        interpretation = []
        # Use percentages from attention_pct, ensure keys exist
        obj_pct = attention_pct.get('objective_pedagogical', 0)
        comp_org_pct = attention_pct.get('competence_organization', 0)
        warmth_pct = attention_pct.get('warmth_nurturing', 0)
        intellect_pct = attention_pct.get('intellect_achievement', 0)
        entertain_pct = attention_pct.get('entertainment_authority', 0)
        f_neg_pct = attention_pct.get('female_negative', 0)
        m_neg_pct = attention_pct.get('male_negative', 0)

        thresholds = {
            'objective_focus': 20.0,  # % attention on objective terms to classify as neutral
            'negative_focus': 10.0,   # % attention on gendered negative terms
            'stereotype_focus': 15.0, # % attention on specific stereotypical positive categories
            'stereotype_bias': 0.20   # Threshold for the stereotype_bias_score
        }

        # --- Step 1: Check for High Objective Focus ---
        # If a significant portion of attention is on objective terms, classify as neutral first.
        if obj_pct > thresholds['objective_focus']:
             # Refine: Check if negative terms are *also* high, which might override objective neutrality
             high_neg = (selected_gender == 'Female' and f_neg_pct > thresholds['negative_focus']) or \
                        (selected_gender == 'Male' and m_neg_pct > thresholds['negative_focus'])
             if not high_neg:
                 if sentiment == "Positive":
                      interpretation.append(
                          f"Positive Sentiment - Objective Focus ({obj_pct:.1f}%): The comment focuses primarily on objective aspects of teaching (e.g., structure, clarity, feedback) without strong gendered patterns."
                      )
                 else: # sentiment == "Negative"
                      interpretation.append(
                          f"Negative Sentiment - Objective Focus ({obj_pct:.1f}%): Criticism focuses primarily on objective aspects of teaching (e.g., structure, clarity, feedback) without strong gendered patterns."
                      )
                 return interpretation # Return early, objective focus is primary finding

        # --- Step 2: Check for Strong Negative Gendered Criticism (If not objective) ---
        if not interpretation and sentiment == "Negative":
            if selected_gender == 'Female' and f_neg_pct > thresholds['negative_focus']:
                 interpretation.append(
                     f"Strong Negative Bias towards Female: Significant focus ({f_neg_pct:.1f}%) on negative terms typically associated with criticism of women."
                 )
                 return interpretation
            elif selected_gender == 'Male' and m_neg_pct > thresholds['negative_focus']:
                 interpretation.append(
                     f"Strong Negative Bias towards Male: Significant focus ({m_neg_pct:.1f}%) on negative terms typically associated with criticism of men."
                 )
                 return interpretation

        # --- Step 3: Check for Positive Stereotypical Praise (If not objective or strong negative) ---
        if not interpretation and sentiment == "Positive":
            if selected_gender == 'Female':
                if warmth_pct > thresholds['stereotype_focus'] and stereotype_bias < -thresholds['stereotype_bias']:
                     interpretation.append(
                         f"Moderate Positive Bias towards Female (Stereotypical Praise): Evaluation focuses heavily ({warmth_pct:.1f}%) on warmth/nurturing, potentially reinforcing stereotypes."
                     )
            elif selected_gender == 'Male':
                if entertain_pct > thresholds['stereotype_focus'] and stereotype_bias > thresholds['stereotype_bias']:
                    interpretation.append(
                        f"Moderate Positive Bias towards Male (Stereotypical Praise): Evaluation focuses heavily ({entertain_pct:.1f}%) on entertainment/authority, potentially reinforcing stereotypes."
                    )

        # --- Step 4: Check General Stereotype Balance (If nothing else triggered) ---
        # Use the stereotype_bias_score, which excludes objective terms.
        if not interpretation:
            abs_stereotype_bias = abs(stereotype_bias)
            if abs_stereotype_bias > thresholds['stereotype_bias']:
                 if stereotype_bias > 0: # More male-associated stereotypes
                     bias_type = "male-associated (intellect, entertainment)"
                     other_type = "female-associated (competence, warmth)"
                     focus_detail = f" (Intellect: {intellect_pct:.1f}%, Entertain: {entertain_pct:.1f}%)"
                     if selected_gender == 'Male':
                          interpretation.append(
                              f"Focus on Male-Associated Stereotypes: Language emphasizes {bias_type}{focus_detail} over {other_type}. Aligns with common patterns for men."
                          )
                     else:
                          interpretation.append(
                              f"Focus on Male-Associated Stereotypes: Language emphasizes {bias_type}{focus_detail} over {other_type}. Less typical pattern for female evaluations."
                          )
                 else: # More female-associated stereotypes
                     bias_type = "female-associated (competence, warmth)"
                     other_type = "male-associated (intellect, entertainment)"
                     focus_detail = f" (Competence: {comp_org_pct:.1f}%, Warmth: {warmth_pct:.1f}%)"
                     if selected_gender == 'Female':
                          interpretation.append(
                              f"Focus on Female-Associated Stereotypes: Language emphasizes {bias_type}{focus_detail} over {other_type}. Aligns with common patterns for women."
                          )
                     else:
                          interpretation.append(
                              f"Focus on Female-Associated Stereotypes: Language emphasizes {bias_type}{focus_detail} over {other_type}. Less typical pattern for male evaluations."
                          )

        # --- Step 5: Final Fallback - Truly Neutral/Undetermined ---
        if not interpretation:
             # Reached only if objective focus wasn't high, negative bias wasn't high,
             # positive stereotypes weren't high, and stereotype balance score was low.
             if sentiment == "Positive":
                  interpretation.append(
                      "Positive Sentiment - Neutral Language Pattern: The comment expresses positive sentiment without strongly aligning with common gendered evaluation patterns or focusing heavily on objective terms."
                  )
             else: # sentiment == "Negative"
                  interpretation.append(
                      "Negative Sentiment - Neutral Language Pattern: The comment expresses criticism without strongly aligning with common gendered evaluation patterns or focusing heavily on objective terms."
                  )

        return interpretation

    def _calculate_attention_gap_correlation(self, stereotype_bias_score, rating_gap):
        """Calculate if stereotype attention bias aligns with rating gap"""
        # Use stereotype_bias_score now
        if (stereotype_bias_score > 0 and rating_gap > 0) or \
           (stereotype_bias_score < 0 and rating_gap < 0) or \
           (abs(stereotype_bias_score) < 0.1 and abs(rating_gap) < 0.1) :
            alignment = "aligned"
        else:
            alignment = "contrary"

        return {
            'alignment': alignment,
            'explanation': self._generate_alignment_explanation(stereotype_bias_score, rating_gap, alignment) # Pass stereotype score
        }

    def _generate_alignment_explanation(self, stereotype_bias_score, rating_gap, alignment):
        """Generate explanation of alignment between stereotype attention and ratings"""
        # Use stereotype_bias_score
        abs_attention = abs(stereotype_bias_score)
        attention_focus = "strong" if abs_attention > 0.4 else "moderate" if abs_attention > 0.15 else "weak"
        attention_dir = "male-associated stereotypes" if stereotype_bias_score > 0 else "female-associated stereotypes"

        # (Rest of gap logic is the same)
        abs_gap = abs(rating_gap)
        gap_strength = "significant" if abs_gap > 0.4 else "moderate" if abs_gap > 0.15 else "minor"
        gender_ratings = "male" if rating_gap >= 0 else "female"

        if alignment == "aligned":
            return (f"The {attention_focus} focus on {attention_dir} aligns with the {gap_strength} rating gap favoring {gender_ratings} professors in this discipline.")
        else:
             return (f"The {attention_focus} focus on {attention_dir} is contrary to the {gap_strength} rating gap favoring {gender_ratings} professors in this discipline.")

    # --- UPDATE VISUALIZATION ---
    def visualize_attention(self, text, selected_gender, discipline=None, save_path=None):
        """Generate attention visualization with descriptor highlighting"""
        explanation = self.explain_prediction(text, selected_gender, discipline)
        tokens = explanation['tokens']
        attention = np.array(explanation['attention']) # Ensure numpy array

        plt.figure(figsize=(14, 6))

        # Use descriptor categories for coloring (simplified)
        colors = []
        for i, token in enumerate(tokens):
            # Use the data already categorized in explanation['gender_bias']['descriptor_categories']
            cat_found = False
            if explanation['gender_bias']['descriptor_categories']['intellect_achievement'] and any(t[0] == i for t in explanation['gender_bias']['descriptor_categories']['intellect_achievement']):
                colors.append('indigo') # Intellect/achievement
                cat_found = True
            elif explanation['gender_bias']['descriptor_categories']['entertainment_authority'] and any(t[0] == i for t in explanation['gender_bias']['descriptor_categories']['entertainment_authority']):
                colors.append('purple') # Entertainment/authority
                cat_found = True
            elif explanation['gender_bias']['descriptor_categories']['competence_organization'] and any(t[0] == i for t in explanation['gender_bias']['descriptor_categories']['competence_organization']):
                colors.append('darkorange') # Competence/organization
                cat_found = True
            elif explanation['gender_bias']['descriptor_categories']['warmth_nurturing'] and any(t[0] == i for t in explanation['gender_bias']['descriptor_categories']['warmth_nurturing']):
                colors.append('pink') # Warmth/nurturing
                cat_found = True
            elif explanation['gender_bias']['descriptor_categories']['male_negative'] and any(t[0] == i for t in explanation['gender_bias']['descriptor_categories']['male_negative']):
                colors.append('slategray') # Male-associated negative
                cat_found = True
            elif explanation['gender_bias']['descriptor_categories']['female_negative'] and any(t[0] == i for t in explanation['gender_bias']['descriptor_categories']['female_negative']):
                colors.append('crimson') # Female-associated negative
                cat_found = True

            if not cat_found:
                 colors.append('gray') # Neutral/other

        # Plot attention weights (Use attention array directly)
        plt.bar(range(len(tokens)), attention, color=colors)
        plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
        plt.xlabel('Tokens')
        plt.ylabel('Attention Weight')

        descriptor_bias = explanation['gender_bias']['descriptor_bias_score']
        # --- UPDATE TITLE ---
        title = f"Sentiment: {explanation['prediction']} ({explanation['confidence']:.2f}) | Evaluated Gender: {selected_gender}\n"
        # Add main bias interpretation
        title += f"Bias Assessment: {explanation['gender_bias']['interpretation'][0]}\n" if explanation['gender_bias']['interpretation'] else ""
        title += f"Descriptor Focus Score: {descriptor_bias:.2f} "
        if descriptor_bias > 0.15: title += "(Focus leans Male-Associated)"
        elif descriptor_bias < -0.15: title += "(Focus leans Female-Associated)"
        else: title += "(Focus relatively balanced)"

        if 'discipline_context' in explanation:
            disc_data = explanation['discipline_context']
            title += f"\nDiscipline: {disc_data['discipline']} (Rating Gap: {disc_data['gender_rating_gap']:.2f})"

        plt.title(title, fontsize=10) # Reduced font size for potentially long titles

        # --- UPDATE LEGEND ---
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='indigo', label='Intellect/Achieve'),
            Patch(facecolor='purple', label='Entertain/Authority'),
            Patch(facecolor='darkorange', label='Competence/Org'),
            Patch(facecolor='pink', label='Warmth/Nurture'),
            Patch(facecolor='slategray', label='Male Neg.'),
            Patch(facecolor='crimson', label='Female Neg.'),
            Patch(facecolor='gray', label='Other Terms')
        ]
        plt.legend(handles=legend_elements, fontsize=8) # Smaller font

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    # --- UPDATE BATCH ANALYSIS ---
    def analyze_comments_batch(self, comments, selected_gender, disciplines=None):
        """Analyze a batch of comments for a specific gender context."""
        results = []
        for i, comment in enumerate(comments):
            discipline = disciplines[i] if disciplines and i < len(disciplines) else None
            # --- Pass selected_gender to explain_prediction ---
            explanation = self.explain_prediction(comment, selected_gender, discipline)
            results.append(explanation)

        # Aggregate results (focus on descriptor bias and sentiment)
        # --- REMOVED overall_bias_score aggregation based on old formula ---
        descriptor_bias_score = np.mean([r['gender_bias']['descriptor_bias_score'] for r in results])
        pos_results = [r for r in results if r['prediction'] == 'Positive']
        neg_results = [r for r in results if r['prediction'] == 'Negative']

        # --- REMOVED pos/neg bias score aggregation based on old formula ---

        # Analyze descriptor patterns by sentiment
        pos_descriptor_bias = np.mean([r['gender_bias']['descriptor_bias_score'] for r in pos_results]) if pos_results else 0
        neg_descriptor_bias = np.mean([r['gender_bias']['descriptor_bias_score'] for r in neg_results]) if neg_results else 0

        # Collect interpretations
        all_interpretations = [interp for r in results for interp in r['gender_bias']['interpretation']]
        interpretation_counts = Counter(all_interpretations)

        # Collect terms by category (no change needed here, just remove explicit later)
        category_terms = defaultdict(list)
        for r in results:
            for category, terms in r['gender_bias']['descriptor_categories'].items():
                 # --- Exclude explicit categories if they somehow sneak back in ---
                 if category not in ['explicit_male', 'explicit_female']:
                    category_terms[category].extend([(term, weight) for _, term, weight in terms])

        # Process term statistics by category (no change)
        category_stats = {}
        for category, terms in category_terms.items():
            if not terms: continue
            term_counter = Counter([term.lower() for term, _ in terms])
            term_avg_attn = defaultdict(list)
            for term, weight in terms: term_avg_attn[term.lower()].append(weight)
            term_avg = {term: sum(weights) / len(weights) for term, weights in term_avg_attn.items()}
            term_relevance = {term: count * term_avg.get(term, 0) for term, count in term_counter.items()}
            top_terms = sorted(term_relevance.items(), key=lambda x: x[1], reverse=True)[:10]
            category_stats[category] = {
                'top_terms': top_terms,
                'total_occurrences': len(terms),
                'unique_terms': len(term_counter)
            }

        # Generate insights based on the new interpretation focus
        insights = self._generate_batch_insights(
            selected_gender, # Pass gender
            descriptor_bias_score,
            pos_descriptor_bias,
            neg_descriptor_bias,
            category_stats,
            interpretation_counts, # Pass interpretation counts
            len(pos_results),
            len(neg_results)
        )

        # --- UPDATED RETURN DICTIONARY ---
        return {
            # 'overall_bias_score': overall_bias_score, # REMOVED
            'selected_gender_context': selected_gender,
            'descriptor_bias_score': descriptor_bias_score,
            # 'positive_comments_bias_score': pos_bias_score, # REMOVED
            # 'negative_comments_bias_score': neg_bias_score, # REMOVED
            'positive_descriptor_bias': pos_descriptor_bias,
            'negative_descriptor_bias': neg_descriptor_bias,
            'comment_count': len(results),
            'positive_count': len(pos_results),
            'negative_count': len(neg_results),
            'category_stats': category_stats, # Explicit categories removed
            'interpretation_summary': dict(interpretation_counts.most_common(5)), # Top 5 interpretations
            'insights': insights
        }

    # --- REVISED BATCH INSIGHTS ---
    def _generate_batch_insights(self, selected_gender, descriptor_bias, pos_descriptor_bias,
                               neg_descriptor_bias, category_stats, interpretation_counts,
                               pos_count, neg_count):
        """Generate insights about bias patterns based on new interpretation."""
        insights = []
        total_comments = pos_count + neg_count

        # 1. Dominant Bias Interpretation
        if interpretation_counts:
            most_common_interp, count = interpretation_counts.most_common(1)[0]
            if count / total_comments > 0.3: # If >30% comments share the top interpretation
                 insights.append(f"Predominant Finding: A significant portion ({count}/{total_comments}) of comments were assessed as: '{most_common_interp}'.")

        # 2. Overall Descriptor Focus Pattern
        if abs(descriptor_bias) > 0.15:
            focus = "male-associated (intellect/entertainment)" if descriptor_bias > 0 else "female-associated (competence/warmth)"
            insights.append(
                f"Overall Descriptor Focus: Comments tend to emphasize {focus} descriptors, which may reflect biased expectations for a {selected_gender} individual."
            )

        # 3. Descriptor Focus differences between Positive and Negative comments
        if abs(pos_descriptor_bias - neg_descriptor_bias) > 0.2:
            if pos_descriptor_bias > neg_descriptor_bias:
                 insights.append(
                    "Praise vs. Criticism Focus: Positive comments emphasize intellect/entertainment more, while negative comments focus more on competence/warmth."
                 )
            else:
                 insights.append(
                    "Praise vs. Criticism Focus: Negative comments emphasize intellect/entertainment more, while positive comments focus more on competence/warmth."
                 )

        # 4. Highlighting specific category dominance (if extreme)
        male_assoc_pct = category_stats.get('intellect_achievement', {}).get('total_occurrences', 0) + \
                         category_stats.get('entertainment_authority', {}).get('total_occurrences', 0)
        female_assoc_pct = category_stats.get('competence_organization', {}).get('total_occurrences', 0) + \
                           category_stats.get('warmth_nurturing', {}).get('total_occurrences', 0)
        total_desc = male_assoc_pct + female_assoc_pct

        if total_desc > 0:
             if male_assoc_pct / total_desc > 0.65:
                 insights.append("Strong Focus Skew: Comments heavily emphasize male-associated descriptors (intellect, entertainment) over female-associated ones.")
             elif female_assoc_pct / total_desc > 0.65:
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