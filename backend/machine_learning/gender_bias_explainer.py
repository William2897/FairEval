import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import os
import json

# Define lexicons for different types of descriptors commonly used in evaluations
# Research shows these patterns differ by gender in evaluations

# Terms more commonly associated with male professors in positive evaluations
INTELLECT_ACHIEVEMENT_DESCRIPTORS = {
    'brilliant', 'smart', 'intelligent', 'genius', 'intellectual', 'knowledgeable', 'expert',
    'outstanding', 'excellent', 'exceptional', 'remarkable', 'phenomenal', 'best', 'talented',
    'impressive', 'fascinating', 'thought-provoking', 'thought', 'deep', 'analytical', 'original',
    'innovative', 'creative', 'critical', 'sharp', 'astute', 'wisdom', 'genius', 'masterful',
    'theoretical', 'philosophical', 'scientific', 'technical', 'research', 'scholar', 'scholarly',
    'academic', 'intellectual', 'scientific', 'rational', 'logical', 'rigorous', 'demanding'
}

# Terms more commonly associated with male professors - entertainment/authority
ENTERTAINMENT_AUTHORITY_DESCRIPTORS = {
    'funny', 'cool', 'entertaining', 'hilarious', 'humor', 'witty', 'laugh', 'funniest', 
    'enjoyable', 'awesome', 'amazing', 'fun', 'interesting', 'engaging', 'enthusiastic', 
    'charismatic', 'energetic', 'charming', 'dynamic', 'lively', 'animated', 'captivating', 
    'confident', 'authoritative', 'powerful', 'strong', 'commanding', 'bold', 'direct', 
    'straightforward', 'assertive', 'decisive', 'fair', 'objective', 'tough', 'challenging',
    'respect', 'respectable', 'no-nonsense', 'leader', 'leadership', 'mentor'
}

# Terms more commonly associated with female professors - competence/organization
COMPETENCE_ORGANIZATION_DESCRIPTORS = {
    'organized', 'prepared', 'thorough', 'clear', 'concise', 'detailed', 'accurate',
    'precise', 'effective', 'efficient', 'informative', 'helpful', 'methodical',
    'structured', 'comprehensive', 'orderly', 'systematic', 'timely', 'punctual', 
    'consistent', 'reliable', 'dependable', 'professional', 'diligent', 'meticulous',
    'careful', 'attentive', 'focused', 'practical', 'pragmatic', 'reasonable',
    'flexible', 'adaptable', 'accessible', 'available', 'responsive', 'feedback',
    'correct', 'improve', 'improvement', 'clear', 'clarity', 'communication'
}

# Terms more commonly associated with female professors - warmth/nurturing
WARMTH_NURTURING_DESCRIPTORS = {
    'personable', 'caring', 'friendly', 'nice', 'sweet', 'kind', 'relatable', 'approachable',
    'easygoing', 'understanding', 'relaxed', 'cheerful', 'excited', 'passionate', 'warm', 
    'delightful', 'joy', 'pleasant', 'lovely', 'wonderful', 'sweetheart', 'positive',
    'supportive', 'encouraging', 'nurturing', 'empathetic', 'compassionate', 'sympathetic',
    'thoughtful', 'considerate', 'gentle', 'patient', 'maternal', 'motherly', 'comfort',
    'comfortable', 'confidence', 'safe', 'welcoming', 'inclusive', 'loved', 'adore',
    'help', 'helpful', 'dedicated', 'devoted', 'committed', 'accommodating', 'gracious'
}

# Potentially negative descriptors commonly used for male professors
MALE_NEGATIVE_DESCRIPTORS = {
    'boring', 'bored', 'bore', 'tedious', 'dull', 'monotonous', 'useless', 'waste',
    'harsh', 'brutal', 'intimidating', 'arrogant', 'condescending', 'dismissive',
    'egotistical', 'pompous', 'aggressive', 'rude', 'insensitive', 'aloof', 'cold',
    'distant', 'detached', 'unapproachable', 'joke', 'idiot', 'insult', 'ass',
    'jerk', 'terrible', 'horrible', 'awful', 'useless', 'incompetent', 'lazy',
    'disorganized', 'unprepared', 'confusing', 'unclear', 'tricky', 'challenging',
    'difficult', 'problem', 'unfair', 'biased', 'cheat', 'hard'
}

# Potentially negative descriptors commonly used for female professors  
FEMALE_NEGATIVE_DESCRIPTORS = {
    'unprofessional', 'emotional', 'moody', 'sensitive', 'defensive', 'dramatic',
    'scattered', 'disorganized', 'unclear', 'confusing', 'vague', 'rambling',
    'chatty', 'talkative', 'loud', 'shrill', 'annoying', 'irritating', 'frustrating',
    'difficult', 'picky', 'fussy', 'demanding', 'strict', 'harsh', 'mean', 'nasty',
    'rude', 'unfriendly', 'cold', 'uptight', 'rigid', 'inflexible', 'unreasonable',
    'unforgiving', 'stressed', 'stressful', 'anxious', 'nervous', 'worried', 'frazzled',
    'overwhelmed', 'scattered', 'condescend', 'condescending', 'patronizing', 'belittling'
}

# Recognize explicit gendered terms for minimal reference support
EXPLICIT_GENDERED_TERMS = {
    'male': {'he', 'him', 'his', 'himself', 'mr', 'sir', 'man', 'men', 'guy', 'guys', 'dude', 
            'father', 'dad', 'brother', 'gentleman', 'gentlemen', 'male', 'males', 'boy', 'boys',
            'husband', 'son', 'sons', 'uncle', 'grandfather', 'grandson', 'king', 'prince'},
    'female': {'she', 'her', 'hers', 'herself', 'ms', 'mrs', 'miss', 'madam', 'woman', 'women', 
              'gal', 'lady', 'ladies', 'mother', 'mom', 'sister', 'girl', 'girls', 'female', 
              'females', 'wife', 'daughter', 'daughters', 'aunt', 'grandmother', 'granddaughter',
              'queen', 'princess'}
}


class GenderBiasExplainer:
    def __init__(self, model, vocab):
        """
        Initialize an explainer for gender bias in LSTM attention
        
        Args:
            model: trained LSTM model with attention
            vocab: vocabulary dictionary (word to index)
        """
        self.model = model
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.device = next(model.parameters()).device
                
        # Load discipline gender gaps from the database if available
        self.discipline_gender_gaps = self._load_discipline_gender_gaps()
        
    def _load_discipline_gender_gaps(self):
        """Load pre-calculated gender gaps in ratings by discipline"""
        try:
            # Try to import Django utilities
            import django
            os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'faireval.settings')
            django.setup()
            
            from api.utils import calculate_gender_discipline_heatmap
            heatmap_data = calculate_gender_discipline_heatmap()
            
            # Transform into a lookup dictionary
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
                    
            # Calculate gaps
            for discipline, data in discipline_gaps.items():
                if data['male_rating'] and data['female_rating']:
                    data['gap'] = round(data['male_rating'] - data['female_rating'], 2)
                    
            return discipline_gaps
            
        except (ImportError, ModuleNotFoundError):
            # If we can't import Django components, return empty dict
            print("Warning: Cannot load discipline gender gaps - Django environment not available")
            return {}
        except Exception as e:
            print(f"Error loading discipline gender gaps: {str(e)}")
            return {}
        
    def explain_prediction(self, text, discipline=None, return_attention_only=False):
        """
        Explain model prediction with attention weights and gender bias analysis
        
        Args:
            text: Text input to analyze
            discipline: Optional discipline for contextual analysis
            return_attention_only: If True, return only the attention weights
            
        Returns:
            Dictionary with prediction explanation and bias analysis
        """
        # Tokenize and prepare input
        tokens = text.lower().split()
        indices = [self.vocab.get(t.lower(), 0) for t in tokens]
        
        # Handle sequence length
        max_len = 100  # Assuming this is the model's expected length
        if len(indices) > max_len:
            indices = indices[:max_len]
            tokens = tokens[:max_len]
        else:
            padding = [1] * (max_len - len(indices))  # 1 is padding index
            indices += padding
            
        # Get model prediction with attention weights
        input_tensor = torch.tensor([indices], dtype=torch.long).to(self.device)
        with torch.no_grad():
            pred, attention = self.model(input_tensor, return_attention=True)
            
        # Process attention weights
        attn_weights = attention.squeeze().cpu().numpy()[:len(tokens)]
        prediction = "Positive" if pred.item() >= 0.5 else "Negative"
        confidence = pred.item() if pred.item() >= 0.5 else 1 - pred.item()
        
        if return_attention_only:
            return attn_weights
        
        # Analyze gender bias in attention weights with enhanced descriptor patterns
        gender_bias_data = self._analyze_gender_patterns(text, tokens, attn_weights)
        
        explanation = {
            'prediction': prediction,
            'confidence': confidence,
            'tokens': tokens,
            'attention': attn_weights.tolist(),
            'gender_bias': gender_bias_data,
        }
        
        # Add discipline context if available
        if discipline and discipline in self.discipline_gender_gaps:
            explanation['discipline_context'] = {
                'discipline': discipline,
                'gender_rating_gap': self.discipline_gender_gaps[discipline]['gap'],
                'male_avg_rating': self.discipline_gender_gaps[discipline]['male_rating'],
                'female_avg_rating': self.discipline_gender_gaps[discipline]['female_rating'],
                'correlation': self._calculate_attention_gap_correlation(
                    gender_bias_data['bias_score'], 
                    self.discipline_gender_gaps[discipline]['gap']
                )
            }
            
        return explanation
        
    def _analyze_gender_patterns(self, text, tokens, attention_weights):
        """
        Analyze gendered patterns in evaluation text, including descriptor types and sentiment
        """
        # Categorize tokens by descriptor type
        descriptor_categories = {
            'intellect_achievement': [], 
            'entertainment_authority': [],
            'competence_organization': [],
            'warmth_nurturing': [],
            'male_negative': [],
            'female_negative': [],
            'explicit_male': [], 
            'explicit_female': [], 
            'other': []
        }
        
        # Associate tokens with categories
        for i, token in enumerate(tokens):
            token_lower = token.lower()
            if token_lower in INTELLECT_ACHIEVEMENT_DESCRIPTORS:
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
            elif token_lower in EXPLICIT_GENDERED_TERMS['male']:
                descriptor_categories['explicit_male'].append((i, token, attention_weights[i]))
            elif token_lower in EXPLICIT_GENDERED_TERMS['female']:
                descriptor_categories['explicit_female'].append((i, token, attention_weights[i]))
            else:
                descriptor_categories['other'].append((i, token, attention_weights[i]))
        
        # Calculate attention by category
        category_attention = {}
        for category, items in descriptor_categories.items():
            category_attention[category] = sum(w for _, _, w in items) if items else 0
        
        # Calculate total attention for normalization
        total_attention = sum(category_attention.values())
        if total_attention == 0:
            total_attention = 1  # Avoid division by zero
            
        # Calculate percentages
        category_attention_pct = {
            k: (v / total_attention) * 100 
            for k, v in category_attention.items()
        }
        
        # Calculate male-associated vs female-associated descriptor bias
        # Male-associated: intellect_achievement + entertainment_authority + male_negative
        # Female-associated: competence_organization + warmth_nurturing + female_negative
        male_pattern_weight = (category_attention['intellect_achievement'] + 
                               category_attention['entertainment_authority'] + 
                               category_attention['male_negative'])
                               
        female_pattern_weight = (category_attention['competence_organization'] + 
                                category_attention['warmth_nurturing'] + 
                                category_attention['female_negative'])
        
        descriptor_bias_score = 0
        if (male_pattern_weight + female_pattern_weight) > 0:
            descriptor_bias_score = (male_pattern_weight - female_pattern_weight) / (male_pattern_weight + female_pattern_weight)
        
        # Calculate explicit gender bias - positive means male, negative means female
        explicit_gender_bias = 0
        total_explicit = category_attention['explicit_male'] + category_attention['explicit_female']
        if total_explicit > 0:
            explicit_gender_bias = (category_attention['explicit_male'] - category_attention['explicit_female']) / total_explicit
            
        # Calculate overall bias score - weighted combination of descriptor pattern bias and explicit gender bias
        bias_score = 0.7 * descriptor_bias_score + 0.3 * explicit_gender_bias
        
        # Find top attended terms by category
        top_terms_by_category = {}
        for category, items in descriptor_categories.items():
            top_terms_by_category[category] = sorted(items, key=lambda x: x[2], reverse=True)[:5]
        
        # Create interpretation of the bias
        interpretation = self._interpret_gender_bias(
            descriptor_bias_score, 
            explicit_gender_bias, 
            category_attention_pct
        )
        
        return {
            'descriptor_categories': descriptor_categories,
            'category_attention': category_attention,
            'category_attention_pct': category_attention_pct,
            'descriptor_bias_score': descriptor_bias_score,
            'explicit_gender_bias': explicit_gender_bias,
            'bias_score': bias_score,
            'top_terms_by_category': top_terms_by_category,
            'interpretation': interpretation
        }
    
    def _interpret_gender_bias(self, descriptor_bias, explicit_bias, attention_pct):
        """Generate interpretation of the detected gender bias patterns"""
        # Determine bias direction and strength
        abs_descriptor_bias = abs(descriptor_bias)
        descriptor_bias_dir = "male-pattern" if descriptor_bias > 0 else "female-pattern"
        descriptor_strength = "strong" if abs_descriptor_bias > 0.5 else "moderate" if abs_descriptor_bias > 0.2 else "weak"
        
        # Calculate combined percentages for different descriptor types
        male_pattern_pct = (attention_pct['intellect_achievement'] + 
                           attention_pct['entertainment_authority'] + 
                           attention_pct['male_negative'])
                           
        female_pattern_pct = (attention_pct['competence_organization'] + 
                             attention_pct['warmth_nurturing'] + 
                             attention_pct['female_negative'])
        
        # Put interpretation together
        interpretation = []
        
        # Analyze balance between descriptor categories
        if abs_descriptor_bias > 0.2:
            if descriptor_bias > 0:
                interpretation.append(
                    f"This evaluation shows a {descriptor_strength} focus on male-associated language patterns "
                    f"({male_pattern_pct:.1f}% of attention) versus female-associated patterns ({female_pattern_pct:.1f}% of attention)."
                )
                
                # Add more specific analysis
                if attention_pct['intellect_achievement'] > 15:
                    interpretation.append(
                        f"There is significant emphasis on intellectual achievement and brilliance, "
                        f"which research shows is more commonly associated with evaluations of male professors."
                    )
                if attention_pct['entertainment_authority'] > 15:
                    interpretation.append(
                        f"The evaluation focuses on entertainment value and authority, "
                        f"attributes that are more frequently emphasized in evaluations of male professors."
                    )
            else:
                interpretation.append(
                    f"This evaluation shows a {descriptor_strength} focus on female-associated language patterns "
                    f"({female_pattern_pct:.1f}% of attention) versus male-associated patterns ({male_pattern_pct:.1f}% of attention)."
                )
                
                # Add more specific analysis
                if attention_pct['competence_organization'] > 15:
                    interpretation.append(
                        f"There is significant emphasis on competence and organizational skills, "
                        f"which research shows are more commonly highlighted in evaluations of female professors."
                    )
                if attention_pct['warmth_nurturing'] > 15:
                    interpretation.append(
                        f"The evaluation focuses on warmth, nurturing qualities, and personality, "
                        f"attributes that are more frequently emphasized in evaluations of female professors."
                    )
        
        # Comment on explicit gender references if significant
        if abs(explicit_bias) > 0.3:
            explicit_gender = "male" if explicit_bias > 0 else "female"
            interpretation.append(
                f"There are significant explicit references to {explicit_gender} gender in this evaluation."
            )
        
        # Add note about negative descriptors if present
        if attention_pct['male_negative'] > 10 or attention_pct['female_negative'] > 10:
            if attention_pct['male_negative'] > attention_pct['female_negative']:
                interpretation.append(
                    f"The negative language used is more typical of criticisms directed at male professors "
                    f"(focusing on being boring, harsh, or arrogant)."
                )
            else:
                interpretation.append(
                    f"The negative language used is more typical of criticisms directed at female professors "
                    f"(focusing on being unprofessional, emotional, or too strict)."
                )
        
        return interpretation
    
    def _calculate_attention_gap_correlation(self, attention_bias_score, rating_gap):
        """Calculate if attention bias aligns with rating gap in discipline"""
        # Both positive = both favor male (aligned)
        # Both negative = both favor female (aligned)
        # Different signs = misaligned
        
        if (attention_bias_score > 0 and rating_gap > 0) or (attention_bias_score < 0 and rating_gap < 0):
            alignment = "aligned"
        else:
            alignment = "contrary"
            
        return {
            'alignment': alignment,
            'explanation': self._generate_alignment_explanation(attention_bias_score, rating_gap, alignment)
        }
        
    def _generate_alignment_explanation(self, attention_bias_score, rating_gap, alignment):
        """Generate human-readable explanation of alignment between attention and ratings"""
        abs_attention = abs(attention_bias_score)
        attention_strength = "strong" if abs_attention > 0.5 else "moderate" if abs_attention > 0.2 else "weak"
        
        abs_gap = abs(rating_gap)
        gap_strength = "significant" if abs_gap > 0.5 else "moderate" if abs_gap > 0.2 else "minor"
        
        gender_attention = "male" if attention_bias_score > 0 else "female"
        gender_ratings = "male" if rating_gap > 0 else "female"
        
        if alignment == "aligned":
            return (f"This comment shows {attention_strength} {gender_attention}-biased language patterns, "
                   f"which aligns with the {gap_strength} higher ratings typically given to "
                   f"{gender_ratings} professors in this discipline.")
        else:
            return (f"This comment shows {attention_strength} {gender_attention}-biased language patterns, "
                   f"yet {gender_ratings} professors typically receive higher ratings "
                   f"({gap_strength} difference) in this discipline.")
    
    def visualize_attention(self, text, discipline=None, save_path=None):
        """Generate attention visualization with gender highlighting"""
        explanation = self.explain_prediction(text, discipline)
        tokens = explanation['tokens']
        attention = explanation['attention']
        
        # Create figure
        plt.figure(figsize=(14, 6))
        
        # Enhanced color-coding for different descriptor categories
        colors = []
        for i, token in enumerate(tokens):
            token_lower = token.lower()
            if token_lower in INTELLECT_ACHIEVEMENT_DESCRIPTORS:
                colors.append('darkblue')  # Intellect/achievement descriptors
            elif token_lower in ENTERTAINMENT_AUTHORITY_DESCRIPTORS:
                colors.append('royalblue')  # Entertainment/authority descriptors
            elif token_lower in COMPETENCE_ORGANIZATION_DESCRIPTORS:
                colors.append('darkorange')  # Competence/organization descriptors
            elif token_lower in WARMTH_NURTURING_DESCRIPTORS:
                colors.append('pink')  # Warmth/nurturing descriptors
            elif token_lower in MALE_NEGATIVE_DESCRIPTORS:
                colors.append('navy')  # Male-associated negative descriptors
            elif token_lower in FEMALE_NEGATIVE_DESCRIPTORS:
                colors.append('crimson')  # Female-associated negative descriptors
            elif token_lower in EXPLICIT_GENDERED_TERMS['male']:
                colors.append('blue')    # Explicitly male terms
            elif token_lower in EXPLICIT_GENDERED_TERMS['female']:
                colors.append('red')     # Explicitly female terms
            else:
                colors.append('gray')    # Neutral/other terms
        
        # Plot attention weights
        plt.bar(range(len(tokens)), attention, color=colors)
        plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
        plt.xlabel('Tokens')
        plt.ylabel('Attention Weight')
        
        bias_score = explanation['gender_bias']['bias_score']
        descriptor_bias = explanation['gender_bias']['descriptor_bias_score']
        
        title = f"Prediction: {explanation['prediction']} (Confidence: {explanation['confidence']:.2f})\n"
        title += f"Gender Bias Score: {bias_score:.2f} "
        
        if bias_score > 0.2:
            title += "(Male-pattern language)"
        elif bias_score < -0.2:
            title += "(Female-pattern language)"
        else:
            title += "(Relatively neutral)"
            
        title += f"\nDescriptor Bias: {descriptor_bias:.2f} "
        
        if descriptor_bias > 0.2:
            title += "(Personality/entertainment focused)"
        elif descriptor_bias < -0.2:
            title += "(Competence focused)"
        else:
            title += "(Balanced descriptors)"
        
        # Add discipline context if available
        if 'discipline_context' in explanation:
            disc_data = explanation['discipline_context']
            title += f"\nDiscipline: {disc_data['discipline']} "
            title += f"(M: {disc_data['male_avg_rating']:.2f}, F: {disc_data['female_avg_rating']:.2f}, "
            title += f"Gap: {disc_data['gender_rating_gap']:.2f})"
            
        plt.title(title)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='darkblue', label='Intellect/Achievement'),
            Patch(facecolor='royalblue', label='Entertainment/Authority'),
            Patch(facecolor='darkorange', label='Competence/Organization'),
            Patch(facecolor='pink', label='Warmth/Nurturing'),
            Patch(facecolor='navy', label='Male-Associated Negative'),
            Patch(facecolor='crimson', label='Female-Associated Negative'),
            Patch(facecolor='blue', label='Explicit Male'),
            Patch(facecolor='red', label='Explicit Female'),
            Patch(facecolor='gray', label='Other Terms')
        ]
        plt.legend(handles=legend_elements)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    def analyze_comments_batch(self, comments, disciplines=None):
        """
        Analyze a batch of comments for gender bias patterns
        
        Args:
            comments: List of comment texts
            disciplines: Optional list of disciplines matching comments
            
        Returns:
            Dictionary with aggregated analysis results
        """
        results = []
        for i, comment in enumerate(comments):
            discipline = disciplines[i] if disciplines and i < len(disciplines) else None
            explanation = self.explain_prediction(comment, discipline)
            results.append(explanation)
            
        # Aggregate results
        overall_bias_score = np.mean([r['gender_bias']['bias_score'] for r in results])
        descriptor_bias_score = np.mean([r['gender_bias']['descriptor_bias_score'] for r in results])
        pos_results = [r for r in results if r['prediction'] == 'Positive']
        neg_results = [r for r in results if r['prediction'] == 'Negative']
        
        pos_bias_score = np.mean([r['gender_bias']['bias_score'] for r in pos_results]) if pos_results else 0
        neg_bias_score = np.mean([r['gender_bias']['bias_score'] for r in neg_results]) if neg_results else 0
        
        # Analyze descriptor patterns by sentiment
        pos_descriptor_bias = np.mean([r['gender_bias']['descriptor_bias_score'] for r in pos_results]) if pos_results else 0
        neg_descriptor_bias = np.mean([r['gender_bias']['descriptor_bias_score'] for r in neg_results]) if neg_results else 0
        
        # Collect terms by category
        category_terms = defaultdict(list)
        for r in results:
            for category, terms in r['gender_bias']['descriptor_categories'].items():
                category_terms[category].extend([(term, weight) for _, term, weight in terms])
            
        # Process term statistics by category
        category_stats = {}
        for category, terms in category_terms.items():
            if not terms:
                category_stats[category] = {'top_terms': []}
                continue
                
            # Count occurrences
            term_counter = Counter([term.lower() for term, _ in terms])
            
            # Calculate average attention per term
            term_avg_attn = defaultdict(list)
            for term, weight in terms:
                term_avg_attn[term.lower()].append(weight)
                
            term_avg = {term: sum(weights) / len(weights) 
                       for term, weights in term_avg_attn.items()}
            
            # Calculate relevance score
            term_relevance = {term: count * term_avg.get(term, 0) 
                             for term, count in term_counter.items()}
            
            # Get top terms by relevance
            top_terms = sorted(term_relevance.items(), key=lambda x: x[1], reverse=True)[:10]
            
            category_stats[category] = {
                'top_terms': top_terms,
                'total_occurrences': len(terms),
                'unique_terms': len(term_counter)
            }
        
        # Generate insights based on the analysis
        insights = self._generate_batch_insights(
            overall_bias_score, 
            descriptor_bias_score,
            pos_bias_score, 
            neg_bias_score,
            pos_descriptor_bias,
            neg_descriptor_bias,
            category_stats,
        )
        
        return {
            'overall_bias_score': overall_bias_score,
            'descriptor_bias_score': descriptor_bias_score,
            'positive_comments_bias_score': pos_bias_score,
            'negative_comments_bias_score': neg_bias_score,
            'positive_descriptor_bias': pos_descriptor_bias,
            'negative_descriptor_bias': neg_descriptor_bias,
            'comment_count': len(results),
            'positive_count': len(pos_results),
            'negative_count': len(neg_results),
            'category_stats': category_stats,
            'insights': insights
        }
        
    def _generate_batch_insights(self, overall_bias, descriptor_bias, pos_bias, neg_bias, 
                                pos_descriptor_bias, neg_descriptor_bias, category_stats):
        """Generate insights about gender bias patterns in the batch of comments"""
        insights = []
        
        # Overall bias pattern
        if abs(overall_bias) > 0.15:
            direction = "masculine-pattern language" if overall_bias > 0 else "feminine-pattern language"
            insights.append(f"Overall, these comments tend to use {direction}.")
            
        # Descriptor type patterns
        if abs(descriptor_bias) > 0.15:
            if descriptor_bias > 0:
                insights.append(
                    "These evaluations focus more on personality and entertainment value than competence, "
                    "a pattern typically seen more in evaluations of male professors."
                )
            else:
                insights.append(
                    "These evaluations focus more on competence and qualifications than personality, "
                    "a pattern typically seen more in evaluations of female professors."
                )
                
        # Compare positive vs negative comments
        if abs(pos_bias - neg_bias) > 0.2:
            if pos_bias > neg_bias:
                insights.append(
                    "Positive comments show stronger masculine-pattern language than negative comments, "
                    "suggesting potential gender bias in how praise is expressed."
                )
            else:
                insights.append(
                    "Negative comments show stronger masculine-pattern language than positive comments, "
                    "suggesting potential gender bias in how criticism is expressed."
                )
                
        # Compare descriptor patterns in positive vs negative comments
        if abs(pos_descriptor_bias - neg_descriptor_bias) > 0.2:
            if pos_descriptor_bias > neg_descriptor_bias:
                insights.append(
                    "Positive comments focus more on personality and entertainment qualities, "
                    "while negative comments focus more on competence and qualifications."
                )
            else:
                insights.append(
                    "Negative comments focus more on personality and entertainment qualities, "
                    "while positive comments focus more on competence and qualifications."
                )
        
        # Look at most common terms
        if (category_stats['personality_entertainment']['total_occurrences'] > 
            category_stats['competence']['total_occurrences'] * 2):
            insights.append(
                "These comments heavily emphasize personality and entertainment qualities, "
                "with much less focus on competence and qualifications."
            )
        elif (category_stats['competence']['total_occurrences'] > 
              category_stats['personality_entertainment']['total_occurrences'] * 2):
            insights.append(
                "These comments heavily emphasize competence and qualifications, "
                "with much less focus on personality and entertainment qualities."
            )
            
        return insights