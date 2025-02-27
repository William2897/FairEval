import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import os
import json

from data_processing.gender_assignment import MALE_KEYWORDS, FEMALE_KEYWORDS


class GenderBiasExplainer:
    def __init__(self, model, vocab, male_terms=None, female_terms=None):
        """
        Initialize an explainer for gender bias in LSTM attention
        
        Args:
            model: trained LSTM model with attention
            vocab: vocabulary dictionary (word to index)
            male_terms: list of male-associated terms (defaults to MALE_KEYWORDS)
            female_terms: list of female-associated terms (defaults to FEMALE_KEYWORDS)
        """
        self.model = model
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.male_terms = set(male_terms or MALE_KEYWORDS)
        self.female_terms = set(female_terms or FEMALE_KEYWORDS)
        self.device = next(model.parameters()).device
        
        # Load discipline gender gaps from the database if available
        self.discipline_gender_gaps = self._load_discipline_gender_gaps()
        
    def _load_discipline_gender_gaps(self):
        """Load pre-calculated gender gaps in ratings by discipline"""
        try:
            # Try to import Django utilities
            from django.conf import settings
            import django
            os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
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
        
        # Analyze gender bias in attention weights
        gender_bias_data = self._analyze_gender_attention(tokens, attn_weights)
        
        explanation = {
            'prediction': prediction,
            'confidence': confidence,
            'tokens': tokens,
            'attention': attn_weights.tolist(),
            'gender_bias': gender_bias_data
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
        
    def _analyze_gender_attention(self, tokens, attention_weights):
        """Analyze how attention distributes across gender terms"""
        gender_markers = {'male': [], 'female': [], 'neutral': []}
        
        for i, token in enumerate(tokens):
            if token.lower() in self.male_terms:
                gender_markers['male'].append((i, token, attention_weights[i]))
            elif token.lower() in self.female_terms:
                gender_markers['female'].append((i, token, attention_weights[i]))
            else:
                gender_markers['neutral'].append((i, token, attention_weights[i]))
                
        # Calculate aggregated attention by gender category
        male_attention = sum(w for _, _, w in gender_markers['male']) if gender_markers['male'] else 0
        female_attention = sum(w for _, _, w in gender_markers['female']) if gender_markers['female'] else 0
        neutral_attention = sum(w for _, _, w in gender_markers['neutral']) if gender_markers['neutral'] else 0
        
        total_attention = male_attention + female_attention + neutral_attention
        if total_attention == 0:
            total_attention = 1  # Avoid division by zero
            
        male_attention_pct = (male_attention / total_attention) * 100
        female_attention_pct = (female_attention / total_attention) * 100
        neutral_attention_pct = (neutral_attention / total_attention) * 100
        
        # Calculate bias score (-1 to 1, positive means male bias)
        total_gendered_attention = male_attention + female_attention
        bias_score = 0
        if total_gendered_attention > 0:
            bias_score = (male_attention - female_attention) / total_gendered_attention
        
        # Find top attended terms by gender
        top_male_terms = sorted(gender_markers['male'], key=lambda x: x[2], reverse=True)[:5]
        top_female_terms = sorted(gender_markers['female'], key=lambda x: x[2], reverse=True)[:5]
        top_neutral_terms = sorted(gender_markers['neutral'], key=lambda x: x[2], reverse=True)[:5]
        
        return {
            'male_terms': gender_markers['male'],
            'female_terms': gender_markers['female'],
            'neutral_terms': gender_markers['neutral'],
            'male_attention_total': male_attention,
            'female_attention_total': female_attention,
            'neutral_attention_total': neutral_attention,
            'male_attention_pct': male_attention_pct,
            'female_attention_pct': female_attention_pct,
            'neutral_attention_pct': neutral_attention_pct,
            'bias_score': bias_score,
            'top_male_terms': top_male_terms,
            'top_female_terms': top_female_terms,
            'top_neutral_terms': top_neutral_terms
        }
    
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
        
        # Color-coding for gender terms
        colors = []
        for token in tokens:
            if token.lower() in self.male_terms:
                colors.append('blue')
            elif token.lower() in self.female_terms:
                colors.append('red')
            else:
                colors.append('gray')
        
        # Plot attention weights
        plt.bar(range(len(tokens)), attention, color=colors)
        plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
        plt.xlabel('Tokens')
        plt.ylabel('Attention Weight')
        
        title = f"Prediction: {explanation['prediction']} (Confidence: {explanation['confidence']:.2f})\n"
        title += f"Gender Bias Score: {explanation['gender_bias']['bias_score']:.2f} "
        title += ("(Male Biased)" if explanation['gender_bias']['bias_score'] > 0.2 else 
                 "(Female Biased)" if explanation['gender_bias']['bias_score'] < -0.2 else "(Neutral)")
        
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
            Patch(facecolor='blue', label='Male Terms'),
            Patch(facecolor='red', label='Female Terms'),
            Patch(facecolor='gray', label='Neutral Terms')
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
        pos_results = [r for r in results if r['prediction'] == 'Positive']
        neg_results = [r for r in results if r['prediction'] == 'Negative']
        
        pos_bias_score = np.mean([r['gender_bias']['bias_score'] for r in pos_results]) if pos_results else 0
        neg_bias_score = np.mean([r['gender_bias']['bias_score'] for r in neg_results]) if neg_results else 0
        
        # Get most attended gendered terms
        all_male_terms = []
        all_female_terms = []
        
        for r in results:
            male_terms = [(term, weight) for _, term, weight in r['gender_bias']['male_terms']]
            female_terms = [(term, weight) for _, term, weight in r['gender_bias']['female_terms']]
            all_male_terms.extend(male_terms)
            all_female_terms.extend(female_terms)
            
        # Count occurrences of gendered terms
        male_term_counter = Counter([term.lower() for term, _ in all_male_terms])
        female_term_counter = Counter([term.lower() for term, _ in all_female_terms])
        
        # Calculate average attention per term
        male_term_avg_attn = defaultdict(list)
        female_term_avg_attn = defaultdict(list)
        
        for term, weight in all_male_terms:
            male_term_avg_attn[term.lower()].append(weight)
        for term, weight in all_female_terms:
            female_term_avg_attn[term.lower()].append(weight)
            
        male_term_avg = {term: sum(weights) / len(weights) 
                         for term, weights in male_term_avg_attn.items()}
        female_term_avg = {term: sum(weights) / len(weights) 
                           for term, weights in female_term_avg_attn.items()}
        
        # Sort by frequency * avg_attention for relevance
        male_term_relevance = {term: count * male_term_avg.get(term, 0) 
                              for term, count in male_term_counter.items()}
        female_term_relevance = {term: count * female_term_avg.get(term, 0) 
                                for term, count in female_term_counter.items()}
        
        top_male_terms = sorted(male_term_relevance.items(), key=lambda x: x[1], reverse=True)[:10]
        top_female_terms = sorted(female_term_relevance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'overall_bias_score': overall_bias_score,
            'positive_comments_bias_score': pos_bias_score,
            'negative_comments_bias_score': neg_bias_score,
            'comment_count': len(results),
            'positive_count': len(pos_results),
            'negative_count': len(neg_results),
            'top_male_terms': top_male_terms,
            'top_female_terms': top_female_terms
        }