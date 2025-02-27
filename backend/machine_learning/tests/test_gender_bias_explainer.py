import unittest
from unittest.mock import patch, MagicMock
import torch
import numpy as np
from ..gender_bias_explainer import GenderBiasExplainer
from ..ml_model_dev.lstm import CustomSentimentLSTM

class TestGenderBiasExplainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests"""
        cls.model = MagicMock(spec=CustomSentimentLSTM)
        cls.explainer = GenderBiasExplainer(cls.model)
        
        # Mock the model's forward pass
        cls.model.forward.return_value = (
            torch.tensor([0.8]),  # prediction
            torch.tensor([[0.3, 0.7]]),  # attention weights
            None  # hidden state (not used in tests)
        )

    def test_explain_prediction_basic(self):
        """Test basic prediction explanation"""
        comment = "The professor explains concepts clearly."
        result = self.explainer.explain_prediction(comment)
        
        self.assertIsNotNone(result)
        self.assertTrue('prediction' in result)
        self.assertTrue('confidence' in result)
        self.assertTrue('gender_bias' in result)
        self.assertTrue(isinstance(result['confidence'], float))

    def test_explain_prediction_with_male_bias(self):
        """Test explanation with male-biased text"""
        comment = "He is an excellent teacher and his explanations are clear."
        result = self.explainer.explain_prediction(comment)
        
        self.assertTrue('gender_bias' in result)
        bias = result['gender_bias']
        self.assertTrue('bias_score' in bias)
        self.assertTrue('male_attention_pct' in bias)
        self.assertTrue('male_terms' in bias)
        self.assertTrue(bias['male_attention_pct'] > 0)

    def test_explain_prediction_with_female_bias(self):
        """Test explanation with female-biased text"""
        comment = "She provides detailed feedback and her lectures are engaging."
        result = self.explainer.explain_prediction(comment)
        
        self.assertTrue('gender_bias' in result)
        bias = result['gender_bias']
        self.assertTrue('bias_score' in bias)
        self.assertTrue('female_attention_pct' in bias)
        self.assertTrue('female_terms' in bias)
        self.assertTrue(bias['female_attention_pct'] > 0)

    def test_explain_prediction_no_gender_terms(self):
        """Test explanation with gender-neutral text"""
        comment = "The class was challenging but interesting."
        result = self.explainer.explain_prediction(comment)
        
        self.assertTrue('gender_bias' in result)
        bias = result['gender_bias']
        self.assertEqual(bias['bias_score'], 0)
        self.assertEqual(bias['male_attention_pct'], 0)
        self.assertEqual(bias['female_attention_pct'], 0)

    def test_analyze_comments_batch(self):
        """Test batch analysis of comments"""
        comments = [
            "He explains concepts clearly.",
            "She is very organized.",
            "The course was interesting."
        ]
        result = self.explainer.analyze_comments_batch(comments)
        
        self.assertIsNotNone(result)
        self.assertTrue('overall_bias_score' in result)
        self.assertTrue('positive_comments_bias_score' in result)
        self.assertTrue('negative_comments_bias_score' in result)
        self.assertTrue('comment_count' in result)

    @patch('torch.load')
    def test_load_model_with_attention(self, mock_torch_load):
        """Test loading model with attention weights"""
        mock_state_dict = {
            'embedding.weight': torch.randn(100, 50),
            'lstm.weight_ih_l0': torch.randn(400, 50),
            'lstm.weight_hh_l0': torch.randn(400, 100),
            'attention.weight': torch.randn(100, 100),
            'fc.weight': torch.randn(2, 100)
        }
        mock_torch_load.return_value = mock_state_dict
        
        model = self.explainer.load_model('dummy_path')
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'attention'))

    def test_get_attention_weights(self):
        """Test attention weight extraction"""
        text = "The professor is helpful"
        words = text.split()
        attention = torch.tensor([[0.2, 0.3, 0.4, 0.1]])
        
        weights = self.explainer.get_attention_weights(words, attention)
        
        self.assertEqual(len(weights), len(words))
        self.assertTrue(all(isinstance(w, float) for w in weights))
        self.assertAlmostEqual(sum(weights), 1.0, places=5)

    def test_invalid_input_handling(self):
        """Test handling of invalid inputs"""
        # Test empty comment
        result_empty = self.explainer.explain_prediction("")
        self.assertIsNotNone(result_empty)
        self.assertEqual(result_empty['gender_bias']['bias_score'], 0)
        
        # Test None comment
        result_none = self.explainer.explain_prediction(None)
        self.assertIsNotNone(result_none)
        self.assertEqual(result_none['gender_bias']['bias_score'], 0)
        
        # Test very long comment
        long_comment = "word " * 1000
        result_long = self.explainer.explain_prediction(long_comment)
        self.assertIsNotNone(result_long)
        
        # Test special characters
        special_comment = "!@#$%^&*()"
        result_special = self.explainer.explain_prediction(special_comment)
        self.assertIsNotNone(result_special)

    def test_bias_detection_thresholds(self):
        """Test different bias detection thresholds"""
        comment = "He is a good teacher but she is better"
        
        # Test with different thresholds
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        for threshold in thresholds:
            self.explainer.bias_threshold = threshold
            result = self.explainer.explain_prediction(comment)
            self.assertTrue('bias_score' in result['gender_bias'])
            
        # Reset threshold to default
        self.explainer.bias_threshold = 0.5

    def test_context_integration(self):
        """Test integration with discipline context"""
        comment = "The professor explains well"
        discipline = "Computer Science"
        
        result = self.explainer.explain_prediction(comment, discipline)
        
        self.assertIsNotNone(result)
        self.assertTrue('discipline_context' in result)
        if 'discipline_context' in result:
            self.assertEqual(result['discipline_context']['discipline'], discipline)

    def test_comparative_analysis(self):
        """Test comparative analysis of different comments"""
        comments = {
            'positive_male': "He is an excellent teacher",
            'negative_male': "His lectures are confusing",
            'positive_female': "She explains concepts clearly",
            'negative_female': "Her grading is inconsistent"
        }
        
        results = {k: self.explainer.explain_prediction(v) for k, v in comments.items()}
        
        # Verify bias detection consistency
        self.assertTrue(results['positive_male']['gender_bias']['male_attention_pct'] > 0)
        self.assertTrue(results['positive_female']['gender_bias']['female_attention_pct'] > 0)
        
        # Compare sentiment consistency
        pos_sentiments = [results['positive_male']['confidence'], 
                         results['positive_female']['confidence']]
        neg_sentiments = [results['negative_male']['confidence'], 
                         results['negative_female']['confidence']]
        
        self.assertTrue(all(s > 0.5 for s in pos_sentiments))
        self.assertTrue(all(s < 0.5 for s in neg_sentiments))

    def test_batch_processing_edge_cases(self):
        """Test batch processing with various edge cases"""
        comments = [
            "",  # Empty comment
            "Normal comment",
            None,  # None comment
            "!@#$%",  # Special characters
            "a" * 1000,  # Very long comment
            "Short"
        ]
        
        result = self.explainer.analyze_comments_batch(comments)
        
        self.assertIsNotNone(result)
        self.assertTrue('comment_count' in result)
        self.assertEqual(result['comment_count'], len([c for c in comments if c]))  # Count non-empty comments