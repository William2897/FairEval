import unittest
import torch
from ..ml_model_dev.lstm import CustomSentimentLSTM
from ..gender_bias_explainer import GenderBiasExplainer
from data_processing.gender_assignment import MALE_KEYWORDS, FEMALE_KEYWORDS

class TestLSTMAttention(unittest.TestCase):
    """Test suite for the LSTM model with attention mechanisms"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests"""
        # Define test parameters
        cls.vocab_size = 5000
        cls.embed_dim = 128
        cls.hidden_dim = 256
        cls.num_layers = 2
        
        # Create a simple vocabulary for testing
        cls.vocab = {"<PAD>": 1, "<UNK>": 0}
        test_words = ["professor", "great", "terrible", "helpful", "he", "she", 
                     "his", "her", "man", "woman", "male", "female", "class",
                     "lecture", "course", "test", "exam", "assignment"]
        for i, word in enumerate(test_words):
            cls.vocab[word] = i + 2  # Start after special tokens
        
        # Create test device
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        cls.model = CustomSentimentLSTM(
            vocab_size=cls.vocab_size,
            embed_dim=cls.embed_dim,
            hidden_dim=cls.hidden_dim,
            num_layers=cls.num_layers
        ).to(cls.device)
        
        # Set model to eval mode
        cls.model.eval()
        
        # Initialize gender bias explainer
        cls.explainer = GenderBiasExplainer(cls.model, cls.vocab, MALE_KEYWORDS, FEMALE_KEYWORDS)
        
        # Sample test comments with known gender term patterns
        cls.test_comments = [
            "The professor is great. He explains concepts well and his exams are fair.",
            "The professor is great. She explains concepts well and her exams are fair.",
            "This was an excellent class with no gender-specific language."
        ]
    
    def test_model_attention_output_shape(self):
        """Test that model returns attention weights with correct shape"""
        # Prepare sample input
        tokens = ["professor", "is", "great"]
        indices = [self.vocab.get(t, 0) for t in tokens]
        max_len = 10
        indices = indices + [1] * (max_len - len(indices))  # Pad
        
        # Convert to tensor
        input_tensor = torch.tensor([indices], dtype=torch.long).to(self.device)
        
        # Get prediction with attention
        with torch.no_grad():
            pred, attention = self.model(input_tensor, return_attention=True)
        
        # Check shapes
        self.assertEqual(pred.shape, torch.Size([1, 1]), "Prediction shape incorrect")
        self.assertEqual(attention.shape, torch.Size([1, max_len]), "Attention shape incorrect")
        
        # Check values
        self.assertTrue(0 <= pred.item() <= 1, "Prediction value out of range")
        self.assertTrue(torch.allclose(attention.sum(dim=1), torch.tensor([1.0]).to(self.device)), 
                        "Attention weights don't sum to 1")
    
    def test_gender_bias_detection(self):
        """Test gender bias detection in comments"""
        # Test male-biased comment
        male_explanation = self.explainer.explain_prediction(self.test_comments[0])
        self.assertTrue(male_explanation['gender_bias']['bias_score'] > 0, 
                       "Failed to detect male bias in comment")
        
        # Test female-biased comment
        female_explanation = self.explainer.explain_prediction(self.test_comments[1])
        self.assertTrue(female_explanation['gender_bias']['bias_score'] < 0, 
                       "Failed to detect female bias in comment")
        
        # Test neutral comment
        neutral_explanation = self.explainer.explain_prediction(self.test_comments[2])
        self.assertTrue(abs(neutral_explanation['gender_bias']['bias_score']) < 0.3, 
                       "Incorrectly detected gender bias in neutral comment")
    
    def test_discipline_context_integration(self):
        """Test integration with discipline context"""
        # Mock discipline data
        mock_discipline = "Computer Science"
        
        # Test with discipline context
        explanation = self.explainer.explain_prediction(self.test_comments[0], mock_discipline)
        
        # Should return discipline context if available in the database
        if 'discipline_context' in explanation:
            self.assertEqual(explanation['discipline_context']['discipline'], mock_discipline,
                           "Discipline context not properly applied")
    
    def test_attention_weight_distribution(self):
        """Test attention weight distribution when gender terms are present"""
        # Comment with male terms
        male_explanation = self.explainer.explain_prediction(self.test_comments[0])
        
        # Total attention on male terms should be positive
        self.assertTrue(male_explanation['gender_bias']['male_attention_total'] > 0,
                       "No attention assigned to male terms")
        
        # Comment with female terms
        female_explanation = self.explainer.explain_prediction(self.test_comments[1])
        
        # Total attention on female terms should be positive
        self.assertTrue(female_explanation['gender_bias']['female_attention_total'] > 0,
                       "No attention assigned to female terms")

if __name__ == "__main__":
    unittest.main()