import unittest
import torch
import torch.nn as nn
from ..ml_model_dev.lstm import CustomSentimentLSTM, AttentionLayer

class TestCustomSentimentLSTM(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.vocab_size = 1000
        self.embedding_dim = 100
        self.hidden_dim = 128
        self.output_dim = 2
        self.n_layers = 2
        self.dropout = 0.2
        
        self.model = CustomSentimentLSTM(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            n_layers=self.n_layers,
            dropout=self.dropout
        )

    def test_model_initialization(self):
        """Test model initialization and architecture"""
        # Test embedding layer
        self.assertIsInstance(self.model.embedding, nn.Embedding)
        self.assertEqual(self.model.embedding.num_embeddings, self.vocab_size)
        self.assertEqual(self.model.embedding.embedding_dim, self.embedding_dim)
        
        # Test LSTM layer
        self.assertIsInstance(self.model.lstm, nn.LSTM)
        self.assertEqual(self.model.lstm.hidden_size, self.hidden_dim)
        self.assertEqual(self.model.lstm.num_layers, self.n_layers)
        
        # Test attention layer
        self.assertIsInstance(self.model.attention, AttentionLayer)
        
        # Test dropout layer
        self.assertIsInstance(self.model.dropout, nn.Dropout)
        self.assertEqual(self.model.dropout.p, self.dropout)
        
        # Test output layer
        self.assertIsInstance(self.model.fc, nn.Linear)
        self.assertEqual(self.model.fc.in_features, self.hidden_dim)
        self.assertEqual(self.model.fc.out_features, self.output_dim)

    def test_attention_layer(self):
        """Test attention layer functionality"""
        attention = AttentionLayer(self.hidden_dim)
        
        # Create sample hidden states and final hidden state
        hidden_states = torch.randn(10, 1, self.hidden_dim)  # sequence_len, batch_size, hidden_dim
        final_hidden = torch.randn(1, 1, self.hidden_dim)  # n_layers, batch_size, hidden_dim
        
        # Test attention computation
        attention_weights, context = attention(hidden_states, final_hidden[-1])
        
        # Check output shapes
        self.assertEqual(attention_weights.shape, (1, 10))  # batch_size, sequence_len
        self.assertEqual(context.shape, (1, self.hidden_dim))  # batch_size, hidden_dim
        
        # Check attention weights sum to 1
        self.assertTrue(torch.allclose(attention_weights.sum(dim=1), 
                                     torch.tensor([1.0]), atol=1e-6))

    def test_forward_pass(self):
        """Test model forward pass"""
        batch_size = 2
        sequence_length = 10
        
        # Create sample input
        text = torch.randint(0, self.vocab_size, (sequence_length, batch_size))
        
        # Forward pass
        predictions, attention, hidden = self.model(text)
        
        # Check output shapes
        self.assertEqual(predictions.shape, (batch_size, self.output_dim))
        self.assertEqual(attention.shape, (batch_size, sequence_length))
        self.assertEqual(hidden[0].shape, (self.n_layers, batch_size, self.hidden_dim))
        self.assertEqual(hidden[1].shape, (self.n_layers, batch_size, self.hidden_dim))
        
        # Check probability distribution
        self.assertTrue(torch.allclose(torch.sum(torch.softmax(predictions, dim=1)), 
                                     torch.tensor([2.0])))  # Sum for both samples

    def test_input_validation(self):
        """Test input validation and error handling"""
        # Test empty input
        with self.assertRaises(RuntimeError):
            self.model(torch.tensor([]))
        
        # Test input with wrong dimensions
        with self.assertRaises(RuntimeError):
            self.model(torch.randn(5, 3, 2))  # Wrong dimensions
        
        # Test input with values out of vocabulary range
        text = torch.tensor([[self.vocab_size + 1]])  # Out of vocab range
        with self.assertRaises(RuntimeError):
            self.model(text)

    def test_training_mode(self):
        """Test model behavior in training vs evaluation modes"""
        self.model.train()
        
        # Create sample input
        text = torch.randint(0, self.vocab_size, (5, 1))
        
        # Get outputs in training mode
        train_pred, train_attn, _ = self.model(text)
        
        # Switch to evaluation mode
        self.model.eval()
        
        # Get outputs in evaluation mode
        with torch.no_grad():
            eval_pred, eval_attn, _ = self.model(text)
        
        # Predictions should be different due to dropout
        self.assertFalse(torch.allclose(train_pred, eval_pred))

    def test_attention_weights_distribution(self):
        """Test attention weights distribution properties"""
        sequence_length = 15
        text = torch.randint(0, self.vocab_size, (sequence_length, 1))
        
        # Get model outputs
        _, attention, _ = self.model(text)
        
        # Check attention properties
        self.assertTrue(torch.all(attention >= 0))  # Non-negative weights
        self.assertTrue(torch.allclose(attention.sum(dim=1), 
                                     torch.tensor([1.0]), atol=1e-6))  # Sum to 1
        self.assertEqual(attention.shape, (1, sequence_length))  # Correct shape

    def test_embedding_gradient_flow(self):
        """Test gradient flow through embedding layer"""
        self.model.train()
        
        # Create sample input and target
        text = torch.randint(0, self.vocab_size, (5, 1))
        target = torch.tensor([1])  # Binary classification target
        
        # Forward pass
        pred, _, _ = self.model(text)
        
        # Compute loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, target)
        
        # Backward pass
        loss.backward()
        
        # Check if gradients are computed
        self.assertIsNotNone(self.model.embedding.weight.grad)
        self.assertTrue(torch.any(self.model.embedding.weight.grad != 0))

    def test_save_load_model(self):
        """Test model state saving and loading"""
        # Get initial predictions
        text = torch.randint(0, self.vocab_size, (5, 1))
        initial_pred, _, _ = self.model(text)
        
        # Save model state
        state_dict = self.model.state_dict()
        
        # Create new model instance
        new_model = CustomSentimentLSTM(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            n_layers=self.n_layers,
            dropout=self.dropout
        )
        
        # Load state into new model
        new_model.load_state_dict(state_dict)
        
        # Get predictions from loaded model
        with torch.no_grad():
            loaded_pred, _, _ = new_model(text)
        
        # Check if predictions are identical
        self.assertTrue(torch.allclose(initial_pred, loaded_pred))

    def test_bidirectional_hidden_states(self):
        """Test bidirectional LSTM hidden states"""
        # Create model with bidirectional LSTM
        bidir_model = CustomSentimentLSTM(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
            bidirectional=True
        )
        
        # Test input
        text = torch.randint(0, self.vocab_size, (5, 1))
        
        # Get outputs
        pred, attn, (h_n, c_n) = bidir_model(text)
        
        # Check hidden states shape (doubled for bidirectional)
        expected_hidden_shape = (self.n_layers * 2, 1, self.hidden_dim)
        self.assertEqual(h_n.shape, expected_hidden_shape)
        self.assertEqual(c_n.shape, expected_hidden_shape)