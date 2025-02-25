#!/usr/bin/env python3
"""
Standalone Model Comparison Script
This script loads all four trained models (LSTM, CNN, RF, and LinearSVC) and 
compares their sentiment predictions on a set of sample comments.
"""

import torch
import numpy as np
import pandas as pd
from joblib import load
from tabulate import tabulate
import torch.nn as nn
import json
import sys
import os
# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.text_preprocessing import clean_text
import warnings
warnings.filterwarnings('ignore')

# Sample comments for testing
POSITIVE_COMMENTS = [
    "Mei is an amazing prof. I have had her for three courses now, I find that her teaching style is appropriate given the context. Shes brilliant and genuinely wants you to succeed. She is more than willing to speak and meet with you outside of class.",
    "She is caring for all her students and has a passion for the language. Lots of homework and interactive, but that's what we need to learn! Take her, you won't regret it.",
    "Dr. Kiang made stats digestible. Lectures use Excel, much more efficient than manual stats. Many extra credit opportunities in the form of participation in the class quizzes every lecture & HW is straightforward.",
    "Great teacher. Very clear about expectations and always willing to help outside of class. Tests are fair if you study the material. Highly recommend!",
    "Professor Kwon was an amazing teacher. She challenges and encourages you to meet your full potential and really cares for her students. She was great at communicating and fair with her grading."
]

NEGATIVE_COMMENTS = [
    "Dr. Lien was very hard to communicate with and gave in class quizzes every day. She was not approachable. For instance during many quizzes, she would say aloud to the class during the quiz time how easy she thought the quiz was.",
    "She's a slow grader, gets distracted easily. Gives too much busy work, takes 2-3 hours just to complete one chapter of terms and questions, questions aren't easy to find; you must read her notes and the book and try your best on the questions.",
    "She is the worst professor I've ever had. Blunt and uncaring, bad tech skills, judgemental in class and a poor teacher who's exams are very difficult and gives you very little time.",
    "This teacher is HORRIBLE, she didnt explain anything and treated us like we were in kidergarten, she made the class soo much harder than what it should have been. I got an A in her class and that was only b/c I skipped class and went to the math lab.",
    "Let's face it, he feels good when his students are failing. He is so happy to give C- instead of helping his students to do better. For 3/4 of the class to fail an exam, there must be something wrong with his teaching!"
]

# Get the absolute path to the ml_models_trained directory
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_models_trained')

MODEL_PATHS = {
    'lstm': {
        'model': os.path.join(MODEL_DIR, 'lstm_sentiment.pt'),
        'vocab': os.path.join(MODEL_DIR, 'vocab.json')
    },
    'cnn': {
        'model': os.path.join(MODEL_DIR, 'cnn_sentiment.pt'), 
        'vocab': os.path.join(MODEL_DIR, 'vocab.json')
    },
    'rf': {
        'model': os.path.join(MODEL_DIR, 'rf_model.joblib'),
        'vectorizer': os.path.join(MODEL_DIR, 'rf_tfidf_vectorizer_repaired.joblib')
    },
    'svc': {
        'model': os.path.join(MODEL_DIR, 'linearsvc_model.joblib'),
        'vectorizer': os.path.join(MODEL_DIR, 'svc_tfidf_vectorizer_repaired.joblib')
    }
}

# Model architectures
class CustomSentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, output_dim=1, dropout=0.3):
        super(CustomSentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def attention_net(self, lstm_output):
        attention_weights = self.attention(lstm_output)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, (h, c) = self.lstm(embedded)
        context = self.attention_net(lstm_out)
        out = self.dropout(self.relu(self.fc1(context)))
        out = self.sigmoid(self.fc2(out))
        return out

class SentimentCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes, output_dim=1, dropout=0.5):
        super(SentimentCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_dim)) 
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.unsqueeze(1)
        conved = [torch.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [torch.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = torch.cat(pooled, dim=1)
        dropped = self.dropout(cat)
        return self.sigmoid(self.fc(dropped))

class ModelComparison:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.vocab = None
        print(f"Using device: {self.device}")

    def load_models(self):
        print("\nLoading models...")
        
        # Load vocabulary for deep learning models
        print(f"Loading vocab from: {MODEL_PATHS['lstm']['vocab']}")
        with open(MODEL_PATHS['lstm']['vocab'], 'r') as f:
            self.vocab = json.load(f)
        
        # Load LSTM with embed_dim=128
        print(f"Loading LSTM from: {MODEL_PATHS['lstm']['model']}")
        lstm = CustomSentimentLSTM(
            vocab_size=len(self.vocab),
            embed_dim=128,
            hidden_dim=256,
            num_layers=2,
            dropout=0.5
        ).to(self.device)
        lstm.load_state_dict(torch.load(MODEL_PATHS['lstm']['model'], map_location=self.device))
        lstm.eval()
        self.models['lstm'] = lstm

        # Load CNN with embed_dim=300
        print(f"Loading CNN from: {MODEL_PATHS['cnn']['model']}")
        cnn = SentimentCNN(
            vocab_size=len(self.vocab),
            embed_dim=300,
            num_filters=100,
            filter_sizes=[3, 4, 5],
            dropout=0.5
        ).to(self.device)
        cnn.load_state_dict(torch.load(MODEL_PATHS['cnn']['model'], map_location=self.device))
        cnn.eval()
        self.models['cnn'] = cnn

        # Load Random Forest
        print(f"Loading Random Forest from: {MODEL_PATHS['rf']['model']}")
        print(f"Loading RF vectorizer from: {MODEL_PATHS['rf']['vectorizer']}")
        rf_model = load(MODEL_PATHS['rf']['model'])
        rf_vectorizer = load(MODEL_PATHS['rf']['vectorizer'])
        print("RF vectorizer loaded. Attributes:", dir(rf_vectorizer))
        self.models['rf'] = {
            'model': rf_model,
            'vectorizer': rf_vectorizer
        }

        # Load LinearSVC
        print(f"Loading LinearSVC from: {MODEL_PATHS['svc']['model']}")
        print(f"Loading SVC vectorizer from: {MODEL_PATHS['svc']['vectorizer']}")
        self.models['svc'] = {
            'model': load(MODEL_PATHS['svc']['model']),
            'vectorizer': load(MODEL_PATHS['svc']['vectorizer'])
        }

    def prepare_deep_learning_input(self, text, max_len=100):
        """Prepare input for deep learning models (LSTM and CNN)"""
        tokens = clean_text(text).split()
        indices = [self.vocab.get(t, 0) for t in tokens]
        if len(indices) > max_len:
            indices = indices[:max_len]
        else:
            indices += [1] * (max_len - len(indices))
        return torch.tensor([indices], dtype=torch.long).to(self.device)

    def predict_sentiment(self, text):
        """Get sentiment predictions from all models"""
        results = {}
        
        # Deep learning models (LSTM and CNN)
        input_tensor = self.prepare_deep_learning_input(text)
        
        with torch.no_grad():
            # LSTM prediction
            lstm_output = self.models['lstm'](input_tensor).squeeze().item()
            results['lstm'] = {
                'prediction': 'Positive' if lstm_output >= 0.5 else 'Negative',
                'confidence': lstm_output
            }
            
            # CNN prediction
            cnn_output = self.models['cnn'](input_tensor).squeeze().item()
            results['cnn'] = {
                'prediction': 'Positive' if cnn_output >= 0.5 else 'Negative',
                'confidence': cnn_output
            }
        
        # Traditional models
        cleaned_text = clean_text(text)
        
        # Random Forest prediction
        rf_features = self.models['rf']['vectorizer'].transform([cleaned_text])
        rf_pred = self.models['rf']['model'].predict(rf_features)[0]
        rf_proba = self.models['rf']['model'].predict_proba(rf_features)[0][1]
        results['rf'] = {
            'prediction': 'Positive' if rf_pred == 1 else 'Negative',
            'confidence': rf_proba
        }
        
        # LinearSVC prediction
        svc_features = self.models['svc']['vectorizer'].transform([cleaned_text])
        svc_pred = self.models['svc']['model'].predict(svc_features)[0]
        svc_decision = self.models['svc']['model'].decision_function(svc_features)[0]
        svc_confidence = 1 / (1 + np.exp(-svc_decision))  # Sigmoid to convert to probability
        results['svc'] = {
            'prediction': 'Positive' if svc_pred == 1 else 'Negative',
            'confidence': svc_confidence
        }
        
        return results

    def evaluate_comments(self, comments, expected_sentiment):
        """Evaluate a list of comments with known sentiment"""
        results = []
        correct_predictions = {'lstm': 0, 'cnn': 0, 'rf': 0, 'svc': 0}
        total = len(comments)
        
        for comment in comments:
            predictions = self.predict_sentiment(comment)
            result = {
                'comment': comment[:100] + '...' if len(comment) > 100 else comment,
                'expected': expected_sentiment
            }
            
            for model, pred in predictions.items():
                result[model] = f"{pred['prediction']} ({pred['confidence']:.2f})"
                if pred['prediction'].lower() == expected_sentiment.lower():
                    correct_predictions[model] += 1
            
            results.append(result)
        
        return results, {model: count/total for model, count in correct_predictions.items()}

def main():
    # Initialize and load models
    comparator = ModelComparison()
    comparator.load_models()
    
    print("\nEvaluating positive comments...")
    pos_results, pos_accuracy = comparator.evaluate_comments(POSITIVE_COMMENTS, "Positive")
    
    print("Evaluating negative comments...")
    neg_results, neg_accuracy = comparator.evaluate_comments(NEGATIVE_COMMENTS, "Negative")
    
    # Combine results
    all_results = pos_results + neg_results
    
    # Calculate overall accuracy
    total_comments = len(POSITIVE_COMMENTS) + len(NEGATIVE_COMMENTS)
    overall_accuracy = {
        model: (pos_accuracy[model] * len(POSITIVE_COMMENTS) + 
                neg_accuracy[model] * len(NEGATIVE_COMMENTS)) / total_comments
        for model in pos_accuracy.keys()
    }
    
    # Display results
    print("\nModel Comparison Results:")
    print(tabulate(pd.DataFrame(all_results), headers='keys', tablefmt='grid', showindex=False))
    
    print("\nAccuracy on Positive Comments:")
    for model, acc in pos_accuracy.items():
        print(f"{model.upper()}: {acc:.2%}")
    
    print("\nAccuracy on Negative Comments:")
    for model, acc in neg_accuracy.items():
        print(f"{model.upper()}: {acc:.2%}")
    
    print("\nOverall Accuracy:")
    for model, acc in overall_accuracy.items():
        print(f"{model.upper()}: {acc:.2%}")
    
    # Find best performing model
    best_model = max(overall_accuracy.items(), key=lambda x: x[1])
    print(f"\nBest performing model: {best_model[0].upper()} with {best_model[1]:.2%} accuracy")
    
    # Save results to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('model_comparison_results.csv', index=False)
    print("\nDetailed results saved to 'model_comparison_results.csv'")

if __name__ == "__main__":
    main()