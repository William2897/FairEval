#!/usr/bin/env python3
"""
Example script to train a custom LSTM sentiment classifier on a CSV dataset
with columns: 'processed_comment' and 'sentiment' (0=negative, 1=positive).
No pretrained transformers are used. We rely on basic embeddings + LSTM.

Usage:
  python lstm.py --csv_path dataset.csv --model_out custom_lstm_sentiment.pt --vocab_out vocab.json [--tuning_results tuning_results.json]
"""

import argparse
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# 1. Configuration
# ----------------------------
# Model configuration
MODEL_CONFIG = {
    'csv_path': 'dataset.csv',  # Path to CSV with 'text' and 'label' columns
    'model_out': 'lstm_sentiment.pt',  # Where to save the trained model
    'vocab_out': 'vocab.json',  # Where to save the built vocabulary
    'max_vocab': 20000,  # Max vocabulary size
    'max_len': 100,  # Max sequence length
    'embed_dim': 128,  # Embedding dimension
    'hidden_dim': 256,  # Hidden dimension in LSTM
    'num_layers': 2,  # Number of LSTM layers
    'batch_size': 64,  # Training batch size
    'dropout': 0.5,  # Dropout rate
    'epochs': 10,  # Number of training epochs
    'lr': 1e-3,  # Learning rate
    'train_split': 0.7,  # Proportion of data for training
    'val_split': 0.15,  # Proportion of data for validation
    'test_split': 0.15,  # Proportion of data for testing
    'results_out': 'LSTM_results_tuned.json'  # Where to save the evaluation results
}

# ----------------------------
# 2. Data Loading and Dataset
# ----------------------------
class SentimentDataset(Dataset):
    def __init__(self, df, word2index, max_len=100):
        # Change from processed_tokens to processed_comment and handle non-string values
        self.texts = [str(text) if pd.notna(text) else '' for text in df['processed_comment'].tolist()]
        self.labels = df['sentiment'].tolist()
        self.word2index = word2index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        txt = self.texts[idx]
        label = self.labels[idx]
        tokens = txt.split()

        # convert tokens to indices
        indices = [self.word2index.get(t, 0) for t in tokens]  # 0 = <unk>
        # pad or truncate
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        else:
            indices += [1] * (self.max_len - len(indices))  # 1 = <pad>

        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.float)


# ----------------------------
# 3. Build Vocabulary
# ----------------------------
def build_vocab(texts, max_vocab=20000):
    """
    Build a token frequency dictionary from preprocessed texts.
    """
    freq = {}
    for txt in texts:
        # Handle non-string values
        if not isinstance(txt, str):
            continue
        tokens = txt.split()
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1

    # sort by freq
    sorted_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    # pick top max_vocab
    sorted_items = sorted_items[:(max_vocab - 2)]  # minus 2 for <unk>, <pad>

    word2index = {}
    word2index["<unk>"] = 0
    word2index["<pad>"] = 1
    idx = 2
    for w, c in sorted_items:
        word2index[w] = idx
        idx += 1
    return word2index


# ----------------------------
# 4. LSTM Model Definition
# ----------------------------
class CustomSentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.5):
        super(CustomSentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.lstm = nn.LSTM(embed_dim, 
                           hidden_dim, 
                           num_layers=num_layers, 
                           bidirectional=True, 
                           dropout=dropout if num_layers > 1 else 0,
                           batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)  # Bidirectional, so hidden_dim * 2
        
        # Output layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, return_attention=False):
        # x shape: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(x))
        # embedded shape: [batch_size, seq_len, embed_dim]
        
        lstm_out, (h, c) = self.lstm(embedded)
        # lstm_out shape: [batch_size, seq_len, hidden_dim * 2]
        
        # Calculate attention weights
        attention_weights = self.attention(lstm_out).squeeze(2)
        # attention_weights shape: [batch_size, seq_len]
        
        # Apply softmax to get normalized weights
        attention_weights = torch.softmax(attention_weights, dim=1)
        # attention_weights shape: [batch_size, seq_len]
        
        # Apply attention to the LSTM outputs
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)
        # context shape: [batch_size, hidden_dim * 2]
        
        # Feed through dense layers
        out = self.dropout(self.relu(self.fc1(context)))
        out = self.sigmoid(self.fc2(out))
        
        if return_attention:
            return out, attention_weights
        return out

def train_one_epoch(model, loader, criterion, optimizer, scheduler, device='cpu', clip_grad=1.0):
    model.train()
    total_loss = 0
    preds = []
    trues = []

    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        total_loss += loss.item()
        pred = (outputs >= 0.5).float()
        preds.extend(pred.cpu().numpy())
        trues.extend(labels.cpu().numpy())

    # Step the scheduler
    scheduler.step()
    
    epoch_acc = accuracy_score(trues, preds)
    epoch_f1 = f1_score(trues, preds)
    return total_loss / len(loader), epoch_acc, epoch_f1

def eval_one_epoch(model, loader, criterion, device='cpu'):
    model.eval()
    total_loss = 0
    preds = []
    trues = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            pred = (outputs >= 0.5).float()
            preds.extend(pred.cpu().numpy())
            trues.extend(labels.cpu().numpy())

    epoch_acc = accuracy_score(trues, preds)
    epoch_f1 = f1_score(trues, preds)
    return total_loss / len(loader), epoch_acc, epoch_f1


def detailed_evaluation(y_true, y_pred, y_prob=None):
    """
    Compute comprehensive evaluation metrics using classification_report
    """
    # Get detailed classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    
    return metrics

def plot_confusion_matrix(cm, save_path=None):
    """Plot confusion matrix using seaborn"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def test_model(model, test_loader, criterion, device='cpu'):
    """Enhanced evaluation on test set"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            probs = outputs.cpu().numpy()
            preds = (outputs >= 0.5).float().cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate comprehensive metrics
    metrics = detailed_evaluation(all_labels, all_preds, all_probs)
    metrics['loss'] = total_loss / len(test_loader)
    
    # Print detailed results
    print("\nTest Results:")
    print(f"Loss: {metrics['loss']:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))
    
    return metrics

def save_results(results, filepath):
    """Save evaluation results to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)


# ----------------------------
# 5. Main Training Routine
# ----------------------------
def main():
    # parse arguments
    csv_path = MODEL_CONFIG['csv_path']
    model_out = MODEL_CONFIG['model_out']
    vocab_out = MODEL_CONFIG['vocab_out']
    results_out = MODEL_CONFIG['results_out']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    df = pd.read_csv(csv_path)
    
    # Data splits
    train_val_df, test_df = train_test_split(
        df, 
        test_size=MODEL_CONFIG['test_split'],
        random_state=42
    )
    
    train_size = MODEL_CONFIG['train_split'] / (MODEL_CONFIG['train_split'] + MODEL_CONFIG['val_split'])
    train_df, val_df = train_test_split(
        train_val_df,
        train_size=train_size,
        random_state=42
    )
    
    print(f"Data splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Build vocab from training data only
    print("Building vocabulary...")
    word2index = build_vocab(train_df['processed_comment'].tolist(), max_vocab=MODEL_CONFIG['max_vocab'])
    vocab_size = len(word2index)
    print(f"Vocab size = {vocab_size}")

    # Save vocab to JSON
    with open(vocab_out, 'w', encoding='utf-8') as f:
        json.dump(word2index, f, ensure_ascii=False)

    # Create datasets and dataloaders
    train_dataset = SentimentDataset(train_df, word2index, max_len=MODEL_CONFIG['max_len'])
    val_dataset = SentimentDataset(val_df, word2index, max_len=MODEL_CONFIG['max_len'])
    test_dataset = SentimentDataset(test_df, word2index, max_len=MODEL_CONFIG['max_len'])

    train_loader = DataLoader(train_dataset, batch_size=MODEL_CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=MODEL_CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=MODEL_CONFIG['batch_size'], shuffle=False)
    
    # Initialize model
    model = CustomSentimentLSTM(
        vocab_size, 
        MODEL_CONFIG['embed_dim'], 
        MODEL_CONFIG['hidden_dim'], 
        MODEL_CONFIG['num_layers'], 
        dropout=MODEL_CONFIG['dropout']
    )
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=MODEL_CONFIG['lr'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MODEL_CONFIG['epochs'],
        eta_min=1e-6
    )

    # Initialize results dictionary
    results = {
        'training': {
            'epochs': [],
            'best_val_f1': None
        },
        'test': {}
    }

    # Training loop
    best_val_f1 = 0.0
    for epoch in range(MODEL_CONFIG['epochs']):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, 
            device=device, clip_grad=1.0
        )
        val_loss, val_acc, val_f1 = eval_one_epoch(model, val_loader, criterion, device=device)

        # Store epoch results
        results['training']['epochs'].append({
            'epoch': epoch + 1,
            'train': {'loss': train_loss, 'accuracy': train_acc, 'f1': train_f1},
            'val': {'loss': val_loss, 'accuracy': val_acc, 'f1': val_f1}
        })

        print(f"Epoch [{epoch+1}/{MODEL_CONFIG['epochs']}] "
              f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f} "
              f"|| Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), model_out)
            print(f"  [*] Saved new best model to {model_out} (val_f1={val_f1:.4f})")
    
    results['training']['best_val_f1'] = best_val_f1

    print("Training complete!")
    
    # Testing phase
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(model_out))
    metrics = test_model(model, test_loader, criterion, device)
    
    # Plot and save confusion matrix
    plot_confusion_matrix(
        np.array(metrics['confusion_matrix']), 
        save_path=results_out.replace('.json', '_confusion_matrix.png')
    )
    
    # Store test results
    results['test'] = metrics
    
    # Save results
    save_results(results, results_out)
    print(f"\nResults saved to {results_out}")

if __name__ == "__main__":
    main()
