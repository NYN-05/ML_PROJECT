"""Train FakeBERT models on all datasets - OPTIMIZED VERSION."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from logging_utils import get_logger

LOGGER = get_logger("ml.train_fakebert")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER.info("Using device: %s", DEVICE)


# Dataset paths - use processed combined data
TRAIN_DATA_PATH = Path("../dataset/processed/train_dataset.csv")
TEST_DATA_PATH = Path("../dataset/processed/test_dataset.csv")


class FastFakeBERT(nn.Module):
    """Fast FakeBERT using DistilBERT - lighter and faster than full BERT."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", dropout: float = 0.3):
        super(FastFakeBERT, self).__init__()
        
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )
        
        self.label2idx = {"FAKE": 0, "REAL": 1}
        self.idx2label = {0: "FAKE", 1: "REAL"}
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token
        cls_output = outputs.last_hidden_state[:, 0, :]
        dropped = self.dropout(cls_output)
        return self.classifier(dropped)


def load_and_prepare_data(path: Path, max_samples: int = 5000) -> Optional[tuple]:
    """Load dataset and prepare for training."""
    try:
        df = pd.read_csv(path)
        
        # Find columns
        text_col = None
        label_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if text_col is None and any(x in col_lower for x in ["text", "article", "content", "body", "title"]):
                text_col = col
            if label_col is None and col_lower in ["label", "class", "type"]:
                label_col = col
        
        if text_col is None:
            text_col = df.columns[0]
        if label_col is None:
            label_col = df.columns[-1]
        
        # Get data
        texts = df[text_col].astype(str).tolist()[:max_samples]
        labels = df[label_col].astype(str).str.upper().tolist()[:max_samples]
        
        # Normalize labels
        labels = ["REAL" if any(x in l for x in ["1", "TRUE", "REAL"]) else "FAKE" for l in labels]
        
        # Filter
        valid = [(t, l) for t, l in zip(texts, labels) if len(str(t).strip()) > 10]
        if not valid:
            return None
            
        texts = [v[0] for v in valid]
        labels = [v[1] for v in valid]
        
        return texts, labels
        
    except Exception as e:
        LOGGER.error(f"Error loading {path}: {e}")
        return None


def train_fakebert_model(
    dataset_name: str,
    texts: List[str],
    labels: List[str],
    output_dir: Path,
    epochs: int = 2,
    batch_size: int = 32,
    max_length: int = 128,
    learning_rate: float = 2e-5
) -> Dict:
    """Train FastFakeBERT model on a dataset."""
    LOGGER.info(f"Training FastFakeBERT on {dataset_name} with {len(texts)} samples")
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    
    LOGGER.info(f"Train: {len(train_texts)}, Val: {len(val_texts)}")
    
    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Tokenize
    train_enc = tokenizer(train_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    val_enc = tokenizer(val_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    
    train_dataset = TensorDataset(
        torch.tensor(train_enc["input_ids"]),
        torch.tensor(train_enc["attention_mask"]),
        torch.tensor([{"FAKE": 0, "REAL": 1}[l] for l in train_labels])
    )
    
    val_dataset = TensorDataset(
        torch.tensor(val_enc["input_ids"]),
        torch.tensor(val_enc["attention_mask"]),
        torch.tensor([{"FAKE": 0, "REAL": 1}[l] for l in val_labels])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Model
    model = FastFakeBERT().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    best_f1 = 0
    best_state = None
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            input_ids, attention_mask, y = batch
            input_ids, attention_mask, y = input_ids.to(DEVICE), attention_mask.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validate
        model.eval()
        preds, true_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, y = batch
                input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
                
                logits = model(input_ids, attention_mask)
                preds.extend(logits.argmax(dim=1).cpu().numpy())
                true_labels.extend(y.numpy())
        
        f1 = f1_score(true_labels, preds, average="weighted")
        acc = accuracy_score(true_labels, preds)
        
        LOGGER.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f}, Val Acc: {acc:.4f}, Val F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    # Load best
    if best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    
    # Final eval
    model.eval()
    preds, true_labels = [], []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, y = batch
            input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
            logits = model(input_ids, attention_mask)
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            true_labels.extend(y.numpy())
    
    metrics = {
        "dataset_name": dataset_name,
        "accuracy": accuracy_score(true_labels, preds),
        "precision": precision_score(true_labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(true_labels, preds, average="weighted", zero_division=0),
        "f1": f1_score(true_labels, preds, average="weighted", zero_division=0),
        "dataset_size": len(texts),
        "train_samples": len(train_texts),
        "epochs": epochs
    }
    
    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "label2idx": model.label2idx,
        "idx2label": model.idx2label,
        "max_length": max_length,
        "metrics": metrics
    }, output_dir / "model.pt")
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))
    
    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    LOGGER.info(f"Saved {dataset_name} to {output_dir}")
    return metrics


def compute_model_weights(metrics_list: List[Dict]) -> Dict[str, float]:
    """Compute weights based on multiple metrics."""
    weights = {}
    
    for m in metrics_list:
        size_normalized = np.log1p(m["dataset_size"]) / 20
        
        score = (
            0.3 * m["f1"] +
            0.2 * m["accuracy"] +
            0.2 * m["precision"] +
            0.2 * m["recall"] +
            0.1 * size_normalized
        )
        
        weights[m["dataset_name"]] = score
    
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    
    return weights


def main():
    """Main training function."""
    artifacts_dir = Path("artifacts/models_fakebert")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Load combined training data
    if not TRAIN_DATA_PATH.exists():
        LOGGER.error(f"Training data not found: {TRAIN_DATA_PATH}")
        return
    
    LOGGER.info("Loading combined training data...")
    df = pd.read_csv(TRAIN_DATA_PATH)
    LOGGER.info(f"Loaded {len(df)} samples")
    
    # Prepare data
    texts = df["text"].astype(str).tolist()[:15000]  # Use 15K samples for faster training
    labels = df["label"].astype(str).str.upper().tolist()[:15000]
    
    # Filter valid
    valid = [(t, l) for t, l in zip(texts, labels) if len(str(t).strip()) > 10]
    texts = [v[0] for v in valid]
    labels = ["REAL" if v[1] == "REAL" else "FAKE" for v in valid]
    
    LOGGER.info(f"Training on {len(texts)} samples")
    
    # Train a single FakeBERT model on all data
    metrics = train_fakebert_model(
        dataset_name="FakeBERT_Ensemble",
        texts=texts,
        labels=labels,
        output_dir=artifacts_dir,
        epochs=2,
        batch_size=32,
        max_length=128,
        learning_rate=2e-5
    )
    
    # Save leaderboard
    leaderboard = {
        "models": [metrics],
        "weights": {"FakeBERT_Ensemble": 1.0},
        "best_model": "FakeBERT_Ensemble"
    }
    
    with open(artifacts_dir / "leaderboard.json", "w") as f:
        json.dump(leaderboard, f, indent=2)
    
    print("\n" + "="*60)
    print("FAKEBERT TRAINING COMPLETE")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print("\nModel saved to:", artifacts_dir)
    print("="*60)


if __name__ == "__main__":
    main()