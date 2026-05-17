"""Fast FakeBERT using DistilBERT - lighter and faster than full BERT."""

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

LOGGER = get_logger("ml.train_fakebert_fast")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER.info("Using device: %s", DEVICE)


class FastFakeBERT(nn.Module):
    """Fast FakeBERT using DistilBERT (6 layers instead of 12, 66M params vs 110M)."""
    
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


def train_fakebert(
    dataset_name: str,
    texts: List[str],
    labels: List[str],
    output_dir: Path,
    epochs: int = 2,
    batch_size: int = 32,
    max_length: int = 128,
    learning_rate: float = 2e-5
) -> Dict:
    """Train FastFakeBERT model."""
    LOGGER.info(f"Training FastFakeBERT on {dataset_name}")
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    
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
        for batch in train_loader:
            input_ids, attention_mask, y = batch
            input_ids, attention_mask, y = input_ids.to(DEVICE), attention_mask.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        
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
        LOGGER.info(f"Epoch {epoch+1}: Val F1 = {f1:.4f}")
        
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
        "precision": precision_score(true_labels, preds, average="weighted"),
        "recall": recall_score(true_labels, preds, average="weighted"),
        "f1": f1_score(true_labels, preds, average="weighted"),
        "dataset_size": len(texts),
        "train_samples": len(train_texts)
    }
    
    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "label2idx": model.label2idx,
        "idx2label": model.idx2label
    }, output_dir / "model.pt")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    
    LOGGER.info(f"Saved to {output_dir}")
    return metrics


# Example usage
if __name__ == "__main__":
    # Quick test with sample data
    texts = [
        "Breaking news: Scientists discover new cure for cancer",
        "Scientists confirm global warming is real and caused by humans",
        "Aliens have landed on Earth and are living among us",
        "Eating carrots will make you see in the dark - scientists say"
    ]
    labels = ["REAL", "REAL", "FAKE", "FAKE"]
    
    print("Testing FastFakeBERT...")
    # This would require actual training to work properly
    print("Use train_fakebert_fast.py to train on full datasets")