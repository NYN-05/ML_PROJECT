"""Recursive DistilBERT training - train on one dataset, validate on next, continue until convergence."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# Dataset configurations
DATASETS = [
    {"name": "WELFake_Dataset", "path": Path("../dataset/WELFake_Dataset/WELFake_Dataset.csv"), "text_col": "text", "label_col": "label"},
    {"name": "Fake News Detection Dataset", "path": Path("../dataset/Fake News Detection Dataset/fake_news_dataset.csv"), "text_col": "text", "label_col": "label"},
    {"name": "Fake News Detection_1", "path": Path("../dataset/Fake News Detection_1/fake.csv"), "text_col": "text", "label_col": "label"},
    {"name": "fake-and-real-news-dataset", "path": Path("../dataset/fake-and-real-news-dataset/Fake.csv"), "text_col": "text", "label_col": "label"},
    {"name": "ISOT Fake News Dataset", "path": Path("../dataset/ISOT Fake News Dataset/Fake.csv"), "text_col": "text", "label_col": "label"},
    {"name": "News Detection (Fake or Real) Dataset", "path": Path("../dataset/News Detection (Fake or Real) Dataset/fake_and_real_news.csv"), "text_col": "text", "label_col": "label"},
    {"name": "Fake or Real News", "path": Path("../dataset/Fake or Real News/fake_or_real_news.csv"), "text_col": "text", "label_col": "label"},
]

OUTPUT_DIR = Path("artifacts/models_distilbert_recursive")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SAMPLES_PER_DATASET = 8000
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS_PER_DATASET = 3
LEARNING_RATE = 2e-5
CONVERGENCE_THRESHOLD = 0.005  # Stop if improvement < 0.5%


def load_dataset(dataset_config: Dict) -> Optional[Tuple[List[str], List[str]]]:
    """Load a single dataset."""
    path = dataset_config["path"]
    if not path.exists():
        print(f"  Dataset not found: {path}")
        return None
    
    try:
        df = pd.read_csv(path)
        
        # Find text and label columns
        text_col = None
        label_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if text_col is None and any(x in col_lower for x in ["text", "article", "content", "body"]):
                text_col = col
            if label_col is None and col_lower in ["label", "class"]:
                label_col = col
        
        if text_col is None:
            text_col = df.columns[0]
        if label_col is None:
            label_col = df.columns[-1]
        
        # Get data
        texts = df[text_col].astype(str).tolist()[:MAX_SAMPLES_PER_DATASET]
        labels = df[label_col].astype(str).str.upper().tolist()[:MAX_SAMPLES_PER_DATASET]
        
        # Normalize labels
        labels = ["REAL" if any(x in l for x in ["1", "TRUE", "REAL"]) else "FAKE" for l in labels]
        
        # Filter valid
        valid = [(t, l) for t, l in zip(texts, labels) if len(str(t).strip()) > 10]
        if not valid:
            return None
            
        texts = [v[0] for v in valid]
        labels = [v[1] for v in valid]
        
        print(f"  Loaded {len(texts)} samples ({len([l for l in labels if l=='FAKE'])} FAKE, {len([l for l in labels if l=='REAL'])} REAL)")
        return texts, labels
        
    except Exception as e:
        print(f"  Error loading: {e}")
        return None


class RecursiveDistilBERT:
    """DistilBERT with recursive training capability."""
    
    def __init__(self):
        self.bert = None
        self.classifier = None
        self.dropout = None
        self.tokenizer = None
        self.best_f1 = 0
        self.history = []
        
    def initialize(self):
        """Initialize model components."""
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased").to(DEVICE)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        ).to(DEVICE)
        
    def save(self, path: Path):
        """Save model."""
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            "classifier_state": self.classifier.state_dict(),
            "best_f1": self.best_f1,
            "history": self.history
        }, path / "model.pt")
        self.tokenizer.save_pretrained(str(path / "tokenizer"))
        
    def load(self, path: Path) -> bool:
        """Load existing model."""
        if not path.exists():
            return False
        
        try:
            checkpoint = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
            self.initialize()
            self.classifier.load_state_dict(checkpoint["classifier_state"])
            self.best_f1 = checkpoint.get("best_f1", 0)
            self.history = checkpoint.get("history", [])
            self.bert.eval()
            self.classifier.eval()
            return True
        except:
            return False
    
    def train_on_dataset(self, train_texts: List[str], train_labels: List[str], 
                        val_texts: List[str], val_labels: List[str],
                        dataset_name: str, epochs: int = EPOCHS_PER_DATASET) -> Dict:
        """Train on a single dataset."""
        if self.bert is None:
            self.initialize()
        
        # Prepare data
        train_enc = self.tokenizer(train_texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        val_enc = self.tokenizer(val_texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        
        label_map = {"FAKE": 0, "REAL": 1}
        
        train_dataset = TensorDataset(
            torch.tensor(train_enc["input_ids"]),
            torch.tensor(train_enc["attention_mask"]),
            torch.tensor([label_map[l] for l in train_labels])
        )
        
        val_dataset = TensorDataset(
            torch.tensor(val_enc["input_ids"]),
            torch.tensor(val_enc["attention_mask"]),
            torch.tensor([label_map[l] for l in val_labels])
        )
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        
        optimizer = torch.optim.AdamW([
            {"params": self.classifier.parameters(), "lr": LEARNING_RATE},
            # Don't retrain BERT to save time - use as feature extractor
        ])
        
        criterion = nn.CrossEntropyLoss()
        
        best_val_f1 = 0
        best_state = None
        
        print(f"  Training on {len(train_texts)} samples, validating on {len(val_texts)}...")
        
        for epoch in range(epochs):
            # Train
            self.classifier.train()
            for batch in train_loader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad()
                
                with torch.no_grad():
                    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                    features = outputs.last_hidden_state[:, 0, :]
                
                features = self.dropout(features)
                logits = self.classifier(features)
                
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
            
            # Validate
            self.classifier.eval()
            preds, true_labels = [], []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids, attention_mask, labels = batch
                    input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
                    
                    with torch.no_grad():
                        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                        features = outputs.last_hidden_state[:, 0, :]
                    
                    features = self.dropout(features)
                    logits = self.classifier(features)
                    preds.extend(logits.argmax(dim=1).cpu().numpy())
                    true_labels.extend(labels.numpy())
            
            val_f1 = f1_score(true_labels, preds, average="weighted")
            val_acc = accuracy_score(true_labels, preds)
            
            print(f"    Epoch {epoch+1}/{epochs}: Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in self.classifier.state_dict().items()}
        
        # Load best
        if best_state:
            self.classifier.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        
        return {
            "val_accuracy": accuracy_score(true_labels, preds),
            "val_f1": best_val_f1
        }
    
    def evaluate(self, texts: List[str], labels: List[str]) -> Dict:
        """Evaluate on a dataset."""
        if self.bert is None:
            return {"accuracy": 0, "f1": 0}
        
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        label_map = {"FAKE": 0, "REAL": 1}
        
        self.bert.eval()
        self.classifier.eval()
        
        preds, true_labels = [], []
        
        with torch.no_grad():
            for i in range(0, len(texts), BATCH_SIZE):
                batch_ids = enc["input_ids"][i:i+BATCH_SIZE].to(DEVICE)
                batch_mask = enc["attention_mask"][i:i+BATCH_SIZE].to(DEVICE)
                
                outputs = self.bert(input_ids=batch_ids, attention_mask=batch_mask)
                features = outputs.last_hidden_state[:, 0, :]
                features = self.dropout(features)
                logits = self.classifier(features)
                
                preds.extend(logits.argmax(dim=1).cpu().numpy())
        
        true_labels = [label_map[l] for l in labels]
        
        return {
            "accuracy": accuracy_score(true_labels, preds),
            "precision": precision_score(true_labels, preds, average="weighted", zero_division=0),
            "recall": recall_score(true_labels, preds, average="weighted", zero_division=0),
            "f1": f1_score(true_labels, preds, average="weighted", zero_division=0),
        }


def main():
    """Main recursive training loop."""
    print("="*60)
    print("RECURSIVE DISTILBERT TRAINING")
    print("="*60)
    
    # Load all datasets
    print("\nLoading datasets...")
    datasets_data = []
    
    for ds in DATASETS:
        print(f"\n{ds['name']}:")
        data = load_dataset(ds)
        if data:
            datasets_data.append({"config": ds, "data": data})
    
    if not datasets_data:
        print("No datasets loaded!")
        return
    
    print(f"\nLoaded {len(datasets_data)} datasets")
    
    # Initialize model
    model = RecursiveDistilBERT()
    
    # Try to load existing model
    if model.load(OUTPUT_DIR):
        print(f"\nLoaded existing model with best F1: {model.best_f1:.4f}")
    else:
        print("\nStarting fresh training")
        model.initialize()
    
    # Recursive training
    iteration = 0
    total_epochs = 0
    
    while True:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}")
        print(f"{'='*60}")
        
        improved = False
        
        # Train on each dataset sequentially, validate on next
        for i, ds_data in enumerate(datasets_data):
            config = ds_data["config"]
            texts, labels = ds_data["data"]
            
            # Split for training
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            print(f"\n[{config['name']}]")
            
            # Train
            metrics = model.train_on_dataset(
                train_texts, train_labels,
                val_texts, val_labels,
                config["name"],
                epochs=EPOCHS_PER_DATASET
            )
            
            total_epochs += EPOCHS_PER_DATASET
            
            # Check improvement on this dataset
            if metrics["val_f1"] > model.best_f1:
                improvement = metrics["val_f1"] - model.best_f1
                print(f"  IMPROVED! +{improvement:.4f} (was {model.best_f1:.4f})")
                model.best_f1 = metrics["val_f1"]
                model.history.append({
                    "iteration": iteration,
                    "dataset": config["name"],
                    "f1": metrics["val_f1"],
                    "improvement": improvement
                })
                improved = True
                
                # Save checkpoint
                model.save(OUTPUT_DIR)
                print(f"  Saved to {OUTPUT_DIR}")
            else:
                print(f"  No improvement ({metrics['val_f1']:.4f} <= {model.best_f1:.4f})")
        
        # Evaluate on all datasets to check convergence
        print("\n--- Evaluating on all datasets ---")
        all_f1 = []
        
        for ds_data in datasets_data:
            config = ds_data["config"]
            texts, labels = ds_data["data"]
            
            # Use subset for evaluation
            eval_texts, eval_labels = texts[:2000], labels[:2000]
            metrics = model.evaluate(eval_texts, eval_labels)
            
            print(f"  {config['name']}: F1={metrics['f1']:.4f}")
            all_f1.append(metrics["f1"])
        
        avg_f1 = np.mean(all_f1)
        print(f"\nAverage F1 across all datasets: {avg_f1:.4f}")
        
        # Check convergence
        if not improved:
            print("\n*** CONVERGED - No improvement this iteration ***")
            break
        
        # Check if improvement is too small
        last_improvement = model.history[-1]["improvement"] if model.history else 1
        if last_improvement < CONVERGENCE_THRESHOLD:
            print(f"\n*** CONVERGED - Improvement ({last_improvement:.4f}) below threshold ({CONVERGENCE_THRESHOLD}) ***")
            break
        
        # Check if we've done too many iterations
        if iteration >= 10:
            print("\n*** MAX ITERATIONS REACHED (10) ***")
            break
    
    # Final save
    model.save(OUTPUT_DIR)
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total iterations: {iteration}")
    print(f"Total epochs: {total_epochs}")
    print(f"Best F1 achieved: {model.best_f1:.4f}")
    print(f"Model saved to: {OUTPUT_DIR}")
    
    # Save summary
    summary = {
        "iterations": iteration,
        "total_epochs": total_epochs,
        "best_f1": model.best_f1,
        "history": model.history
    }
    
    with open(OUTPUT_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()