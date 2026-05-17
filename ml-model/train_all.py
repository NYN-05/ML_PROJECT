"""Multi-Dataset Training Pipeline - Train separate models per dataset with weighted scoring."""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from config import AppConfig, build_default_config
from logging_utils import get_logger
from model_builder import build_lstm_model
from preprocessing import TextPreprocessor
from tokenizer_utils import WordTokenizer, build_tokenizer, compute_sequence_length, save_tokenizer, texts_to_padded

LOGGER = get_logger("ml.train_all")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER.info("Using device: %s", DEVICE)


@dataclass
class DatasetInfo:
    """Information about a single dataset."""
    name: str
    path: Path
    df: pd.DataFrame
    size: int


@dataclass
class ModelMetrics:
    """Metrics for a trained model."""
    dataset_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    dataset_size: int
    train_samples: int
    test_samples: int
    training_time: float
    
    def to_dict(self) -> Dict:
        return {
            "dataset_name": self.dataset_name,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "dataset_size": self.dataset_size,
            "train_samples": self.train_samples,
            "test_samples": self.test_samples,
            "training_time": self.training_time
        }


@dataclass 
class ModelWeight:
    """Weighted score for model selection."""
    dataset_name: str
    accuracy_weight: float
    precision_weight: float
    recall_weight: float
    f1_weight: float
    size_weight: float
    final_weight: float
    
    def to_dict(self) -> Dict:
        return {
            "dataset_name": self.dataset_name,
            "weights": {
                "accuracy": self.accuracy_weight,
                "precision": self.precision_weight,
                "recall": self.recall_weight,
                "f1_score": self.f1_weight,
                "dataset_size": self.size_weight
            },
            "final_weight": self.final_weight
        }


class TextDataset(Dataset):
    """PyTorch Dataset for text sequences."""
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.labels[idx]


class MultiDatasetTrainer:
    """Train separate LSTM model for each dataset with weighted scoring."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.dataset_results: List[ModelMetrics] = []
        self.model_weights: List[ModelWeight] = []
        
    def discover_datasets(self, dataset_dir: Path) -> List[DatasetInfo]:
        """Discover all datasets in the dataset directory."""
        datasets = []
        
        for item in dataset_dir.iterdir():
            if not item.is_dir() or item.name in ['processed']:
                continue
                
            df = self._load_dataset(item)
            if df is not None and len(df) > 100:
                datasets.append(DatasetInfo(
                    name=item.name,
                    path=item,
                    df=df,
                    size=len(df)
                ))
                LOGGER.info(f"Discovered dataset: {item.name} ({len(df)} samples)")
                
        return datasets
    
    def _load_dataset(self, dataset_path: Path) -> Optional[pd.DataFrame]:
        """Load a single dataset based on its structure."""
        # Check for fake/true CSV pair
        fake_csv = dataset_path / "Fake.csv"
        true_csv = dataset_path / "True.csv"
        
        if fake_csv.exists() and true_csv.exists():
            fake_df = pd.read_csv(fake_csv)
            true_df = pd.read_csv(true_csv)
            fake_df['label'] = 'FAKE'
            true_df['label'] = 'REAL'
            df = pd.concat([fake_df, true_df], ignore_index=True)
            return self._normalize_dataframe(df)
            
        # Check for single CSV
        for csv_file in dataset_path.glob("*.csv"):
            if csv_file.name.lower() in ['fake.csv', 'true.csv', 'train_dataset.csv']:
                continue
            try:
                df = pd.read_csv(csv_file)
                if 'label' in df.columns:
                    return self._normalize_dataframe(df)
            except:
                continue
                
        return None
    
    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize dataframe columns."""
        df = df.copy()
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Ensure required columns exist
        if 'title' not in df.columns:
            df['title'] = df.get('statement', df.get('text', '')).fillna('')
        if 'text' not in df.columns:
            df['text'] = ''
        if 'label' not in df.columns:
            return None
            
        # Normalize labels
        df['label'] = df['label'].astype(str).str.upper().str.strip()
        df['label'] = df['label'].map(lambda x: 'REAL' if x in ['1', 'TRUE', 'REAL'] else 'FAKE')
        
        # Fill missing
        df['title'] = df['title'].fillna('')
        df['text'] = df['text'].fillna('')
        
        return df[['title', 'text', 'label']].dropna()
    
    def train_single_dataset(self, dataset_info: DatasetInfo, output_dir: Path) -> Optional[ModelMetrics]:
        """Train LSTM model on a single dataset."""
        import time
        start_time = time.time()
        
        dataset_dir = output_dir / dataset_info.name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"Training on dataset: {dataset_info.name}")
        LOGGER.info(f"{'='*60}")
        
        df = dataset_info.df.copy()
        
        # Preprocess text
        preprocessor = TextPreprocessor()
        df['clean_text'] = (df['title'].astype(str) + ' ' + df['text'].astype(str)).apply(preprocessor.preprocess_text)
        df = df[df['clean_text'].str.len() > 10].reset_index(drop=True)
        
        if len(df) < 100:
            LOGGER.warning(f"Dataset {dataset_info.name} too small after cleaning")
            return None
        
        # Encode labels
        y = df['label'].map({'REAL': 1, 'FAKE': 0}).values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['clean_text'].tolist(), y,
            test_size=0.2, random_state=42, stratify=y
        )
        
        # Compute sequence length
        max_len = compute_sequence_length(X_train, min_len=50, max_len=500, percentile=95)
        
        # Build tokenizer
        tokenizer = build_tokenizer(X_train, self.config)
        x_train = texts_to_padded(tokenizer, X_train, max_len)
        x_test = texts_to_padded(tokenizer, X_test, max_len)
        
        # Create data loaders
        train_ds = TextDataset(x_train, y_train)
        test_ds = TextDataset(x_test, y_test)
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=32)
        
        # Compute class weights
        class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
        weight_tensor = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float, device=DEVICE)
        
        # Build model
        model = build_lstm_model(self.config, max_len).to(DEVICE)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor[1]/weight_tensor[0])
        
        # Training loop with early stopping
        best_val_acc = 0
        patience = 5
        patience_counter = 0
        
        for epoch in range(15):
            model.train()
            for sequences, labels_batch in train_loader:
                sequences, labels_batch = sequences.to(DEVICE), labels_batch.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(sequences)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for sequences, labels_batch in test_loader:
                    sequences, labels_batch = sequences.to(DEVICE), labels_batch.to(DEVICE)
                    outputs = model(sequences)
                    preds = (torch.sigmoid(outputs) >= 0.5).float()
                    correct += (preds == labels_batch).sum().item()
                    total += len(labels_batch)
            
            val_acc = correct / total
            LOGGER.info(f"  Epoch {epoch+1}/15 - Val Accuracy: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), dataset_dir / "best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    LOGGER.info(f"  Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model and evaluate
        model.load_state_dict(torch.load(dataset_dir / "best_model.pt"))
        model.eval()
        
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for sequences, labels_batch in test_loader:
                sequences = sequences.to(DEVICE)
                outputs = model(sequences)
                preds = (torch.sigmoid(outputs) >= 0.5).cpu().float().numpy().astype(int)
                all_preds.extend(preds.tolist())
                all_labels.extend(labels_batch.numpy().tolist())
        
        # Compute metrics
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        
        metrics = ModelMetrics(
            dataset_name=dataset_info.name,
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, zero_division=0),
            recall=recall_score(y_true, y_pred, zero_division=0),
            f1_score=f1_score(y_true, y_pred, zero_division=0),
            dataset_size=len(df),
            train_samples=len(y_train),
            test_samples=len(y_test),
            training_time=time.time() - start_time
        )
        
        # Save artifacts
        torch.save(model.state_dict(), dataset_dir / "model.pt")
        save_tokenizer(tokenizer, dataset_dir / "tokenizer.json")
        
        # Save metrics
        with open(dataset_dir / "metrics.json", 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        
        LOGGER.info(f"  Accuracy: {metrics.accuracy:.4f}, F1: {metrics.f1_score:.4f}")
        
        return metrics
    
    def compute_model_weights(self) -> List[ModelWeight]:
        """Compute weights for each model based on metrics and dataset size."""
        if not self.dataset_results:
            return []
        
        # Get metrics
        accuracies = [r.accuracy for r in self.dataset_results]
        precisions = [r.precision for r in self.dataset_results]
        recalls = [r.recall for r in self.dataset_results]
        f1s = [r.f1_score for r in self.dataset_results]
        sizes = [r.dataset_size for r in self.dataset_results]
        
        # Normalize to 0-1 range
        def normalize(values):
            min_val, max_val = min(values), max(values)
            if max_val == min_val:
                return [0.5] * len(values)
            return [(v - min_val) / (max_val - min_val) for v in values]
        
        norm_acc = normalize(accuracies)
        norm_prec = normalize(precisions)
        norm_rec = normalize(recalls)
        norm_f1 = normalize(f1s)
        norm_size = normalize(sizes)
        
        # Weights: 20% each metric, 20% dataset size
        weights = []
        for i, result in enumerate(self.dataset_results):
            final = (norm_acc[i] * 0.2 + norm_prec[i] * 0.2 + 
                    norm_rec[i] * 0.2 + norm_f1[i] * 0.2 + norm_size[i] * 0.2)
            
            weights.append(ModelWeight(
                dataset_name=result.dataset_name,
                accuracy_weight=round(norm_acc[i], 4),
                precision_weight=round(norm_prec[i], 4),
                recall_weight=round(norm_rec[i], 4),
                f1_weight=round(norm_f1[i], 4),
                size_weight=round(norm_size[i], 4),
                final_weight=round(final, 4)
            ))
        
        return weights
    
    def train_all(self, dataset_dir: Path, output_dir: Path) -> Dict:
        """Train models on all datasets and compute weights."""
        # Discover datasets
        datasets = self.discover_datasets(dataset_dir)
        LOGGER.info(f"Found {len(datasets)} datasets to train")
        
        # Train on each dataset
        for ds in datasets:
            metrics = self.train_single_dataset(ds, output_dir)
            if metrics:
                self.dataset_results.append(metrics)
        
        # Compute weights
        self.model_weights = self.compute_model_weights()
        
        # Save leaderboard
        leaderboard = {
            "training_date": datetime.now().isoformat(),
            "total_datasets": len(self.dataset_results),
            "models": [m.to_dict() for m in self.dataset_results],
            "weights": [w.to_dict() for w in self.model_weights],
            "best_model": max(self.model_weights, key=lambda w: w.final_weight).dataset_name if self.model_weights else None
        }
        
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "leaderboard.json", 'w') as f:
            json.dump(leaderboard, f, indent=2)
        
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info("TRAINING COMPLETE")
        LOGGER.info(f"{'='*60}")
        LOGGER.info(f"Total models trained: {len(self.dataset_results)}")
        
        if self.model_weights:
            best = max(self.model_weights, key=lambda w: w.final_weight)
            LOGGER.info(f"Best model: {best.dataset_name} (weight: {best.final_weight:.4f})")
        
        return leaderboard


def main():
    """Main entry point."""
    config = build_default_config()
    
    # Get absolute paths
    base_dir = Path(__file__).parent.parent
    dataset_dir = base_dir / "dataset"
    output_dir = base_dir / "artifacts" / "models"
    
    trainer = MultiDatasetTrainer(config)
    results = trainer.train_all(dataset_dir, output_dir)
    
    print("\n" + "="*60)
    print("MODEL LEADERBOARD")
    print("="*60)
    
    for w in sorted(results['weights'], key=lambda x: x['final_weight'], reverse=True):
        print(f"\n{w['dataset_name']}:")
        print(f"  Accuracy Weight: {w['weights']['accuracy']}")
        print(f"  Precision Weight: {w['weights']['precision']}")
        print(f"  Recall Weight: {w['weights']['recall']}")
        print(f"  F1 Weight: {w['weights']['f1_score']}")
        print(f"  Size Weight: {w['weights']['dataset_size']}")
        print(f"  FINAL WEIGHT: {w['final_weight']}")


if __name__ == "__main__":
    main()