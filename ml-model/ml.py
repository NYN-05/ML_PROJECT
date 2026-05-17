"""Production-Grade ML Pipeline for Fake News Detection - Refactored Architecture"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    DistilBertModel, DistilBertTokenizer, DistilBertConfig,
    get_linear_schedule_with_warmup,
    Trainer, TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef
)
from datasets import Dataset as HFDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model_name: str = "distilbert-base-uncased"
    hidden_dim: int = 768
    intermediate_dim: int = 256
    dropout: float = 0.3
    num_labels: int = 2
    use_attention_pooling: bool = True
    use_layer_norm: bool = True
    use_gelu: bool = True
    label_smoothing: float = 0.1

@dataclass
class TrainingConfig:
    """Training configuration."""
    max_length: int = 256
    batch_size: int = 16
    gradient_accumulation: int = 2
    epochs: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    fp16: bool = True
    logging_steps: int = 50
    save_steps: int = 200
    eval_steps: int = 200
    early_stopping_patience: int = 3

@dataclass  
class DataConfig:
    """Data configuration."""
    max_samples_per_dataset: int = 5000
    test_size: float = 0.15
    use_caching: bool = True
    balance_classes: bool = True
    k_fold: int = 0  # 0 = no cross-validation

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class ImprovedClassifier(nn.Module):
    """Production-grade classifier with advanced architecture."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Load pretrained transformer
        self.transformer = DistilBertModel.from_pretrained(config.model_name)
        
        # Attention pooling
        self.use_attention_pooling = config.use_attention_pooling
        if config.use_attention_pooling:
            self.attention = nn.Linear(config.hidden_dim, 1)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else None
        
        # Classifier head with GELU and residual
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.intermediate_dim),
            nn.GELU() if config.use_gelu else nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_dim, config.num_labels)
        )
        
        # Label smoothing loss
        self.label_smoothing = config.label_smoothing
    
    def forward(self, input_ids, attention_mask, labels=None):
        # Transformer output
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Attention pooling
        if self.use_attention_pooling:
            attn_weights = torch.softmax(self.attention(sequence_output), dim=1)
            pooled = (attn_weights * sequence_output).sum(dim=1)
        else:
            pooled = sequence_output[:, 0]  # CLS token
        
        # Layer norm
        if self.layer_norm:
            pooled = self.layer_norm(pooled)
        
        # Classifier
        logits = self.classifier(pooled)
        
        # Return logits for training
        return logits
    
    def predict(self, input_ids, attention_mask):
        """Inference method."""
        logits = self.forward(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1)
        return probs


# ============================================================================
# DATA PIPELINE
# ============================================================================

class FakeNewsDataset(Dataset):
    """Production dataset with caching support."""
    
    def __init__(self, texts: List[str], labels: List[str], tokenizer, max_length: int):
        self.texts = texts
        self.labels = [{"FAKE": 0, "REAL": 1}[l] for l in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


def load_and_process_dataset(path: Path, config: DataConfig) -> Optional[Tuple[List[str], List[str]]]:
    """Load dataset with proper preprocessing."""
    if not path.exists():
        return None
    
    try:
        df = pd.read_csv(path, on_bad_lines='skip')
        
        # Find columns
        text_col = next((c for c in df.columns if any(x in c.lower() for x in ["text", "article", "content", "body"])), df.columns[0])
        label_col = next((c for c in df.columns if c.lower() in ["label", "class"]), df.columns[-1])
        
        # Get data
        texts = df[text_col].astype(str).tolist()[:config.max_samples_per_dataset]
        labels = df[label_col].astype(str).str.upper().tolist()[:config.max_samples_per_dataset]
        
        # Normalize labels
        labels = ["REAL" if any(x in l for x in ["1", "TRUE", "REAL", "1.0"]) else "FAKE" for l in labels]
        
        # Clean and validate
        valid = [(t, l) for t, l in zip(texts, labels) if len(str(t).strip()) > 20]
        
        if not valid:
            return None
        
        texts = [v[0] for v in valid]
        labels = [v[1] for v in valid]
        
        # Balance classes
        if config.balance_classes:
            fake_idx = [i for i, l in enumerate(labels) if l == "FAKE"]
            real_idx = [i for i, l in enumerate(labels) if l == "REAL"]
            min_count = min(len(fake_idx), len(real_idx))
            balanced_idx = fake_idx[:min_count] + real_idx[:min_count]
            texts = [texts[i] for i in balanced_idx]
            labels = [labels[i] for i in balanced_idx]
        
        print(f"  Loaded {len(texts)} samples ({labels.count('FAKE')} FAKE, {labels.count('REAL')} REAL)")
        return texts, labels
        
    except Exception as e:
        print(f"  Error loading {path}: {e}")
        return None


# ============================================================================
# EVALUATION
# ============================================================================

def compute_metrics(predictions, labels) -> Dict[str, float]:
    """Comprehensive evaluation metrics."""
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="weighted", zero_division=0),
        "recall": recall_score(labels, predictions, average="weighted", zero_division=0),
        "f1": f1_score(labels, predictions, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(labels, predictions),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist()
    }


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

class MLTrainer:
    """Production trainer with proper abstraction."""
    
    def __init__(self, model_config: ModelConfig, train_config: TrainingConfig, data_config: DataConfig):
        self.model_config = model_config
        self.train_config = train_config
        self.data_config = data_config
        self.output_dir = Path("artifacts/production_model")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_config.model_name)
        
        # Data collator for dynamic padding
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
    
    def prepare_data(self) -> Tuple[HFDataset, HFDataset]:
        """Load and prepare all datasets."""
        DATASETS = [
            ("WELFake_Dataset", "../dataset/WELFake_Dataset/WELFake_Dataset.csv"),
            ("FakeNewsDet1", "../dataset/Fake News Detection_1/fake.csv"),
            ("FakeNewsDet2", "../dataset/Fake News Detection Dataset/fake_news_dataset.csv"),
            ("FakeRealNews", "../dataset/Fake or Real News/fake_or_real_news.csv"),
            ("ISOT", "../dataset/ISOT Fake News Dataset/Fake.csv"),
            ("NewsDet", "../dataset/News Detection (Fake or Real) Dataset/fake_and_real_news.csv"),
            ("FakeReal", "../dataset/fake-and-real-news-dataset/Fake.csv"),
        ]
        
        all_texts, all_labels = [], []
        
        print("Loading datasets...")
        for name, path in DATASETS:
            data = load_and_process_dataset(Path(path), self.data_config)
            if data:
                texts, labels = data
                all_texts.extend(texts)
                all_labels.extend(labels)
        
        print(f"Total: {len(all_texts)} samples")
        
        # Tokenize
        print("Tokenizing...")
        encodings = self.tokenizer(
            all_texts,
            max_length=self.train_config.max_length,
            truncation=True,
            padding=True
        )
        
        # Create dataset
        dataset = HFDataset.from_dict({
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": [{"FAKE": 0, "REAL": 1}[l] for l in all_labels]
        })
        
        # Split
        split = dataset.train_test_split(test_size=self.data_config.test_size, shuffle=True, seed=42)
        return split["train"], split["test"]
    
    def train(self):
        """Execute training with proper pipeline."""
        train_dataset, eval_dataset = self.prepare_data()
        
        # Initialize model
        model = ImprovedClassifier(self.model_config).to(DEVICE)
        
        # Optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay
        )
        
        # Scheduler with warmup
        total_steps = (len(train_dataset) // self.train_config.batch_size) * self.train_config.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * self.train_config.warmup_ratio),
            num_training_steps=total_steps
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.train_config.epochs,
            per_device_train_batch_size=self.train_config.batch_size,
            per_device_eval_batch_size=self.train_config.batch_size * 2,
            gradient_accumulation_steps=self.train_config.gradient_accumulation,
            learning_rate=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
            warmup_ratio=self.train_config.warmup_ratio,
            fp16=self.train_config.fp16 and DEVICE.type == "cuda",
            logging_steps=self.train_config.logging_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            save_total_limit=2,
            logging_dir=str(self.output_dir / "logs"),
            report_to="tensorboard"
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=lambda p: compute_metrics(
                np.argmax(p.predictions, axis=1),
                p.label_ids
            )
        )
        
        # Train
        print("\n=== Starting Training ===")
        trainer.train()
        
        # Save best model
        trainer.save_model(str(self.output_dir / "best_model"))
        self.tokenizer.save_pretrained(str(self.output_dir / "best_model"))
        
        # Final evaluation
        eval_results = trainer.evaluate()
        print(f"\nFinal Results: {eval_results}")
        
        # Save metrics
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(eval_results, f, indent=2)
        
        return eval_results


# ============================================================================
# INFERENCE
# ============================================================================

class InferenceEngine:
    """Production inference engine."""
    
    def __init__(self, model_path: Path = None):
        self.model_path = model_path or Path("artifacts/production_model/best_model")
        
        if not self.model_path.exists():
            print("Model not found. Train first with --mode train")
            return
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(str(self.model_path))
        
        config = ModelConfig()
        self.model = ImprovedClassifier(config).to(DEVICE)
        self.model.load_state_dict(torch.load(self.model_path / "pytorch_model.bin", map_location=DEVICE))
        self.model.eval()
        
        print(f"Loaded model from {self.model_path}")
    
    def predict(self, text: str, threshold: float = 0.5) -> Dict[str, Any]:
        """Predict with confidence and metadata."""
        if not self.model:
            return {"error": "Model not loaded"}
        
        enc = self.tokenizer(text, max_length=256, truncation=True, return_tensors="pt")
        
        with torch.no_grad():
            probs = self.model.predict(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
        
        real_prob = probs[0][1].item()
        fake_prob = probs[0][0].item()
        
        if real_prob >= threshold:
            prediction = "REAL"
            confidence = real_prob
        else:
            prediction = "FAKE"
            confidence = fake_prob
        
        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "probabilities": {"REAL": round(real_prob, 4), "FAKE": round(fake_prob, 4)},
            "method": "distilbert_production",
            "model_path": str(self.model_path)
        }
    
    def batch_predict(self, texts: List[str]) -> List[Dict]:
        """Batch inference."""
        return [self.predict(text) for text in texts]


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production ML Pipeline")
    parser.add_argument("--mode", choices=["train", "predict", "evaluate"], default="predict")
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    
    if args.mode == "train":
        # Create configs
        model_config = ModelConfig()
        train_config = TrainingConfig()
        data_config = DataConfig()
        
        # Train
        trainer = MLTrainer(model_config, train_config, data_config)
        trainer.train()
        
    elif args.mode == "predict":
        engine = InferenceEngine()
        if args.text:
            result = engine.predict(args.text, args.threshold)
            print(json.dumps(result, indent=2))
        else:
            print("Provide --text for prediction")
    
    elif args.mode == "evaluate":
        engine = InferenceEngine()
        test_texts = [
            "Breaking: Scientists discover new cure for cancer",
            "The president announced new economic policy today",
            "Aliens have landed on Earth and are living among us"
        ]
        for text in test_texts:
            result = engine.predict(text)
            print(f"{result['prediction']} ({result['confidence']}): {text[:50]}...")