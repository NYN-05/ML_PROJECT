"""DistilBERT-based fake news inference."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer


class DistilBERTInference:
    """DistilBERT inference system."""
    
    def __init__(self, model_dir: Path = Path("artifacts/models_fakebert")):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert = None
        self.classifier = None
        self.dropout = None
        self.tokenizer: Optional[DistilBertTokenizer] = None
        self.max_length = 128
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer."""
        if not self.model_dir.exists():
            print(f"Model not found at {self.model_dir}")
            return
        
        # Load tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(str(self.model_dir / "tokenizer"))
        
        # Load checkpoint
        checkpoint = torch.load(self.model_dir / "model.pt", map_location="cpu", weights_only=False)
        
        # Load pretrained BERT
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        
        # Create classifier
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
        
        # Load trained weights (only classifier)
        state_dict = checkpoint["model_state"]
        classifier_state = {k: v for k, v in state_dict.items() if k.startswith("classifier.")}
        
        self.classifier.load_state_dict(classifier_state, strict=False)
        self.max_length = checkpoint.get("max_length", 128)
        
        # Move to device
        self.bert.to(self.device)
        self.classifier.to(self.device)
        self.bert.eval()
        self.classifier.eval()
        
        print(f"Loaded DistilBERT model from {self.model_dir}")
    
    def predict(self, text: str, threshold: float = 0.5) -> Tuple[str, float]:
        """Predict on a single text."""
        if not self.bert or not self.classifier or not self.tokenizer:
            return "UNKNOWN", 0.0
        
        encoding = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]
            dropped = self.dropout(cls_output)
            logits = self.classifier(dropped)
            probs = torch.softmax(logits, dim=1)
            
            real_prob = probs[0][1].item()
            
            if real_prob >= threshold:
                return "REAL", real_prob
            else:
                return "FAKE", probs[0][0].item()
    
    def get_info(self) -> Dict:
        """Get model info."""
        metrics_path = self.model_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                return json.load(f)
        return {}


# Global instance
_distilbert_inference: Optional[DistilBERTInference] = None


def get_distilbert_inference() -> DistilBERTInference:
    """Get or create DistilBERT inference instance."""
    global _distilbert_inference
    if _distilbert_inference is None:
        _distilbert_inference = DistilBERTInference()
    return _distilbert_inference


def predict(text: str, threshold: float = 0.5) -> Dict:
    """Quick prediction function."""
    pred, conf = get_distilbert_inference().predict(text, threshold)
    return {"prediction": pred, "confidence": conf}


if __name__ == "__main__":
    inf = get_distilbert_inference()
    
    tests = [
        "Breaking news: Scientists discover new cure for cancer",
        "The president announced a new economic policy today"
    ]
    
    for text in tests:
        pred, conf = inf.predict(text)
        print(f"{pred} ({conf:.2f}): {text}")