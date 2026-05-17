"""Direct inference using trained models from artifacts."""

import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from config import AppConfig
from preprocessing import TextPreprocessor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FakeNewsClassifier(nn.Module):
    """Simple embedding + mean + linear classifier."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x.squeeze(-1)


class DatasetModel:
    """Wrapper for a single dataset's trained model."""
    
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.metadata = None
        self.max_len = 100
        self.vocab_size = 10000
        
    def load(self) -> bool:
        """Load model and artifacts."""
        try:
            model_dir = self.model_dir
            
            # Load tokenizer
            tokenizer_file = model_dir / "tokenizer.json"
            if tokenizer_file.exists():
                with open(tokenizer_file) as f:
                    self.tokenizer = json.load(f)
                    # Get vocab size from tokenizer
                    self.vocab_size = self.tokenizer.get('vocab_size', 10000)
                    self.max_len = 100  # Default
                    
            # Load metadata
            meta_file = model_dir / "metrics.json"
            if meta_file.exists():
                with open(meta_file) as f:
                    self.metadata = json.load(f)
            
            # Find model file (prefer best_model.pt, fallback to model.pt)
            model_file = model_dir / "best_model.pt"
            if not model_file.exists():
                model_file = model_dir / "model.pt"
                
            if not model_file.exists():
                print(f"No model file found in {model_dir}")
                return False
            
            # Load model with correct vocab size
            self.model = FakeNewsClassifier(
                vocab_size=self.vocab_size,
                embedding_dim=128,
                dropout=0.3
            ).to(DEVICE)
            
            state_dict = torch.load(model_file, map_location=DEVICE)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            print(f"Loaded model: {self.model_dir.name} (vocab={self.vocab_size})")
            return True
            
        except Exception as e:
            print(f"Error loading model from {self.model_dir}: {e}")
            import traceback
            traceback.print_exc()
            
        return False
    
    def predict(self, text: str) -> Tuple[str, float]:
        """Predict on text."""
        if not self.model or not self.tokenizer:
            return "UNKNOWN", 0.0
            
        # Tokenize - handle both word_index and word_to_idx formats
        word_to_idx = self.tokenizer.get('word_index', self.tokenizer.get('word_to_idx', {}))
        oov_token = self.tokenizer.get('oov_token', '<OOV>')
        oov_idx = word_to_idx.get(oov_token, 1)
        
        words = text.lower().split()
        indices = [word_to_idx.get(w, oov_idx) for w in words]
        
        # Pad
        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
            
        # Predict
        with torch.no_grad():
            x = torch.tensor([indices], dtype=torch.long).to(DEVICE)
            output = self.model(x)
            prob = torch.sigmoid(output).item()
            
        prediction = "FAKE" if prob < 0.5 else "REAL"
        confidence = prob if prediction == "REAL" else 1 - prob
        
        return prediction, confidence


class MultiModelInference:
    """Inference system using multiple trained models with weighted selection."""
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.models: Dict[str, DatasetModel] = {}
        self.weights: Dict[str, float] = {}
        self.best_model_name: Optional[str] = None
        
    def load_all_models(self) -> int:
        """Load all trained models from artifacts."""
        if not self.models_dir.exists():
            print(f"Models directory not found: {self.models_dir}")
            return 0
            
        # Load leaderboard weights
        leaderboard_file = self.models_dir / "leaderboard.json"
        if leaderboard_file.exists():
            with open(leaderboard_file) as f:
                data = json.load(f)
                for w in data.get('weights', []):
                    self.weights[w['dataset_name']] = w['final_weight']
                    
        # Load each model
        loaded = 0
        for model_dir in self.models_dir.iterdir():
            if not model_dir.is_dir() or model_dir.name == 'leaderboard.json':
                continue
                
            dataset_model = DatasetModel(model_dir)
            if dataset_model.load():
                self.models[model_dir.name] = dataset_model
                loaded += 1
                
        # Set best model
        if self.weights and self.models:
            best_name = max(self.weights, key=self.weights.get)
            if best_name in self.models:
                self.best_model_name = best_name
                
        print(f"Loaded {loaded} models")
        return loaded
    
    def predict(self, text: str, model_name: Optional[str] = None) -> Dict:
        """Predict using specified model or best model."""
        if not self.models:
            return {"error": "No models loaded"}
            
        # Use specified model or best model
        if model_name and model_name in self.models:
            model = self.models[model_name]
        elif self.best_model_name:
            model = self.models[self.best_model_name]
        else:
            # Use first available model
            model = list(self.models.values())[0]
            
        prediction, confidence = model.predict(text)
        
        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "model_used": model_name or self.best_model_name or "auto",
            "available_models": list(self.models.keys()),
            "weights": self.weights
        }
    
    def predict_ensemble(self, text: str) -> Dict:
        """Predict using weighted ensemble of all models."""
        if not self.models:
            return {"error": "No models loaded"}
            
        predictions = {}
        confidences = {}
        
        for name, model in self.models.items():
            pred, conf = model.predict(text)
            predictions[name] = pred
            confidences[name] = conf
            
        # Weighted vote based on model weights
        weighted_scores = {"FAKE": 0.0, "REAL": 0.0}
        
        for name, pred in predictions.items():
            weight = self.weights.get(name, 0.5)
            weighted_scores[pred] += weight
            
        final_prediction = max(weighted_scores, key=weighted_scores.get)
        avg_confidence = sum(confidences.values()) / len(confidences)
        
        return {
            "prediction": final_prediction,
            "confidence": round(avg_confidence, 4),
            "method": "ensemble",
            "model_votes": predictions,
            "model_weights": self.weights,
            "available_models": list(self.models.keys())
        }
    
    def get_info(self) -> Dict:
        """Get information about loaded models."""
        return {
            "total_models": len(self.models),
            "models": list(self.models.keys()),
            "weights": self.weights,
            "best_model": self.best_model_name,
            "best_weight": self.weights.get(self.best_model_name, 0) if self.best_model_name else 0
        }


# Global inference instance
_inference_instance: Optional[MultiModelInference] = None


def get_inference() -> MultiModelInference:
    """Get or create the global inference instance."""
    global _inference_instance
    
    if _inference_instance is None:
        base_dir = Path(__file__).parent.parent
        models_dir = base_dir / "artifacts" / "models"
        _inference_instance = MultiModelInference(models_dir)
        _inference_instance.load_all_models()
        
    return _inference_instance


def predict(text: str, model_name: Optional[str] = None) -> Dict:
    """Quick prediction function."""
    return get_inference().predict(text, model_name)


def predict_ensemble(text: str) -> Dict:
    """Ensemble prediction function."""
    return get_inference().predict_ensemble(text)


if __name__ == "__main__":
    # Test the inference
    inference = get_inference()
    print(f"Loaded models: {inference.get_info()}")
    
    test_text = "Breaking: Scientists discover new cure for cancer"
    result = predict(test_text)
    print(f"\nSingle model prediction: {result}")
    
    result = predict_ensemble(test_text)
    print(f"\nEnsemble prediction: {result}")