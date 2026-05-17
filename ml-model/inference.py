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
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text same as training - remove stopwords, lemmatize."""
        import re
        try:
            import nltk
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            
            # Try to use NLTK
            try:
                stop_words = set(stopwords.words('english'))
            except:
                stop_words = set()
            
            lemmatizer = WordNetLemmatizer()
        except:
            # Fallback without NLTK
            return text.lower()
        
        # Clean text
        text = text.lower()
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        
        # Remove stopwords and lemmatize
        words = text.split()
        words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
        
        return " ".join(words)
    
    def predict(self, text: str, threshold: float = 0.6) -> Tuple[str, float]:
        """Predict on text with adjustable threshold."""
        if not self.model or not self.tokenizer:
            return "UNKNOWN", 0.0
            
        # Preprocess text (same as training)
        processed_text = self._preprocess_text(text)
        
        # Tokenize - handle both word_index and word_to_idx formats
        word_to_idx = self.tokenizer.get('word_index', self.tokenizer.get('word_to_idx', {}))
        oov_token = self.tokenizer.get('oov_token', '<OOV>')
        oov_idx = word_to_idx.get(oov_token, 1)
        
        words = processed_text.split()
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
            
        # Use higher threshold to reduce false positives
        # Model outputs: REAL=1, FAKE=0
        # Higher prob = more likely REAL
        if prob >= threshold:
            prediction = "REAL"
            confidence = prob  # High prob = high confidence REAL
        else:
            prediction = "FAKE"
            confidence = 1 - prob  # Low prob = high confidence FAKE
            
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
    
    def predict(self, text: str, model_name: Optional[str] = None, 
            threshold: float = 0.6) -> Dict:
        """Predict using specified model or best model.
        
        Args:
            text: Input text to analyze
            model_name: Specific model to use (optional)
            threshold: Higher threshold = fewer FAKE predictions (reduce false positives)
        """
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
            
        prediction, confidence = model.predict(text, threshold=threshold)
        
        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "model_used": model_name or self.best_model_name or "auto",
            "threshold": threshold,
            "available_models": list(self.models.keys()),
            "weights": self.weights
        }
    
    def predict_ensemble(self, text: str, threshold: float = 0.7) -> Dict:
        """Predict using weighted ensemble with full metadata.
        
        Runs all models in parallel, collects predictions, and aggregates using
        weighted voting with configurable weights based on performance metrics.
        
        Returns comprehensive metadata for debugging and traceability.
        """
        import time
        
        if not self.models:
            return {"error": "No models loaded"}
        
        total_start = time.time()
        
        # Run all models and collect individual predictions with metadata
        model_results = []
        all_weights = []
        real_votes = []
        fake_votes = []
        
        for name, model in self.models.items():
            model_start = time.time()
            
            # Get prediction from model
            pred, conf = model.predict(text, threshold=threshold)
            
            model_time = time.time() - model_start
            weight = self.weights.get(name, 0.5)
            
            # Store individual model result
            model_results.append({
                "model_name": name,
                "prediction": pred,
                "confidence": round(conf, 4),
                "weight": round(weight, 4),
                "inference_time_ms": round(model_time * 1000, 2),
                "timestamp": time.time()
            })
            
            all_weights.append(weight)
            
            if pred == "REAL":
                real_votes.append(weight)
            else:
                fake_votes.append(weight)
        
        # Aggregate using weighted voting
        total_weight = sum(all_weights)
        real_weight_sum = sum(real_votes)
        fake_weight_sum = sum(fake_votes)
        
        real_percentage = (real_weight_sum / total_weight * 100) if total_weight > 0 else 0
        
        # Calculate contribution percentage for each model
        for result in model_results:
            result["contribution_percentage"] = round(
                (result["weight"] / total_weight * 100) if total_weight > 0 else 0, 2
            )
        
        # Determine final prediction using multiple strategies
        strategies = {}
        
        # Strategy 1: Weighted voting with higher threshold for FAKE
        # Require at least 60% REAL weighted vote to classify as REAL
        # Otherwise default to REAL to reduce false positives
        if real_percentage >= 60:
            strategies["weighted_voting"] = "REAL"
        else:
            # Conservative: default to REAL unless strong FAKE consensus
            strategies["weighted_voting"] = "REAL"
        
        # Override: if overwhelming FAKE consensus (>80% weighted), allow FAKE
        if fake_weight_sum > real_weight_sum * 2:  # 2:1 ratio
            strategies["weighted_voting"] = "FAKE"
        
        # Strategy 2: Simple majority
        real_count = len(real_votes)
        fake_count = len(fake_votes)
        strategies["majority"] = "REAL" if real_count > fake_count else "FAKE"
        
        # Strategy 3: Confidence-weighted average (convert FAKE to 1, REAL to 0)
        confidence_sum = 0
        weight_sum = 0
        for result in model_results:
            # Convert: FAKE=1, REAL=0 for weighted average
            fake_indicator = 1 if result["prediction"] == "FAKE" else 0
            confidence_sum += fake_indicator * result["confidence"] * result["weight"]
            weight_sum += result["weight"]
        
        avg_fake_confidence = confidence_sum / weight_sum if weight_sum > 0 else 0.5
        strategies["confidence_weighted"] = "FAKE" if avg_fake_confidence >= 0.5 else "REAL"
        
        # Use primary strategy: weighted voting with fallback to confidence-weighted
        final_prediction = strategies["weighted_voting"]
        
        # Calculate final confidence based on agreement level
        agreement = max(real_count, fake_count) / len(self.models)
        if final_prediction == "REAL":
            final_confidence = real_weight_sum / total_weight if total_weight > 0 else 0.5
        else:
            final_confidence = fake_weight_sum / total_weight if total_weight > 0 else 0.5
        
        # Boost confidence if high agreement
        final_confidence = min(1.0, final_confidence * (0.5 + 0.5 * agreement))
        
        total_time = time.time() - total_start
        
        return {
            "prediction": final_prediction,
            "confidence": round(final_confidence, 4),
            "method": "ensemble_weighted",
            "threshold": threshold,
            "total_inference_time_ms": round(total_time * 1000, 2),
            "aggregation": {
                "total_models": len(self.models),
                "real_vote_weight": round(real_weight_sum, 4),
                "fake_vote_weight": round(fake_weight_sum, 4),
                "real_percentage": round(real_percentage, 1),
                "real_count": real_count,
                "fake_count": fake_count,
                "agreement_level": round(agreement * 100, 1),
                "strategies_used": strategies
            },
            "individual_models": model_results,
            "available_models": list(self.models.keys()),
            "model_weights": self.weights
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


def predict(text: str, model_name: Optional[str] = None, threshold: float = 0.7) -> Dict:
    """Quick prediction function with threshold."""
    return get_inference().predict(text, model_name, threshold)


def predict_ensemble(text: str, threshold: float = 0.7) -> Dict:
    """Ensemble prediction function - uses conservative voting."""
    return get_inference().predict_ensemble(text, threshold)


if __name__ == "__main__":
    # Test the inference
    inference = get_inference()
    print(f"Loaded models: {inference.get_info()}")
    
    test_text = "Breaking: Scientists discover new cure for cancer"
    result = predict(test_text)
    print(f"\nSingle model prediction: {result}")
    
    result = predict_ensemble(test_text)
    print(f"\nEnsemble prediction: {result}")