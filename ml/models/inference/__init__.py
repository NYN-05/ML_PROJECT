"""Unified inference system for predictions."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

import numpy as np
import joblib

logger = logging.getLogger(__name__)


class Predictor:
    """Unified model predictor."""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model = None
        self.vectorizer = None
        self.label_encoder = {}
        self.model_type = None
        
        if model_path:
            self.load(model_path)
    
    def load(self, model_dir: Path) -> None:
        """Load model and artifacts from directory."""
        # Load model
        model_file = model_dir / "model.joblib"
        if model_file.exists():
            self.model = joblib.load(model_file)
            
        # Load vectorizer
        vectorizer_file = model_dir / "vectorizer.joblib"
        if vectorizer_file.exists():
            self.vectorizer = joblib.load(vectorizer_file)
            
        # Load label encoder
        encoder_file = model_dir / "label_encoder.json"
        if encoder_file.exists():
            import json
            with open(encoder_file, 'r') as f:
                self.label_encoder = json.load(f)
                
        logger.info(f"Loaded model from {model_dir}")
        
    def preprocess_text(self, text: str) -> np.ndarray:
        """Preprocess single text for prediction."""
        from ml.data.preprocessors import TextCleaner, PreprocessingConfig
        
        cleaner = TextCleaner(PreprocessingConfig())
        
        # Clean text
        cleaned = cleaner.clean(text)
        
        # Vectorize
        if self.vectorizer:
            X = self.vectorizer.transform([cleaned]).toarray()
            return X
            
        return np.array([[0]])
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Make prediction on text."""
        if self.model is None:
            raise ValueError("Model not loaded")
            
        # Preprocess
        X = self.preprocess_text(text)
        
        # Predict
        prediction = self.model.predict(X)[0]
        
        # Get probability if available
        probability = None
        if hasattr(self.model, 'predict_proba'):
            try:
                proba = self.model.predict_proba(X)[0]
                probability = {
                    'confidence': float(max(proba)),
                    'probabilities': proba.tolist()
                }
            except:
                pass
                
        # Map prediction to label
        reverse_encoder = {v: k for k, v in self.label_encoder.items()}
        label = reverse_encoder.get(prediction, str(prediction))
        
        return {
            'prediction': label,
            'prediction_code': int(prediction),
            'probability': probability,
            'model_type': self.model_type
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Make predictions on batch of texts."""
        return [self.predict(text) for text in texts]
    
    def explain(self, text: str) -> Dict[str, Any]:
        """Get feature importance explanation."""
        if not hasattr(self.model, 'coef_'):
            return {'error': 'Model does not support explanation'}
            
        # Preprocess
        X = self.preprocess_text(text)
        
        # Get feature importance
        importance = self.model.coef_[0]
        
        # Get feature names if available
        feature_names = []
        if self.vectorizer:
            feature_names = self.vectorizer.get_feature_names_out().tolist()
            
        # Get top features
        top_indices = np.argsort(np.abs(importance))[-10:][::-1]
        
        top_features = []
        for idx in top_indices:
            top_features.append({
                'feature': feature_names[idx] if feature_names else f"feature_{idx}",
                'importance': float(importance[idx])
            })
            
        return {
            'top_features': top_features,
            'prediction': self.predict(text)['prediction']
        }


class EnsemblePredictor:
    """Ensemble of multiple models for voting predictions."""
    
    def __init__(self):
        self.predictors = {}
        self.weights = {}
        
    def add_model(self, name: str, predictor: Predictor, weight: float = 1.0) -> None:
        """Add model to ensemble."""
        self.predictors[name] = predictor
        self.weights[name] = weight
        
    def predict(self, text: str) -> Dict[str, Any]:
        """Aggregate predictions from all models."""
        predictions = {}
        confidences = {}
        
        for name, predictor in self.predictors.items():
            result = predictor.predict(text)
            predictions[name] = result['prediction']
            if result['probability']:
                confidences[name] = result['probability']['confidence']
                
        # Majority voting
        from collections import Counter
        votes = Counter(predictions.values())
        final_prediction = votes.most_common(1)[0][0]
        
        # Average confidence
        avg_confidence = sum(confidences.values()) / len(confidences) if confidences else 0.5
        
        return {
            'prediction': final_prediction,
            'ensemble_size': len(self.predictors),
            'votes': dict(votes),
            'average_confidence': avg_confidence,
            'individual_predictions': predictions
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Batch ensemble predictions."""
        return [self.predict(text) for text in texts]