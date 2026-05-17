"""Ensemble Learning Module - Combine multiple trained models."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import VotingClassifier

logger = logging.getLogger("ensemble")


@dataclass
class EnsembleResult:
    """Result of ensemble prediction."""
    predictions: np.ndarray
    probabilities: Optional[np.ndarray]
    method: str
    individual_predictions: Dict[str, np.ndarray]


class EnsembleSystem:
    """Ensemble system for combining multiple models."""
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.models: Dict[str, Any] = {}
        self.label_encoders: Dict[str, Dict[int, str]] = {}
    
    def load_models(self, model_names: List[str]) -> None:
        """Load multiple trained models."""
        import joblib
        
        for name in model_names:
            model_path = self.models_dir / name / "model.pkl"
            label_path = self.models_dir / name / "label_encoder.json"
            
            if model_path.exists():
                self.models[name] = joblib.load(model_path)
                
                if label_path.exists():
                    with open(label_path, 'r') as f:
                        self.label_encoders[name] = json.load(f)
                
                logger.info(f"Loaded model: {name}")
            else:
                logger.warning(f"Model not found: {model_path}")
    
    def predict_hard_voting(self, X: np.ndarray) -> EnsembleResult:
        """Hard voting - majority class prediction."""
        predictions_dict = {}
        
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions_dict[name] = pred
        
        all_preds = np.array(list(predictions_dict.values()))
        
        from scipy import stats
        final_pred = stats.mode(all_preds, axis=0, keepdims=False)[0]
        
        return EnsembleResult(
            predictions=final_pred,
            probabilities=None,
            method="hard_voting",
            individual_predictions=predictions_dict
        )
    
    def predict_soft_voting(self, X: np.ndarray) -> EnsembleResult:
        """Soft voting - average probabilities."""
        predictions_dict = {}
        probabilities_dict = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                predictions_dict[name] = np.argmax(proba, axis=1)
                probabilities_dict[name] = proba
            else:
                pred = model.predict(X)
                predictions_dict[name] = pred
        
        if not probabilities_dict:
            logger.warning("No models support probability prediction, falling back to hard voting")
            return self.predict_hard_voting(X)
        
        avg_proba = np.mean(list(probabilities_dict.values()), axis=0)
        final_pred = np.argmax(avg_proba, axis=1)
        
        return EnsembleResult(
            predictions=final_pred,
            probabilities=avg_proba,
            method="soft_voting",
            individual_predictions=predictions_dict
        )
    
    def predict_weighted_voting(self, X: np.ndarray, weights: Dict[str, float]) -> EnsembleResult:
        """Weighted voting - weighted average of probabilities."""
        predictions_dict = {}
        probabilities_dict = {}
        
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        
        for name, model in self.models.items():
            if name in weights:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    predictions_dict[name] = np.argmax(proba, axis=1)
                    probabilities_dict[name] = proba
                else:
                    pred = model.predict(X)
                    predictions_dict[name] = pred
        
        if not probabilities_dict:
            return self.predict_hard_voting(X)
        
        weighted_proba = np.zeros_like(list(probabilities_dict.values())[0])
        
        for name, proba in probabilities_dict.items():
            weight = normalized_weights.get(name, 1.0 / len(probabilities_dict))
            weighted_proba += weight * proba
        
        final_pred = np.argmax(weighted_proba, axis=1)
        
        return EnsembleResult(
            predictions=final_pred,
            probabilities=weighted_proba,
            method="weighted_voting",
            individual_predictions=predictions_dict
        )
    
    def evaluate_ensemble(self, X: np.ndarray, y_true: np.ndarray, method: str = "soft_voting") -> Dict:
        """Evaluate ensemble performance."""
        if method == "hard_voting":
            result = self.predict_hard_voting(X)
        elif method == "weighted_voting":
            result = self.predict_weighted_voting(X, {})
        else:
            result = self.predict_soft_voting(X)
        
        accuracy = accuracy_score(y_true, result.predictions)
        f1 = f1_score(y_true, result.predictions, average='binary')
        
        report = classification_report(y_true, result.predictions, output_dict=True)
        
        return {
            "method": method,
            "accuracy": accuracy,
            "f1_score": f1,
            "classification_report": report,
            "individual_predictions": {
                name: accuracy_score(y_true, pred) 
                for name, pred in result.individual_predictions.items()
            }
        }
    
    def save_ensemble_config(self, output_path: Path, config: Dict) -> None:
        """Save ensemble configuration."""
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / "ensemble_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved ensemble config to: {output_path}")


class StackingClassifier:
    """Stacking classifier - meta-learner on top of base models."""
    
    def __init__(self, base_models: Dict[str, Any], meta_model: Any):
        self.base_models = base_models
        self.meta_model = meta_model
        self.meta_features: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StackingClassifier':
        """Fit the stacking classifier."""
        meta_features = []
        
        for name, model in self.base_models.items():
            model.fit(X, y)
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                meta_features.append(proba)
            else:
                pred = model.predict(X).reshape(-1, 1)
                meta_features.append(pred)
        
        self.meta_features = np.hstack(meta_features)
        
        self.meta_model.fit(self.meta_features, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using stacking."""
        meta_features = []
        
        for name, model in self.base_models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                meta_features.append(proba)
            else:
                pred = model.predict(X).reshape(-1, 1)
                meta_features.append(pred)
        
        meta_features = np.hstack(meta_features)
        
        return self.meta_model.predict(meta_features)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        meta_features = []
        
        for name, model in self.base_models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                meta_features.append(proba)
            else:
                pred = model.predict(X).reshape(-1, 1)
                meta_features.append(pred)
        
        meta_features = np.hstack(meta_features)
        
        if hasattr(self.meta_model, 'predict_proba'):
            return self.meta_model.predict_proba(meta_features)
        return None


def create_ensemble(
    models_dir: Path,
    model_names: List[str],
    method: str = "soft_voting",
    weights: Optional[Dict[str, float]] = None
) -> EnsembleSystem:
    """Create an ensemble system."""
    ensemble = EnsembleSystem(models_dir)
    ensemble.load_models(model_names)
    return ensemble