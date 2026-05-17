"""Unified model training system."""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Training results."""
    model_type: str
    train_accuracy: float
    test_accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    classification_report: str
    confusion_matrix: list
    train_samples: int
    test_samples: int
    training_time: float
    epochs: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_type': self.model_type,
            'train_accuracy': self.train_accuracy,
            'test_accuracy': self.test_accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'roc_auc': self.roc_auc,
            'classification_report': self.classification_report,
            'confusion_matrix': self.confusion_matrix,
            'train_samples': self.train_samples,
            'test_samples': self.test_samples,
            'training_time': self.training_time,
            'epochs': self.epochs,
        }


class ModelFactory:
    """Factory for creating models."""
    
    MODELS = {
        'logistic_regression': LogisticRegression,
        'linear_svm': LinearSVC,
        'naive_bayes': MultinomialNB,
        'random_forest': RandomForestClassifier,
    }
    
    @staticmethod
    def create(model_type: str, **kwargs):
        """Create model by type."""
        if model_type not in ModelFactory.MODELS:
            raise ValueError(f"Unknown model type: {model_type}")
            
        model_class = ModelFactory.MODELS[model_type]
        
        if model_type == 'naive_bayes':
            return model_class(alpha=1.0)
        elif model_type == 'random_forest':
            return model_class(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth'),
                random_state=kwargs.get('random_seed', 42),
                n_jobs=-1
            )
        else:
            return model_class(
                C=kwargs.get('C', 1.0),
                max_iter=kwargs.get('max_iter', 1000),
                random_state=kwargs.get('random_seed', 42),
                n_jobs=-1
            )


class ModelTrainer:
    """Unified model trainer for sklearn models."""
    
    def __init__(self, model_type: str = 'logistic_regression', 
                 test_size: float = 0.2, random_seed: int = 42,
                 patience: int = 5):
        self.model_type = model_type
        self.test_size = test_size
        self.random_seed = random_seed
        self.patience = patience
        self.model = None
        self.label_encoder = {}
        
    def train(self, X: np.ndarray, y: np.ndarray, 
              dataset_name: str = "unknown") -> TrainingResult:
        """Train model on data."""
        start_time = time.time()
        
        logger.info(f"Training {self.model_type} on {dataset_name}")
        
        # Encode labels
        y_encoded, self.label_encoder = self._encode_labels(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=self.test_size, 
            random_state=self.random_seed,
            stratify=y_encoded
        )
        
        logger.info(f"  Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Create and train model
        self.model = ModelFactory.create(
            self.model_type, 
            random_seed=self.random_seed
        )
        
        self.model.fit(X_train, y_train)
        
        # Predictions
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Metrics
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        precision = precision_score(y_test, test_pred, average='binary', zero_division=0)
        recall = recall_score(y_test, test_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, test_pred, average='binary', zero_division=0)
        
        # ROC-AUC
        try:
            if hasattr(self.model, 'predict_proba'):
                y_proba = self.model.predict_proba(X_test)[:, 1]
            else:
                y_proba = self.model.decision_function(X_test)
            roc_auc = roc_auc_score(y_test, y_proba)
        except:
            roc_auc = 0.0
            
        # Report
        class_report = classification_report(
            y_test, test_pred, 
            target_names=list(self.label_encoder.keys())
        )
        conf_matrix = confusion_matrix(y_test, test_pred).tolist()
        
        training_time = time.time() - start_time
        
        logger.info(f"  Accuracy: {test_accuracy:.4f}, F1: {f1:.4f}, Time: {training_time:.2f}s")
        
        return TrainingResult(
            model_type=self.model_type,
            train_accuracy=train_accuracy,
            test_accuracy=test_accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            classification_report=class_report,
            confusion_matrix=conf_matrix,
            train_samples=len(y_train),
            test_samples=len(y_test),
            training_time=training_time
        )
    
    def _encode_labels(self, y: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Encode string labels to integers."""
        unique_labels = sorted(list(set(y)))
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        y_encoded = np.array([label_map[str(label)] for label in y])
        return y_encoded, label_map
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if self.model is None:
            raise ValueError("Model not trained")
            
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
            
        return None
    
    def save(self, path: Path) -> None:
        """Save trained model."""
        import joblib
        
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "model.joblib")
        
        import json
        with open(path / "label_encoder.json", 'w') as f:
            json.dump(self.label_encoder, f)
            
        logger.info(f"Model saved to {path}")
        
    def load(self, path: Path) -> None:
        """Load trained model."""
        import joblib
        import json
        
        self.model = joblib.load(path / "model.joblib")
        
        with open(path / "label_encoder.json", 'r') as f:
            self.label_encoder = json.load(f)
            
        logger.info(f"Model loaded from {path}")


class ModelRegistry:
    """Registry for managing multiple trained models."""
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        
    def register(self, name: str, model: ModelTrainer, 
                result: TrainingResult) -> None:
        """Register a trained model."""
        save_path = self.models_dir / name
        model.save(save_path)
        
        self.models[name] = {
            'model': model,
            'result': result,
            'path': save_path
        }
        
    def get(self, name: str) -> Optional[ModelTrainer]:
        """Get model by name."""
        if name in self.models:
            return self.models[name]['model']
            
        # Try loading from disk
        load_path = self.models_dir / name
        if load_path.exists():
            model = ModelTrainer(self.model_type)
            model.load(load_path)
            return model
            
        return None
    
    def list_models(self) -> list:
        """List all registered models."""
        return list(self.models.keys())
    
    def get_best(self, metric: str = 'test_accuracy') -> Optional[str]:
        """Get best model by metric."""
        if not self.models:
            return None
            
        best_name = None
        best_score = 0
        
        for name, data in self.models.items():
            result = data['result']
            score = getattr(result, metric, 0)
            
            if score > best_score:
                best_score = score
                best_name = name
                
        return best_name