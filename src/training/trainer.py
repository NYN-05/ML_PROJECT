"""Training module for ML models with early stopping support."""

from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier

from src.training.config import TrainingConfig, ModelType

logger = logging.getLogger(__name__)


class TrainingResult:
    """Container for training results."""

    def __init__(
        self,
        dataset_name: str,
        model_type: str,
        train_accuracy: float,
        test_accuracy: float,
        precision: float,
        recall: float,
        f1_score: float,
        roc_auc: float,
        classification_report: str,
        confusion_matrix: list,
        train_samples: int,
        test_samples: int,
        training_time_seconds: float,
    ):
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.train_accuracy = train_accuracy
        self.test_accuracy = test_accuracy
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.roc_auc = roc_auc
        self.classification_report = classification_report
        self.confusion_matrix = confusion_matrix
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.training_time_seconds = training_time_seconds

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "model_type": self.model_type,
            "train_accuracy": self.train_accuracy,
            "test_accuracy": self.test_accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
            "classification_report": self.classification_report,
            "confusion_matrix": self.confusion_matrix,
            "train_samples": self.train_samples,
            "test_samples": self.test_samples,
            "training_time_seconds": self.training_time_seconds,
        }


class ModelFactory:
    """Factory for creating ML models."""

    @staticmethod
    def create(config: TrainingConfig) -> Any:
        """Create a model based on configuration."""

        if config.model_type == "logistic_regression":
            return LogisticRegression(
                C=config.C,
                max_iter=config.max_iter,
                class_weight=config.class_weight,
                random_state=config.random_seed,
                solver='lbfgs',
                n_jobs=-1
            )

        elif config.model_type == "linear_svm":
            return LinearSVC(
                C=config.C,
                max_iter=config.max_iter,
                class_weight=config.class_weight,
                random_state=config.random_seed,
                dual=True
            )

        elif config.model_type == "naive_bayes":
            return MultinomialNB(alpha=1.0)

        elif config.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                class_weight=config.class_weight,
                random_state=config.random_seed,
                n_jobs=-1
            )

        elif config.model_type == "xgboost":
            return XGBClassifier(
                n_estimators=config.n_estimators_xgb,
                learning_rate=config.learning_rate,
                max_depth=config.max_depth_xgb,
                random_state=config.random_seed,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=-1
            )

        else:
            raise ValueError(f"Unknown model type: {config.model_type}")


class Trainer:
    """Trainer class for ML models with early stopping support."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.label_encoder: Optional[Dict[str, int]] = None
        self.best_model = None
        self.best_score = 0
        self.patience_counter = 0

    def train(self, X: np.ndarray, y: np.ndarray, dataset_name: str) -> TrainingResult:
        """Train a model on the given data with early stopping."""
        start_time = time.time()

        logger.info(f"Training {self.config.model_type} on {dataset_name}")

        # Ensure y is a proper numpy array
        try:
            y_array = np.asarray(y)
            if y_array.ndim == 0:
                y_labels = np.array([y_array.item()])
            elif y_array.size == 0:
                raise ValueError("Labels array is empty")
            else:
                y_labels = y_array.flatten()
        except Exception as e:
            raise ValueError(f"Invalid labels: {type(y)}, {str(e)}")

        # Handle case where labels are strings (like 'FAKE', 'REAL')
        unique_labels = np.unique(y_labels)
        if len(unique_labels) == 1:
            raise ValueError(f"Dataset has only one class: {unique_labels[0]}")

        if len(unique_labels) > 2:
            logger.warning(f"Dataset has {len(unique_labels)} classes, using first 2")

        labels_count = y_labels.shape[0] if hasattr(y_labels, 'shape') else len(y_labels) if hasattr(y_labels, '__len__') else 1
        logger.info(f"  Data shape: {X.shape}, Labels: {labels_count}, Classes: {unique_labels}")

        y_encoded, self.label_encoder = self._encode_labels(y_labels)

        # Ensure y_encoded is 1D
        y_encoded = np.asarray(y_encoded).flatten()
        if y_encoded.ndim != 1:
            raise ValueError(f"y_encoded should be 1D but got shape {y_encoded.shape}")

        test_size = self.config.test_size
        val_size = self.config.validation_size

        if val_size > 0:
            test_val_size = test_size + val_size
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y_encoded, test_size=test_val_size, random_state=self.config.random_seed, stratify=y_encoded
            )
            X_val_size = val_size / test_val_size
            X_test, X_val, y_test, y_val = train_test_split(
                X_temp, y_temp, test_size=1 - X_val_size, random_state=self.config.random_seed, stratify=y_temp
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=self.config.random_seed, stratify=y_encoded
            )
            X_val = None
            y_val = None

        # Safe length calculation
        train_count = int(y_train.shape[0]) if hasattr(y_train, 'shape') and y_train.ndim > 0 else int(y_train)
        test_count = int(y_test.shape[0]) if hasattr(y_test, 'shape') and y_test.ndim > 0 else int(y_test)
        val_count = int(y_val.shape[0]) if y_val is not None and hasattr(y_val, 'shape') and y_val.ndim > 0 else (int(y_val) if y_val is not None else 0)
        
        logger.info(f"  Train: {train_count}, Test: {test_count}, Val: {val_count}")

        # Train with early stopping if validation set is available
        if X_val is not None and self.config.patience > 0:
            self.model, best_score = self._train_with_early_stopping(X_train, y_train, X_val, y_val)
            logger.info(f"  Early stopping: best validation score = {best_score:.4f}")
        else:
            self.model = ModelFactory.create(self.config)

            if self.config.model_type == "xgboost" and X_val is not None:
                self.model.set_params(
                    early_stopping_rounds=self.config.patience,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )

            self.model.fit(X_train, y_train)
            best_score = 0

        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)

        test_pred = self.model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_pred)

        precision = precision_score(y_test, test_pred, average='binary', zero_division=0)
        recall = recall_score(y_test, test_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, test_pred, average='binary', zero_division=0)

        try:
            if hasattr(self.model, 'predict_proba'):
                y_proba = self.model.predict_proba(X_test)[:, 1]
            else:
                y_proba = self.decision_function(X_test)
            roc_auc = roc_auc_score(y_test, y_proba)
        except Exception:
            roc_auc = 0.0

        class_report = classification_report(y_test, test_pred, target_names=list(self.label_encoder.keys()))
        conf_matrix = confusion_matrix(y_test, test_pred).tolist()

        training_time = time.time() - start_time

        logger.info(f"  Training complete in {training_time:.2f}s")
        logger.info(f"  Test Accuracy: {test_accuracy:.4f}, F1: {f1:.4f}")

        return TrainingResult(
            dataset_name=dataset_name,
            model_type=self.config.model_type,
            train_accuracy=train_accuracy,
            test_accuracy=test_accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            classification_report=class_report,
            confusion_matrix=conf_matrix,
            train_samples=int(y_train.shape[0]) if hasattr(y_train, 'shape') and y_train.ndim > 0 else int(y_train),
            test_samples=int(y_test.shape[0]) if hasattr(y_test, 'shape') and y_test.ndim > 0 else int(y_test),
            training_time_seconds=training_time
        )

    def decision_function(self, X):
        """Compute decision function for LinearSVC."""
        return self.model.decision_function(X)

    def _encode_labels(self, y: np.ndarray) -> tuple:
        """Encode string labels to integers."""
        unique_labels = sorted(list(set(y)))
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        y_encoded = np.array([label_map[label] for label in y])
        return y_encoded, label_map

    def _train_with_early_stopping(self, X_train, y_train, X_val, y_val):
        """Train model with early stopping."""
        from sklearn.model_selection import cross_val_score

        patience = self.config.patience
        best_model = None
        best_score = 0
        no_improvement = 0

        if self.config.model_type == "xgboost":
            model = XGBClassifier(
                n_estimators=self.config.n_estimators_xgb,
                learning_rate=self.config.learning_rate,
                max_depth=self.config.max_depth_xgb,
                random_state=self.config.random_seed,
                use_label_encoder=False,
                eval_metric='logloss',
                early_stopping_rounds=patience,
                n_jobs=-1
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            return model, 0

        # For other models, use cross-validation
        model = ModelFactory.create(self.config)

        try:
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
            best_score = scores.mean()
        except:
            best_score = 0

        model.fit(X_train, y_train)
        return model, best_score

    def save_model(self, path: Path) -> None:
        """Save the trained model."""
        import joblib
        joblib.dump(self.model, path / "model.joblib")

        if self.label_encoder:
            import json
            with open(path / "label_encoder.json", 'w') as f:
                json.dump(self.label_encoder, f)

    def load_model(self, path: Path) -> None:
        """Load a trained model."""
        import joblib
        self.model = joblib.load(path / "model.joblib")

        import json
        with open(path / "label_encoder.json", 'r') as f:
            self.label_encoder = json.load(f)