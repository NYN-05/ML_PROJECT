"""Training configuration for ML models."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ModelType(str, Enum):
    """Model types supported by the pipeline."""
    LOGISTIC_REGRESSION = "logistic_regression"
    LINEAR_SVM = "linear_svm"
    NAIVE_BAYES = "naive_bayes"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    model_type: str = "logistic_regression"
    test_size: float = 0.2
    validation_size: float = 0.1
    random_seed: int = 42

    # Logistic Regression / SVM
    C: float = 1.0
    max_iter: int = 1000
    class_weight: Optional[str] = "balanced"

    # Random Forest
    n_estimators: int = 100
    max_depth: Optional[int] = None

    # XGBoost
    n_estimators_xgb: int = 100
    learning_rate: float = 0.1
    max_depth_xgb: int = 6

    # Early stopping
    patience: int = 10