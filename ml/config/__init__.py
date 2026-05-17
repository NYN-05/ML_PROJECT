"""Unified configuration for ML pipeline."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any


class ModelType(str, Enum):
    """Supported model types."""
    LSTM = "lstm"
    LOGISTIC_REGRESSION = "logistic_regression"
    LINEAR_SVM = "linear_svm"
    NAIVE_BAYES = "naive_bayes"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model_type: ModelType = ModelType.LSTM
    embedding_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    vocab_size: int = 10000
    max_seq_length: int = 256
    
    # sklearn model params
    n_estimators: int = 100
    max_depth: Optional[int] = None
    C: float = 1.0
    max_iter: int = 1000


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    
    # sklearn params
    test_size: float = 0.2
    validation_size: float = 0.1
    random_seed: int = 42
    
    # Early stopping
    patience: int = 5
    min_delta: float = 0.001
    
    # Class balancing
    use_class_weights: bool = True


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    data_dir: Path = Path("dataset")
    output_dir: Path = Path("artifacts")
    
    # Supported datasets
    datasets: List[str] = field(default_factory=lambda: [
        "Fake News Detection Dataset",
        "Fake or Real News", 
        "News Detection (Fake or Real) Dataset",
        "LIAR Fake news dataset"
    ])
    
    text_columns: List[str] = field(default_factory=lambda: ["title", "text"])
    target_column: str = "label"
    
    # Preprocessing
    min_text_length: int = 10
    max_text_length: int = 10000
    remove_duplicates: bool = True
    
    # Caching
    cache_enabled: bool = True
    cache_dir: Path = Path("artifacts/cache")


@dataclass  
class PipelineConfig:
    """Complete pipeline configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    
    # System
    device: str = "cuda"
    num_workers: int = 4
    verbose: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model": {
                "type": self.model.model_type.value,
                "embedding_dim": self.model.embedding_dim,
                "hidden_dim": self.model.hidden_dim,
                "num_layers": self.model.num_layers,
                "dropout": self.model.dropout,
                "vocab_size": self.model.vocab_size,
                "max_seq_length": self.model.max_seq_length,
            },
            "training": {
                "epochs": self.training.epochs,
                "batch_size": self.training.batch_size,
                "learning_rate": self.training.learning_rate,
                "test_size": self.training.test_size,
                "patience": self.training.patience,
            },
            "dataset": {
                "data_dir": str(self.dataset.data_dir),
                "output_dir": str(self.dataset.output_dir),
            }
        }