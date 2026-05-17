"""Configuration models for the ML pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class ProjectPaths(BaseModel):
    """Project path configuration.

    Attributes:
        root_dir: Repository root directory.
        dataset_dir: Root dataset directory.
        processed_dir: Processed dataset output directory.
        models_dir: Trained model output directory.
        eval_dir: Evaluation artifact directory.
    """

    root_dir: Path
    dataset_dir: Path
    processed_dir: Path
    models_dir: Path
    eval_dir: Path


class DataConfig(BaseModel):
    """Dataset configuration.

    Attributes:
        primary_dataset_dir: Directory containing Fake.csv and True.csv.
        fake_label: Label for fake news rows.
        real_label: Label for real news rows.
        validation_split: Validation ratio.
        test_split: Test ratio.
        random_seed: Reproducibility seed.
    """

    primary_dataset_dir: Optional[Path] = None
    fake_label: str = "FAKE"
    real_label: str = "REAL"
    validation_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42


class PreprocessConfig(BaseModel):
    """Text preprocessing configuration.

    Attributes:
        min_tokens: Minimum token count for a valid input.
        max_seq_len_min: Minimum sequence length cap.
        max_seq_len_max: Maximum sequence length cap.
        seq_len_percentile: Percentile used to compute max sequence length.
    """

    min_tokens: int = 15
    max_seq_len_min: int = 300
    max_seq_len_max: int = 500
    seq_len_percentile: int = 95


class TokenizerConfig(BaseModel):
    """Tokenizer configuration.

    Attributes:
        vocab_size: Maximum vocabulary size.
        oov_token: Out-of-vocabulary token.
    """

    vocab_size: int = 30000
    oov_token: str = "<OOV>"


class ModelConfig(BaseModel):
    """Model architecture configuration.

    Attributes:
        embedding_dim: Embedding vector size.
        lstm_units: LSTM unit count.
        dropout: Dropout ratio.
        recurrent_dropout: Recurrent dropout ratio.
        num_filters: Number of Conv1D filters.
        kernel_size: Conv1D kernel size.
    """

    embedding_dim: int = 128
    lstm_units: int = 96
    dropout: float = 0.3
    recurrent_dropout: float = 0.0
    num_filters: int = 128
    kernel_size: int = 5


class TrainingConfig(BaseModel):
    """Training hyperparameters.

    Attributes:
        batch_size: Training batch size.
        epochs: Maximum epoch count.
        learning_rate: Optimizer learning rate.
        early_stopping_patience: Early stopping patience.
    """

    batch_size: int = 128
    epochs: int = 12
    learning_rate: float = 0.001
    early_stopping_patience: int = 3


class InferenceConfig(BaseModel):
    """Inference configuration.

    Attributes:
        uncertainty_threshold: Probability band for UNCERTAIN output.
        high_confidence_threshold: High confidence threshold.
        medium_confidence_threshold: Medium confidence threshold.
    """

    uncertainty_threshold: float = 0.6
    high_confidence_threshold: float = 0.9
    medium_confidence_threshold: float = 0.7


class AppConfig(BaseModel):
    """Aggregate configuration for the ML pipeline."""

    paths: ProjectPaths
    data: DataConfig = Field(default_factory=DataConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)


def build_default_config(root_dir: Optional[Path] = None) -> AppConfig:
    """Build default configuration using repository-relative paths.

    Args:
        root_dir: Optional repository root directory override.

    Returns:
        Fully populated configuration object.
    """
    repo_root = root_dir or Path(__file__).resolve().parents[1]
    dataset_dir = repo_root / "dataset"
    processed_dir = dataset_dir / "processed"
    models_dir = repo_root / "models"
    eval_dir = models_dir / "eval"
    paths = ProjectPaths(
        root_dir=repo_root,
        dataset_dir=dataset_dir,
        processed_dir=processed_dir,
        models_dir=models_dir,
        eval_dir=eval_dir,
    )
    return AppConfig(paths=paths)
