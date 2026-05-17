"""Multi-Dataset Independent Training System for Fake News Detection."""

__version__ = "1.0.0"

from .data.dataset_discovery import DatasetDiscovery, discover_datasets
from .data.dataset_loader import DatasetLoader
from .preprocessing.pipeline import TextPreprocessor, PreprocessingConfig
from .training.trainer import Trainer, ModelFactory
from .training.config import TrainingConfig
from .training.independent_pipeline import IndependentPipeline
from .evaluation.benchmark import ModelBenchmark
from .ensemble.voting import EnsembleSystem

__all__ = [
    'DatasetDiscovery',
    'discover_datasets',
    'DatasetLoader',
    'TextPreprocessor',
    'PreprocessingConfig',
    'Trainer',
    'TrainingConfig',
    'ModelFactory',
    'IndependentPipeline',
    'ModelBenchmark',
    'EnsembleSystem'
]