"""Unified ML Pipeline for Fake News Detection."""

__version__ = "2.0.0"
__author__ = "ML Team"

from ml.config import ModelConfig, TrainingConfig, DatasetConfig
from ml.data.loaders import DatasetLoader
from ml.models.trainers import ModelTrainer
from ml.models.evaluators import ModelEvaluator

__all__ = [
    'ModelConfig',
    'TrainingConfig', 
    'DatasetConfig',
    'DatasetLoader',
    'ModelTrainer',
    'ModelEvaluator',
]