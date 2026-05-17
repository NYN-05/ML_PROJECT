"""Independent Dataset Pipeline - One dataset = One pipeline = One model."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from ..data.dataset_discovery import DatasetInfo
from ..data.dataset_loader import DatasetLoader, LoadedDataset
from ..preprocessing.pipeline import TextPreprocessor, PreprocessingConfig
from ..training.trainer import Trainer, TrainingResult
from ..training.config import TrainingConfig, ModelType

logger = logging.getLogger("training.pipeline")


@dataclass
class PipelineResult:
    """Result of a complete dataset pipeline."""
    dataset_name: str
    dataset_info: DatasetInfo
    success: bool
    error_message: Optional[str]
    preprocessing_config: Dict
    training_result: Optional[TrainingResult]
    output_path: Path
    execution_time_seconds: float


class IndependentPipeline:
    """
    Independent pipeline for a single dataset.
    One dataset = One pipeline = One model
    """
    
    def __init__(
        self,
        output_dir: Path,
        preprocessing_config: Optional[PreprocessingConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        text_column: Optional[str] = None,
        target_column: Optional[str] = None
    ):
        self.output_dir = output_dir
        self.preprocessing_config = preprocessing_config or PreprocessingConfig()
        self.training_config = training_config or TrainingConfig()
        self.text_column = text_column
        self.target_column = target_column
        
        self.preprocessor: Optional[TextPreprocessor] = None
        self.trainer: Optional[Trainer] = None
        self.dataset_info: Optional[DatasetInfo] = None
    
    def run(self, dataset_info: DatasetInfo) -> PipelineResult:
        """Run the complete pipeline for a dataset."""
        start_time = time.time()
        
        dataset_name = dataset_info.name
        logger.info(f"{'='*60}")
        logger.info(f"Starting pipeline for dataset: {dataset_name}")
        logger.info(f"{'='*60}")
        
        try:
            self.dataset_info = dataset_info
            
            output_path = self.output_dir / dataset_name
            output_path.mkdir(parents=True, exist_ok=True)
            
            loaded_data = self._load_data(dataset_info)
            
            if loaded_data.train is None or loaded_data.train.empty:
                raise ValueError("No training data loaded")
            
            train_df, test_df = self._prepare_data(loaded_data)
            
            X_train, y_train = self._preprocess_train(train_df)
            
            X_test = None
            y_test = None
            if test_df is not None and not test_df.empty:
                X_test = self._preprocess_test(test_df)
                y_test = test_df[self.target_column].values if self.target_column in test_df.columns else None
            
            training_result = self._train_model(X_train, y_train, dataset_name)
            
            self._save_results(training_result, output_path)
            
            execution_time = time.time() - start_time
            
            logger.info(f"Pipeline completed in {execution_time:.2f}s")
            logger.info(f"Results saved to: {output_path}")
            
            return PipelineResult(
                dataset_name=dataset_name,
                dataset_info=dataset_info,
                success=True,
                error_message=None,
                preprocessing_config=self._get_preprocessing_config_dict(),
                training_result=training_result,
                output_path=output_path,
                execution_time_seconds=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Pipeline failed for {dataset_name}: {str(e)}")
            
            return PipelineResult(
                dataset_name=dataset_name,
                dataset_info=dataset_info,
                success=False,
                error_message=str(e),
                preprocessing_config={},
                training_result=None,
                output_path=Path(),
                execution_time_seconds=execution_time
            )
    
    def _load_data(self, dataset_info: DatasetInfo) -> LoadedDataset:
        """Load dataset."""
        loader = DatasetLoader(random_seed=self.training_config.random_seed)
        return loader.load(dataset_info)
    
    def _prepare_data(self, loaded_data: LoadedDataset) -> tuple:
        """Prepare dataframes for training."""
        text_col = self.text_column
        target_col = self.target_column
        
        if not text_col and loaded_data.metadata.text_columns:
            text_col = loaded_data.metadata.text_columns[0]
        
        if not target_col:
            target_col = loaded_data.metadata.target_column
        
        if not text_col:
            text_col = 'text'
            if text_col not in loaded_data.train.columns:
                for col in loaded_data.train.columns:
                    if loaded_data.train[col].dtype == object:
                        text_col = col
                        break
        
        if not target_col:
            for col in loaded_data.train.columns:
                if col.lower() in ['label', 'target', 'class']:
                    target_col = col
                    break
        
        self.text_column = text_col
        self.target_column = target_col
        
        train_df = loaded_data.train.copy() if loaded_data.train is not None else pd.DataFrame()
        test_df = loaded_data.test.copy() if loaded_data.test is not None else None
        
        logger.info(f"Using text column: {text_col}, target column: {target_col}")
        
        if target_col and target_col in train_df.columns:
            train_df = train_df.dropna(subset=[target_col])
        
        return train_df, test_df
    
    def _preprocess_train(self, train_df: pd.DataFrame) -> tuple:
        """Preprocess training data."""
        print(f"DEBUG _preprocess_train: train_df.shape={train_df.shape}, target_col={self.target_column}")
        
        self.preprocessor = TextPreprocessor(self.preprocessing_config)
        
        train_df = self.preprocessor.prepare_dataframe(train_df, self.text_column, self.target_column)
        
        X, y = self.preprocessor.fit_transform_dataframe(train_df)
        
        print(f"DEBUG after fit_transform: X.shape={X.shape if hasattr(X, 'shape') else 'N/A'}, type(y)={type(y)}, y={y}")
        
        if y is not None:
            y = self._normalize_labels(y)
            print(f"DEBUG after normalize: type(y)={type(y)}, y shape={y.shape if hasattr(y, 'shape') else 'N/A'}")
        
        logger.info(f"Preprocessed training data: {X.shape}")
        
        return X, y
    
    def _preprocess_test(self, test_df: pd.DataFrame) -> np.ndarray:
        """Preprocess test data."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not initialized")
        
        test_df = test_df.copy()
        
        if self.text_column in test_df.columns:
            test_df = test_df.dropna(subset=[self.text_column])
        
        return self.preprocessor.transform_dataframe(test_df)
    
    def _normalize_labels(self, y: np.ndarray) -> np.ndarray:
        """Normalize labels to binary (0/1)."""
        try:
            y_arr = np.asarray(y).flatten()
        except:
            return y
        
        if y_arr.size == 0:
            return y
        
        unique = sorted(set(y_arr))
        if len(unique) == 2:
            label_map = {unique[0]: 0, unique[1]: 1}
            return np.array([label_map.get(v, v) for v in y_arr])
        return y_arr
    
    def _train_model(self, X: np.ndarray, y: np.ndarray, dataset_name: str) -> TrainingResult:
        """Train the model."""
        print(f"DEBUG _train_model: X.shape={X.shape}, type(y)={type(y)}, y={y}")
        self.trainer = Trainer(self.training_config)
        return self.trainer.train(X, y, dataset_name)
    
    def _save_results(self, result: TrainingResult, output_path: Path) -> None:
        """Save all results and artifacts."""
        metrics = {
            "dataset_name": result.dataset_name,
            "model_type": result.model_type,
            "train_accuracy": result.train_accuracy,
            "test_accuracy": result.test_accuracy,
            "precision": result.precision,
            "recall": result.recall,
            "f1_score": result.f1_score,
            "roc_auc": result.roc_auc,
            "train_samples": result.train_samples,
            "test_samples": result.test_samples,
            "training_time_seconds": result.training_time_seconds
        }
        
        with open(output_path / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        with open(output_path / "classification_report.txt", 'w') as f:
            f.write(result.classification_report)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            np.array(result.confusion_matrix),
            annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'],
            ax=ax
        )
        ax.set_title(f"Confusion Matrix - {result.dataset_name}")
        fig.savefig(output_path / "confusion_matrix.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        if self.trainer:
            self.trainer.save(output_path)
        
        if self.preprocessor:
            self.preprocessor.save(output_path)
        
        config_info = {
            "preprocessing": asdict(self.preprocessing_config),
            "training": asdict(self.training_config),
            "text_column": self.text_column,
            "target_column": self.target_column
        }
        
        with open(output_path / "pipeline_config.json", 'w') as f:
            json.dump(config_info, f, indent=2, default=str)
        
        logger.info(f"Saved all artifacts to: {output_path}")
    
    def _get_preprocessing_config_dict(self) -> Dict:
        """Get preprocessing config as dict."""
        return asdict(self.preprocessing_config)


def asdict(obj):
    """Convert dataclass to dict."""
    if hasattr(obj, '__dataclass_fields__'):
        return {k: asdict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, dict):
        return {k: asdict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [asdict(v) for v in obj]
    else:
        return obj