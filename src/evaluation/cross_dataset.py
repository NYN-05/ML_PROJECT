"""Cross-Dataset Evaluation Module - Test models on different datasets."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

from ..data.dataset_discovery import DatasetInfo
from ..data.dataset_loader import DatasetLoader
from ..preprocessing.pipeline import TextPreprocessor

logger = logging.getLogger("evaluation.cross_dataset")


@dataclass
class CrossDatasetResult:
    """Result of cross-dataset evaluation."""
    train_dataset: str
    test_dataset: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    samples: int
    
    def to_dict(self) -> Dict:
        return {
            "train_dataset": self.train_dataset,
            "test_dataset": self.test_dataset,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "samples": self.samples
        }


class CrossDatasetEvaluator:
    """Evaluate models across different datasets."""
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.models: Dict[str, Any] = {}
        self.preprocessors: Dict[str, TextPreprocessor] = {}
        self.label_encoders: Dict[str, Dict[int, str]] = {}
        self.dataset_configs: Dict[str, Dict] = {}
    
    def load_all_models(self) -> None:
        """Load all trained models and their preprocessors."""
        import joblib
        
        for dataset_dir in self.models_dir.iterdir():
            if dataset_dir.is_dir():
                dataset_name = dataset_dir.name
                
                model_path = dataset_dir / "model.pkl"
                preprocessor_path = dataset_dir / "preprocessing_config.json"
                config_path = dataset_dir / "pipeline_config.json"
                label_path = dataset_dir / "label_encoder.json"
                
                if model_path.exists():
                    self.models[dataset_name] = joblib.load(model_path)
                    logger.info(f"Loaded model: {dataset_name}")
                
                if preprocessor_path.exists():
                    with open(preprocessor_path, 'r') as f:
                        self.dataset_configs[dataset_name] = json.load(f)
                
                if label_path.exists():
                    with open(label_path, 'r') as f:
                        self.label_encoders[dataset_name] = json.load(f)
    
    def evaluate_on_dataset(
        self,
        train_dataset: str,
        test_dataset: str,
        test_data: pd.DataFrame
    ) -> CrossDatasetResult:
        """Evaluate a trained model on a different dataset."""
        
        if train_dataset not in self.models:
            raise ValueError(f"Model not found: {train_dataset}")
        
        model = self.models[train_dataset]
        
        config = self.dataset_configs.get(train_dataset, {})
        text_col = config.get("text_column", "text")
        target_col = config.get("target_column", "label")
        
        if text_col not in test_data.columns:
            logger.warning(f"Text column '{text_col}' not found in test dataset")
            text_col = None
            for col in test_data.columns:
                if test_data[col].dtype == object:
                    text_col = col
                    break
        
        if not text_col:
            raise ValueError("No text column found in test data")
        
        test_data = test_data.dropna(subset=[text_col])
        
        if target_col not in test_data.columns:
            if 'label' in test_data.columns:
                target_col = 'label'
            else:
                for col in test_data.columns:
                    if col.lower() in ['label', 'target', 'class']:
                        target_col = col
                        break
        
        texts = test_data[text_col].tolist()
        
        if train_dataset in self.label_encoders:
            le = self.label_encoders[train_dataset]
            le_inverted = {v: k for k, v in le.items()}
        
        logger.info(f"Evaluating {train_dataset} model on {test_dataset} data")
        
        X_test = self._transform_text(model, texts, train_dataset)
        
        y_true = None
        if target_col and target_col in test_data.columns:
            y_true = test_data[target_col].values
            if train_dataset in self.label_encoders:
                le = self.label_encoders[train_dataset]
                y_true = np.array([le.get(v, v) for v in y_true])
        
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_test)
        
        if y_true is None:
            return CrossDatasetResult(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                samples=len(y_pred)
            )
        
        y_true = self._normalize_labels(y_true)
        y_pred = self._normalize_labels(y_pred)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        logger.info(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return CrossDatasetResult(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            samples=len(y_pred)
        )
    
    def _transform_text(self, model, texts: List[str], dataset_name: str) -> np.ndarray:
        """Transform text using the model's vectorizer."""
        config = self.dataset_configs.get(dataset_name, {})
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        import joblib
        
        vectorizer_path = self.models_dir / dataset_name / "vectorizer.pkl"
        
        if vectorizer_path.exists():
            try:
                vectorizer = joblib.load(vectorizer_path)
                return vectorizer.transform(texts)
            except Exception as e:
                logger.warning(f"Could not load vectorizer: {e}")
        
        return None
    
    def _normalize_labels(self, y: np.ndarray) -> np.ndarray:
        """Normalize labels to binary."""
        unique = sorted(set(y))
        if len(unique) == 2:
            label_map = {unique[0]: 0, unique[1]: 1}
            return np.array([label_map.get(v, v) for v in y])
        return y
    
    def run_evaluation(
        self,
        test_datasets: Dict[str, pd.DataFrame]
    ) -> List[CrossDatasetResult]:
        """Run cross-dataset evaluation on all combinations."""
        
        if not self.models:
            self.load_all_models()
        
        results = []
        
        for train_name in self.models.keys():
            for test_name, test_df in test_datasets.items():
                if train_name == test_name:
                    continue
                
                try:
                    result = self.evaluate_on_dataset(train_name, test_name, test_df)
                    results.append(result)
                    logger.info(f"  {train_name} -> {test_name}: {result.accuracy:.4f}")
                except Exception as e:
                    logger.error(f"Failed evaluation {train_name} -> {test_name}: {e}")
        
        return results
    
    def save_results(self, results: List[CrossDatasetResult], output_path: Path) -> None:
        """Save cross-dataset results."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_data = [r.to_dict() for r in results]
        
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(results_data, f, indent=2)
        
        df = pd.DataFrame(results_data)
        df.to_csv(output_path.with_suffix('.csv'), index=False)
        
        self._create_evaluation_matrix(df, output_path.parent)
        
        logger.info(f"Saved cross-dataset results to: {output_path}")
    
    def _create_evaluation_matrix(self, results_df: pd.DataFrame, output_dir: Path) -> None:
        """Create evaluation matrix heatmap."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            pivot = results_df.pivot_table(
                values='accuracy',
                index='train_dataset',
                columns='test_dataset',
                aggfunc='mean'
            )
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax, vmin=0, vmax=1)
            ax.set_title('Cross-Dataset Evaluation Matrix (Accuracy)')
            ax.set_xlabel('Test Dataset')
            ax.set_ylabel('Train Dataset')
            
            fig.savefig(output_dir / 'cross_dataset_matrix.png', dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create matrix: {e}")


def run_cross_dataset_evaluation(
    models_dir: Path,
    test_datasets: Dict[str, pd.DataFrame],
    output_path: Path
) -> List[CrossDatasetResult]:
    """Convenience function to run cross-dataset evaluation."""
    evaluator = CrossDatasetEvaluator(models_dir)
    results = evaluator.run_evaluation(test_datasets)
    evaluator.save_results(results, output_path)
    return results