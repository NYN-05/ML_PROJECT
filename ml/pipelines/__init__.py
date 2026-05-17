"""Unified training pipeline."""

import logging
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np

from ml.config import ModelConfig, TrainingConfig, DatasetConfig, PipelineConfig
from ml.data.loaders import DatasetLoader, DatasetInfo
from ml.data.preprocessors import TextPreprocessor, PreprocessingConfig, TextCleaner
from ml.models.trainers import ModelTrainer, TrainingResult, ModelRegistry
from ml.models.evaluators import ModelEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete pipeline result."""
    dataset_name: str
    model_name: str
    training_result: TrainingResult
    evaluation_result: Optional[EvaluationResult]
    artifacts_dir: Path
    training_time: float


class TrainingPipeline:
    """Unified training pipeline."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.dataset_loader = DatasetLoader()
        self.preprocessor = TextPreprocessor()
        self.evaluator = ModelEvaluator()
        self.registry = None
        
    def run(self, data_dir: Path, output_dir: Path, 
            dataset_names: Optional[List[str]] = None) -> List[PipelineResult]:
        """Run complete training pipeline."""
        start_time = time.time()
        
        # Discover datasets
        datasets = self.dataset_loader.discover(data_dir)
        logger.info(f"Discovered {len(datasets)} datasets")
        
        # Filter specific datasets if requested
        if dataset_names:
            datasets = [d for d in datasets if d.name in dataset_names]
            
        results = []
        
        # Train on each dataset
        for dataset_info in datasets:
            logger.info(f"Processing: {dataset_info.name}")
            
            result = self._train_single_dataset(dataset_info, output_dir)
            if result:
                results.append(result)
                
        total_time = time.time() - start_time
        logger.info(f"Pipeline complete in {total_time:.2f}s - {len(results)} models trained")
        
        return results
    
    def _train_single_dataset(self, dataset_info: DatasetInfo, 
                              output_dir: Path) -> Optional[PipelineResult]:
        """Train on single dataset."""
        try:
            # Load data
            df = self.dataset_loader.load(dataset_info)
            
            if df.empty:
                logger.warning(f"Empty dataset: {dataset_info.name}")
                return None
                
            # Find text and target columns
            text_col = dataset_info.text_columns[0] if dataset_info.text_columns else 'text'
            target_col = dataset_info.target_column or 'label'
            
            # Prepare data
            X_texts = df[text_col].fillna('').astype(str).tolist()
            y = df[target_col].values
            
            logger.info(f"  Data: {len(X_texts)} samples")
            
            # Preprocess
            X = self.preprocessor.preprocess(X_texts)
            
            # Train
            model_name = self.config.model.model_type.value
            trainer = ModelTrainer(
                model_type=model_name,
                test_size=self.config.training.test_size,
                random_seed=self.config.training.random_seed
            )
            
            train_result = trainer.train(X, y, dataset_info.name)
            
            # Save artifacts
            artifacts_dir = output_dir / dataset_info.name
            trainer.save(artifacts_dir)
            self.preprocessor.save_vectorizer(str(artifacts_dir / "vectorizer.joblib"))
            
            # Save metrics
            metrics_file = artifacts_dir / "metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(train_result.to_dict(), f, indent=2)
                
            return PipelineResult(
                dataset_name=dataset_info.name,
                model_name=model_name,
                training_result=train_result,
                evaluation_result=None,
                artifacts_dir=artifacts_dir,
                training_time=train_result.training_time
            )
            
        except Exception as e:
            logger.error(f"Error training on {dataset_info.name}: {e}")
            return None


class EvaluationPipeline:
    """Unified evaluation pipeline."""
    
    def __init__(self):
        self.evaluator = ModelEvaluator()
        
    def evaluate_model(self, model_path: Path, X_test: np.ndarray, 
                      y_test: np.ndarray, model_name: str = "model",
                      dataset_name: str = "test") -> EvaluationResult:
        """Evaluate trained model."""
        import joblib
        
        model = joblib.load(model_path / "model.joblib")
        
        return self.evaluator.evaluate(
            model, X_test, y_test,
            model_name=model_name,
            dataset_name=dataset_name
        )
    
    def compare_all(self, results_dir: Path) -> Dict[str, Any]:
        """Compare all trained models."""
        from ml.models.trainers import ModelTrainer
        
        all_results = []
        
        for model_dir in results_dir.iterdir():
            if not model_dir.is_dir():
                continue
                
            metrics_file = model_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    all_results.append(json.load(f))
                    
        return self.evaluator.compare_models_results(all_results)
    
    def generate_report(self, results_dir: Path, output_file: Path) -> None:
        """Generate evaluation report."""
        comparison = self.compare_all(results_dir)
        
        with open(output_file, 'w') as f:
            json.dump(comparison, f, indent=2)
            
        logger.info(f"Report saved to {output_file}")


def run_training(config: PipelineConfig, data_dir: Path, 
                output_dir: Path) -> List[PipelineResult]:
    """Run training pipeline."""
    pipeline = TrainingPipeline(config)
    return pipeline.run(data_dir, output_dir)


def run_evaluation(models_dir: Path, test_data: tuple, 
                  output_dir: Path) -> EvaluationResult:
    """Run evaluation pipeline."""
    X_test, y_test = test_data
    
    evaluator = ModelEvaluator()
    
    # Evaluate each model
    results = []
    for model_dir in models_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        try:
            result = evaluator.evaluate_model(model_dir, X_test, y_test)
            results.append(result)
            
            evaluator.save_results(result, model_dir)
        except Exception as e:
            logger.error(f"Error evaluating {model_dir.name}: {e}")
            
    # Generate comparison
    if results:
        comparison = evaluator.compare_models(results)
        
        with open(output_dir / "benchmark.json", 'w') as f:
            json.dump(comparison, f, indent=2)
            
    return results[0] if results else None