"""Unified model evaluation system."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Evaluation results."""
    model_name: str
    dataset_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: List[List[int]]
    classification_report: str
    predictions: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'dataset_name': self.dataset_name,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'roc_auc': self.roc_auc,
            'confusion_matrix': self.confusion_matrix,
            'classification_report': self.classification_report,
        }


class ModelEvaluator:
    """Unified model evaluator."""
    
    def __init__(self):
        self.results = {}
        
    def evaluate(self, model, X_test: np.ndarray, y_test: np.ndarray,
                 model_name: str = "model", dataset_name: str = "test") -> EvaluationResult:
        """Evaluate model on test data."""
        logger.info(f"Evaluating {model_name} on {dataset_name}")
        
        # Predictions
        predictions = model.predict(X_test)
        
        # Probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(X_test)
            except:
                pass
        elif hasattr(model, 'decision_function'):
            try:
                probabilities = model.decision_function(X_test)
            except:
                pass
        
        # Metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='binary', zero_division=0)
        recall = recall_score(y_test, predictions, average='binary', zero_division=0)
        f1 = f1_score(y_test, predictions, average='binary', zero_division=0)
        
        # ROC-AUC
        roc_auc = 0.0
        if probabilities is not None:
            try:
                if probabilities.ndim > 1:
                    roc_auc = roc_auc_score(y_test, probabilities[:, 1])
                else:
                    roc_auc = roc_auc_score(y_test, probabilities)
            except:
                pass
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, predictions).tolist()
        
        # Classification report
        class_report = classification_report(y_test, predictions)
        
        result = EvaluationResult(
            model_name=model_name,
            dataset_name=dataset_name,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            confusion_matrix=conf_matrix,
            classification_report=class_report,
            predictions=predictions,
            probabilities=probabilities
        )
        
        logger.info(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
        
        return result
    
    def save_results(self, result: EvaluationResult, output_dir: Path) -> None:
        """Save evaluation results."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON
        metrics_file = output_dir / f"{result.model_name}_metrics.json"
        
        with open(metrics_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
            
        logger.info(f"Results saved to {metrics_file}")
        
    def compare_models(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Compare multiple model evaluations."""
        comparison = {
            'models': [],
            'best_by_accuracy': None,
            'best_by_f1': None,
            'best_by_roc_auc': None
        }
        
        best_accuracy = 0
        best_f1 = 0
        best_roc_auc = 0
        
        for result in results:
            model_data = {
                'name': result.model_name,
                'dataset': result.dataset_name,
                'accuracy': result.accuracy,
                'precision': result.precision,
                'recall': result.recall,
                'f1_score': result.f1_score,
                'roc_auc': result.roc_auc,
            }
            comparison['models'].append(model_data)
            
            if result.accuracy > best_accuracy:
                best_accuracy = result.accuracy
                comparison['best_by_accuracy'] = result.model_name
                
            if result.f1_score > best_f1:
                best_f1 = result.f1_score
                comparison['best_by_f1'] = result.model_name
                
            if result.roc_auc > best_roc_auc:
                best_roc_auc = result.roc_auc
                comparison['best_by_roc_auc'] = result.model_name
                
        return comparison
    
    def cross_dataset_evaluate(self, model, datasets: Dict[str, tuple]) -> Dict[str, EvaluationResult]:
        """Evaluate model across multiple datasets."""
        results = {}
        
        for dataset_name, (X, y) in datasets.items():
            result = self.evaluate(model, X, y, model_name=model.__class__.__name__, 
                                  dataset_name=dataset_name)
            results[dataset_name] = result
            
        return results
    
    def generate_report(self, results: List[EvaluationResult]) -> str:
        """Generate text report from evaluation results."""
        lines = ["=" * 60, "MODEL EVALUATION REPORT", "=" * 60, ""]
        
        for result in results:
            lines.append(f"Model: {result.model_name}")
            lines.append(f"Dataset: {result.dataset_name}")
            lines.append("-" * 40)
            lines.append(f"  Accuracy:  {result.accuracy:.4f}")
            lines.append(f"  Precision: {result.precision:.4f}")
            lines.append(f"  Recall:    {result.recall:.4f}")
            lines.append(f"  F1 Score:  {result.f1_score:.4f}")
            lines.append(f"  ROC-AUC:   {result.roc_auc:.4f}")
            lines.append("")
            
        return "\n".join(lines)


class Benchmark:
    """Model benchmarking utility."""
    
    @staticmethod
    def run(models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Benchmark multiple models."""
        import time
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Benchmarking {name}...")
            
            start = time.time()
            predictions = model.predict(X_test)
            elapsed = time.time() - start
            
            acc = accuracy_score(y_test, predictions)
            
            results[name] = {
                'accuracy': acc,
                'inference_time': elapsed,
                'samples': len(y_test),
                'throughput': len(y_test) / elapsed
            }
            
        return results