"""Metrics computation utilities."""

import logging
from typing import Dict, List, Any, Tuple
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, average_precision_score
)

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   y_prob: np.ndarray = None) -> Dict[str, float]:
    """Compute all classification metrics."""
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
    }
    
    if y_prob is not None:
        try:
            if y_prob.ndim > 1:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
                metrics['avg_precision'] = average_precision_score(y_true, y_prob[:, 1])
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
                metrics['avg_precision'] = average_precision_score(y_true, y_prob)
        except:
            metrics['roc_auc'] = 0.0
            metrics['avg_precision'] = 0.0
            
    return metrics


def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute confusion matrix metrics."""
    cm = confusion_matrix(y_true, y_pred)
    
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'confusion_matrix': cm.tolist(),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0
    }


def compute_threshold_metrics(y_true: np.ndarray, y_scores: np.ndarray,
                             thresholds: List[float] = None) -> List[Dict]:
    """Compute metrics at different thresholds."""
    
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1).tolist()
        
    results = []
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        
        metrics = compute_metrics(y_true, y_pred, y_scores)
        metrics['threshold'] = threshold
        
        results.append(metrics)
        
    return results


def find_optimal_threshold(y_true: np.ndarray, y_scores: np.ndarray,
                          metric: str = 'f1') -> Tuple[float, Dict]:
    """Find optimal classification threshold."""
    
    thresholds = np.arange(0.1, 1.0, 0.01)
    best_threshold = 0.5
    best_score = 0
    best_metrics = {}
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        metrics = compute_metrics(y_true, y_pred)
        
        score = metrics.get(metric, 0)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics
            
    return best_threshold, best_metrics


def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                  class_names: List[str] = None) -> str:
    """Generate detailed classification report."""
    
    return classification_report(y_true, y_pred, target_names=class_names)


def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: List[str]) -> Dict[str, Dict]:
    """Compute per-class metrics."""
    
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    metrics = {}
    for i, name in enumerate(class_names):
        metrics[name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }
        
    return metrics


def compute_error_analysis(y_true: np.ndarray, y_pred: np.ndarray,
                          texts: List[str] = None) -> Dict[str, Any]:
    """Analyze prediction errors."""
    
    errors = {
        'total_errors': int((y_true != y_pred).sum()),
        'error_rate': float((y_true != y_pred).mean()),
        'false_positives': [],
        'false_negatives': []
    }
    
    if texts is not None:
        for i, (true, pred) in enumerate(zip(y_true, y_pred)):
            if true != pred:
                error_entry = {'index': i, 'text': texts[i][:100]}
                
                if pred == 1 and true == 0:
                    errors['false_positives'].append(error_entry)
                else:
                    errors['false_negatives'].append(error_entry)
                    
    return errors


class MetricsHistory:
    """Track metrics over training epochs."""
    
    def __init__(self):
        self.history = {
            'train': [],
            'val': [],
            'epoch': []
        }
        
    def update(self, epoch: int, train_metrics: Dict, val_metrics: Dict = None):
        """Update with new metrics."""
        self.history['epoch'].append(epoch)
        self.history['train'].append(train_metrics)
        
        if val_metrics:
            self.history['val'].append(val_metrics)
            
    def get_best(self, metric: str = 'val_loss', mode: str = 'min') -> Tuple[int, Dict]:
        """Get best epoch by metric."""
        if not self.history['val']:
            return 0, {}
            
        if mode == 'min':
            best_idx = min(range(len(self.history['val'])),
                         key=lambda i: self.history['val'][i].get(metric, float('inf')))
        else:
            best_idx = max(range(len(self.history['val'])),
                         key=lambda i: self.history['val'][i].get(metric, 0))
                         
        return best_idx, self.history['val'][best_idx]
    
    def save(self, path: Path) -> None:
        """Save metrics history."""
        import json
        
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
            
    def load(self, path: Path) -> None:
        """Load metrics history."""
        import json
        
        with open(path, 'r') as f:
            self.history = json.load(f)