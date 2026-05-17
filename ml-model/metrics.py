"""Evaluation metrics and reporting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from logging_utils import get_logger

LOGGER = get_logger("ml.metrics")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary of aggregated metrics.
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    return {
        "accuracy": report["accuracy"],
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"],
    }


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    """Save classification report to disk.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        output_path: Output path for the report text.
    """
    report = classification_report(y_true, y_pred)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


def save_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    """Save confusion matrix plot.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        output_path: Output path for the plot.
    """
    matrix = confusion_matrix(y_true, y_pred)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
