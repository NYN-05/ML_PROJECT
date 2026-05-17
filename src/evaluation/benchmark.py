"""Model Comparison and Benchmarking Module."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger("evaluation.benchmark")


@dataclass
class BenchmarkResult:
    """Result of model comparison."""
    dataset_name: str
    model_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    rank: int


class ModelBenchmark:
    """Benchmark and compare trained models."""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.benchmark_results: List[BenchmarkResult] = []
    
    def add_result(self, metrics_path: Path) -> None:
        """Add a result from a metrics.json file."""
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            result = BenchmarkResult(
                dataset_name=metrics.get("dataset_name", "unknown"),
                model_type=metrics.get("model_type", "unknown"),
                accuracy=metrics.get("test_accuracy", 0.0),
                precision=metrics.get("precision", 0.0),
                recall=metrics.get("recall", 0.0),
                f1_score=metrics.get("f1_score", 0.0),
                roc_auc=metrics.get("roc_auc", 0.0),
                rank=0
            )
            
            self.benchmark_results.append(result)
            logger.info(f"Added benchmark result for: {result.dataset_name}")
            
        except Exception as e:
            logger.error(f"Failed to load metrics from {metrics_path}: {e}")
    
    def scan_all_results(self, base_dir: Path) -> None:
        """Scan all dataset result directories."""
        if not base_dir.exists():
            logger.warning(f"Results directory does not exist: {base_dir}")
            return
        
        for dataset_dir in base_dir.iterdir():
            if dataset_dir.is_dir():
                metrics_path = dataset_dir / "metrics.json"
                if metrics_path.exists():
                    self.add_result(metrics_path)
    
    def rank_models(self, sort_by: str = "f1_score") -> pd.DataFrame:
        """Rank models by a specific metric."""
        if not self.benchmark_results:
            return pd.DataFrame()
        
        results_df = pd.DataFrame([asdict(r) for r in self.benchmark_results])
        
        results_df = results_df.sort_values(by=sort_by, ascending=False)
        
        results_df['rank'] = range(1, len(results_df) + 1)
        
        for i, r in enumerate(self.benchmark_results):
            r.rank = i + 1
        
        return results_df
    
    def generate_leaderboard(self, output_path: Path, sort_by: str = "f1_score") -> pd.DataFrame:
        """Generate a leaderboard and save to CSV/JSON."""
        leaderboard = self.rank_models(sort_by)
        
        if leaderboard.empty:
            logger.warning("No results to generate leaderboard")
            return leaderboard
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        leaderboard.to_csv(output_path.with_suffix('.csv'), index=False)
        
        leaderboard_json = leaderboard.to_dict(orient='records')
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(leaderboard_json, f, indent=2)
        
        logger.info(f"Generated leaderboard: {output_path}")
        
        return leaderboard
    
    def generate_comparison_report(self, output_path: Path) -> str:
        """Generate a markdown comparison report."""
        leaderboard = self.rank_models()
        
        if leaderboard.empty:
            return "# No Results Available\n"
        
        report_lines = [
            "# Model Comparison Report\n",
            f"## Total Models: {len(leaderboard)}\n",
            f"## Sorted by: F1 Score\n",
            "\n## Leaderboard\n",
            "| Rank | Dataset | Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |",
            "|------|---------|-------|----------|-----------|--------|-----------|---------|"
        ]
        
        for _, row in leaderboard.iterrows():
            report_lines.append(
                f"| {row['rank']} | {row['dataset_name']} | {row['model_type']} | "
                f"{row['accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | "
                f"{row['f1_score']:.4f} | {row['roc_auc']:.4f} |"
            )
        
        report_lines.append("\n## Summary Statistics\n")
        
        for col in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            report_lines.append(
                f"- **{col.capitalize()}**: "
                f"Mean={leaderboard[col].mean():.4f}, "
                f"Std={leaderboard[col].std():.4f}, "
                f"Min={leaderboard[col].min():.4f}, "
                f"Max={leaderboard[col].max():.4f}"
            )
        
        report_text = "\n".join(report_lines)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Generated comparison report: {output_path}")
        
        return report_text
    
    def get_best_model(self, sort_by: str = "f1_score") -> Optional[BenchmarkResult]:
        """Get the best performing model."""
        if not self.benchmark_results:
            return None
        
        ranked = sorted(self.benchmark_results, key=lambda x: getattr(x, sort_by), reverse=True)
        return ranked[0]


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


def run_benchmark(results_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Run benchmark on all results."""
    benchmark = ModelBenchmark(results_dir)
    benchmark.scan_all_results(results_dir)
    
    leaderboard = benchmark.generate_leaderboard(output_dir / "leaderboard")
    benchmark.generate_comparison_report(output_dir / "comparison_report.md")
    
    return leaderboard