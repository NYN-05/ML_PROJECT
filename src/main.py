"""Main execution script for the Multi-Dataset Independent Training Pipeline.

This script implements the complete architecture:
- Phase 1: Independent Dataset Training
- Phase 2: Model Comparison and Benchmarking
- Phase 3: Ensemble Learning
- Phase 4: Cross-Dataset Evaluation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.dataset_discovery import DatasetDiscovery, discover_datasets
from src.data.dataset_loader import DatasetLoader
from src.preprocessing.pipeline import PreprocessingConfig
from src.training.independent_pipeline import IndependentPipeline
from src.training.trainer import TrainingConfig, ModelType
from src.evaluation.benchmark import ModelBenchmark, run_benchmark
from src.ensemble.voting import EnsembleSystem
from src.evaluation.cross_dataset import CrossDatasetEvaluator
from src.utils.helpers import setup_logging, set_random_seed, ensure_dir


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Dataset Independent Training Pipeline"
    )
    
    parser.add_argument(
        "--datasets",
        type=str,
        default=r"C:\Users\JHASHANK\Downloads\ml_PROJ\dataset",
        help="Path to datasets directory"
    )
    
    parser.add_argument(
        "--outputs",
        type=str,
        default=r"C:\Users\JHASHANK\Downloads\ml_PROJ\outputs",
        help="Path to outputs directory"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Path to trained models (for evaluation/ensemble)"
    )
    
    parser.add_argument(
        "--phase",
        type=str,
        choices=["all", "discover", "train", "benchmark", "ensemble", "cross_eval"],
        default="all",
        help="Phase to execute"
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        default="logistic_regression",
        choices=["logistic_regression", "linear_svm", "naive_bayes", "random_forest", "xgboost"],
        help="Model type to train"
    )
    
    parser.add_argument(
        "--max-features",
        type=int,
        default=10000,
        help="Maximum number of features for TF-IDF"
    )
    
    parser.add_argument(
        "--ngram-range",
        type=str,
        default="1,2",
        help="N-gram range (e.g., '1,2' for unigrams and bigrams)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()


def setup_output_structure(base_dir: Path) -> Dict[str, Path]:
    """Create output directory structure."""
    return {
        "root": base_dir,
        "datasets": ensure_dir(base_dir / "datasets"),
        "models": ensure_dir(base_dir / "models"),
        "benchmarks": ensure_dir(base_dir / "benchmarks"),
        "ensemble": ensure_dir(base_dir / "ensemble"),
        "cross_eval": ensure_dir(base_dir / "cross_eval"),
        "logs": ensure_dir(base_dir / "logs")
    }


def run_dataset_discovery(datasets_path: Path, output_path: Path, logger: logging.Logger) -> List:
    """Phase 1: Discover all datasets."""
    logger.info("=" * 60)
    logger.info("PHASE 1: DATASET DISCOVERY")
    logger.info("=" * 60)
    
    discovery = DatasetDiscovery(datasets_path)
    datasets = discovery.discover()
    
    logger.info(f"Discovered {len(datasets)} datasets:")
    for d in datasets:
        logger.info(f"  - {d.name} ({d.dataset_type}, {d.n_rows} rows)")
    
    registry_path = output_path / "datasets" / "datasets_registry.json"
    discovery.save_registry(registry_path)
    
    return datasets


def run_independent_training(
    datasets: List,
    outputs: Dict[str, Path],
    model_type: str,
    max_features: int,
    ngram_range: tuple,
    seed: int,
    logger: logging.Logger
) -> List:
    """Phase 1 & 2: Train models on each dataset independently."""
    logger.info("=" * 60)
    logger.info("PHASE 1 & 2: INDEPENDENT TRAINING")
    logger.info("=" * 60)
    
    preprocessing_config = PreprocessingConfig(
        lowercase=True,
        remove_punctuation=True,
        remove_urls=True,
        remove_html=True,
        remove_stopwords=False,
        max_features=max_features,
        ngram_range=ngram_range,
        vectorizer_type="tfidf"
    )
    
    training_config = TrainingConfig(
        model_type=model_type,
        test_size=0.2,
        validation_size=0.1,
        random_seed=seed,
        C=1.0,
        max_iter=1000
    )
    
    results = []
    successful = 0
    failed = 0
    
    for dataset_info in datasets:
        logger.info(f"\nTraining on: {dataset_info.name}")
        
        pipeline = IndependentPipeline(
            output_dir=outputs["models"],
            preprocessing_config=preprocessing_config,
            training_config=training_config
        )
        
        try:
            result = pipeline.run(dataset_info)
            
            if result.success:
                successful += 1
                logger.info(f"  SUCCESS - Accuracy: {result.training_result.test_accuracy:.4f}")
            else:
                failed += 1
                logger.error(f"  FAILED - {result.error_message}")
            
            results.append(result)
            
        except Exception as e:
            failed += 1
            logger.error(f"  ERROR - {str(e)}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training Summary: {successful} successful, {failed} failed")
    logger.info(f"{'='*60}")
    
    return results


def run_benchmarking(outputs: Dict[str, Path], logger: logging.Logger):
    """Phase 2: Model comparison and benchmarking."""
    logger.info("=" * 60)
    logger.info("PHASE 2: MODEL BENCHMARKING")
    logger.info("=" * 60)
    
    benchmark = ModelBenchmark(outputs["models"])
    benchmark.scan_all_results(outputs["models"])
    
    leaderboard = benchmark.generate_leaderboard(outputs["benchmarks"] / "leaderboard")
    benchmark.generate_comparison_report(outputs["benchmarks"] / "comparison_report.md")
    
    logger.info(f"\nLeaderboard:")
    logger.info(leaderboard.to_string())
    
    return leaderboard


def run_ensemble_evaluation(models_dir: Path, outputs: Dict[str, Path], logger: logging.Logger):
    """Phase 3: Ensemble learning evaluation."""
    logger.info("=" * 60)
    logger.info("PHASE 3: ENSEMBLE LEARNING")
    logger.info("=" * 60)
    
    available_models = [d.name for d in models_dir.iterdir() if d.is_dir()]
    
    if len(available_models) < 2:
        logger.warning("Need at least 2 models for ensemble")
        return
    
    ensemble = EnsembleSystem(models_dir)
    ensemble.load_models(available_models[:3])
    
    logger.info(f"Created ensemble with models: {list(ensemble.models.keys())}")
    
    ensemble.save_ensemble_config(outputs["ensemble"], {
        "models": list(ensemble.models.keys()),
        "method": "soft_voting"
    })
    
    logger.info(f"Ensemble configuration saved to: {outputs['ensemble']}")


def run_cross_dataset_evaluation(models_dir: Path, datasets_path: Path, outputs: Dict[str, Path], logger: logging.Logger):
    """Phase 4: Cross-dataset evaluation."""
    logger.info("=" * 60)
    logger.info("PHASE 4: CROSS-DATASET EVALUATION")
    logger.info("=" * 60)
    
    discovery = DatasetDiscovery(datasets_path)
    datasets = discovery.discover()
    
    evaluator = CrossDatasetEvaluator(models_dir)
    evaluator.load_all_models()
    
    test_datasets = {}
    for d in datasets:
        loader = DatasetLoader()
        loaded = loader.load(d)
        
        if loaded.train is not None:
            test_datasets[d.name] = loaded.train
    
    if len(test_datasets) < 2:
        logger.warning("Need at least 2 datasets for cross-evaluation")
        return
    
    results = evaluator.run_evaluation(test_datasets)
    evaluator.save_results(results, outputs["cross_eval"] / "cross_dataset_results")
    
    logger.info(f"\nCross-dataset results:")
    for r in results:
        logger.info(f"  {r.train_dataset} -> {r.test_dataset}: {r.accuracy:.4f}")


def main():
    """Main execution function."""
    args = parse_args()
    
    datasets_path = Path(args.datasets)
    outputs_base = Path(args.outputs)
    
    outputs = setup_output_structure(outputs_base)
    
    logger = setup_logging(
        name="ml_pipeline",
        level=logging.INFO,
        log_file=outputs["logs"] / f"pipeline_{int(time.time())}.log"
    )
    
    set_random_seed(args.seed)
    
    logger.info("Multi-Dataset Independent Training Pipeline")
    logger.info(f"Datasets: {datasets_path}")
    logger.info(f"Outputs: {outputs_base}")
    logger.info(f"Model: {args.model_type}")
    
    ngram_tuple = tuple(map(int, args.ngram_range.split(',')))
    
    if args.phase in ["all", "discover"]:
        datasets = run_dataset_discovery(datasets_path, outputs["datasets"], logger)
    else:
        datasets = []
    
    if args.phase in ["all", "train"]:
        if not datasets:
            discovery = DatasetDiscovery(datasets_path)
            datasets = discovery.discover()
        
        results = run_independent_training(
            datasets,
            outputs,
            args.model_type,
            args.max_features,
            ngram_tuple,
            args.seed,
            logger
        )
    
    if args.phase in ["all", "benchmark"]:
        run_benchmarking(outputs["models"], logger)
    
    if args.phase in ["all", "ensemble"]:
        run_ensemble_evaluation(outputs["models"], outputs["ensemble"], logger)
    
    if args.phase in ["all", "cross_eval"]:
        run_cross_dataset_evaluation(outputs["models"], datasets_path, outputs["cross_eval"], logger)
    
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()