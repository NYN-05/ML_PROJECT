"""Main entry point for unified ML pipeline."""

import argparse
import logging
from pathlib import Path

from ml.config import PipelineConfig, ModelType
from ml.pipelines import run_training, run_evaluation
from ml.utils import setup_logging, set_random_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Unified ML Pipeline")
    
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'evaluate', 'inference'],
                       help='Pipeline mode')
    
    parser.add_argument('--data-dir', type=str, default='dataset',
                       help='Dataset directory')
    
    parser.add_argument('--output-dir', type=str, default='artifacts',
                       help='Output directory')
    
    parser.add_argument('--model-type', type=str, default='logistic_regression',
                       choices=['lstm', 'logistic_regression', 'linear_svm', 
                               'naive_bayes', 'random_forest', 'xgboost'],
                       help='Model type')
    
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    
    parser.add_argument('--max-features', type=int, default=10000,
                       help='Maximum TF-IDF features')
    
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test split ratio')
    
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed')
    
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                       help='Specific datasets to train on')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level)
    
    set_random_seed(args.random_seed)
    
    logger.info(f"Starting ML Pipeline in {args.mode} mode")
    logger.info(f"Model: {args.model_type}")
    
    # Configuration
    config = PipelineConfig(
        model=type('ModelConfig', (), {
            'model_type': ModelType(args.model_type),
            'max_seq_length': 256,
            'vocab_size': args.max_features
        })(),
        training=type('TrainingConfig', (), {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'test_size': args.test_size,
            'random_seed': args.random_seed,
            'patience': 5
        })(),
        dataset=type('DatasetConfig', (), {
            'data_dir': Path(args.data_dir),
            'output_dir': Path(args.output_dir)
        })(),
        verbose=args.verbose
    )
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run based on mode
    if args.mode == 'train':
        results = run_training(config, data_dir, output_dir)
        
        logger.info(f"Training complete! Trained {len(results)} models.")
        
        # Summary
        for result in results:
            logger.info(f"  {result.dataset_name}: {result.training_result.test_accuracy:.4f}")
            
    elif args.mode == 'evaluate':
        logger.info("Evaluation mode - specify models and test data")
        
    elif args.mode == 'inference':
        logger.info("Inference mode - load model and make predictions")
        
    logger.info("Done!")


if __name__ == '__main__':
    main()