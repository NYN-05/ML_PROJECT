# Multi-Dataset Independent Training System - Implementation Guide

## Overview

This document describes the implementation of the multi-dataset independent training architecture for fake news detection. The system allows each dataset to train its own dedicated model, enabling better generalization, ensemble learning, and cross-dataset evaluation.

## Architecture

### Core Principle

```
1 Dataset = 1 Pipeline = 1 Model
```

Each dataset maintains its own:
- Preprocessing pipeline
- Tokenizer/Vectorizer
- Feature extraction
- Trained model
- Evaluation metrics

### System Architecture

```
                    MULTIPLE DATASETS
                          │
    ┌─────────────────────┼─────────────────────┐
    │                     │                     │
    ▼                     ▼                     ▼
Dataset A           Dataset B              Dataset C
    │                     │                     │
    ▼                     ▼                     ▼
Pipeline A          Pipeline B             Pipeline C
    │                     │                     │
    ▼                     ▼                     ▼
Model A             Model B                Model C
    │                     │                     │
    └───────────────┬───────┴───────────────┬───────┘
                    ▼                       ▼
             Model Comparison       Cross Evaluation
                    │
                    ▼
             Ensemble Learning
                    │
                    ▼
             Final Prediction System
```

## Implementation Details

### 1. Dataset Discovery System

**File:** `src/data/dataset_discovery.py`

Automatically scans dataset folders and detects:
- Fake.csv + True.csv format
- Train/test split format
- Single CSV with label column
- TSV format

**Features:**
- Recursive folder scanning
- Excludes output/logs directories
- Detects target column automatically
- Identifies text columns
- Counts total rows

### 2. Dataset Loader

**File:** `src/data/dataset_loader.py`

Loads datasets with automatic splitting:
- Fake/True format → combines with labels
- Train/test format → uses as-is
- Single file → splits into train/test/validation
- Handles encoding issues automatically

### 3. Preprocessing Pipeline

**File:** `src/preprocessing/pipeline.py`

Configurable text preprocessing:
- Lowercase conversion
- Punctuation/URL/HTML removal
- Stopword removal (optional)
- Stemming/Lemmatization (optional)
- TF-IDF or Count vectorization
- N-gram configuration

Each dataset gets its own vectorizer to prevent vocabulary contamination.

### 4. Model Training

**File:** `src/training/trainer.py`

Supports multiple model types:
- Logistic Regression
- Linear SVM
- Naive Bayes
- Random Forest
- XGBoost

Features:
- Configurable hyperparameters
- Class weight balancing
- Train/validation/test split
- Early stopping support

### 5. Independent Pipeline

**File:** `src/training/independent_pipeline.py`

Orchestrates the complete pipeline for a single dataset:
1. Load dataset
2. Preprocess text
3. Vectorize
4. Train model
5. Evaluate
6. Save artifacts

### 6. Model Benchmarking

**File:** `src/evaluation/benchmark.py`

Compares all trained models:
- Generates leaderboard (CSV/JSON)
- Creates comparison report (Markdown)
- Ranks by accuracy, precision, recall, F1, ROC-AUC

### 7. Ensemble Learning

**File:** `src/ensemble/voting.py`

Combines multiple models:
- Hard voting (majority)
- Soft voting (probability average)
- Weighted voting (custom weights)
- Stacking classifier support

### 8. Cross-Dataset Evaluation

**File:** `src/evaluation/cross_dataset.py`

Tests generalization:
- Train on dataset A, test on dataset B
- Creates evaluation matrix
- Generates heatmap visualization

## Usage

### Running the Pipeline

```bash
# Full pipeline
python -m src.main --datasets dataset --outputs outputs

# Individual phases
python -m src.main --phase discover --datasets dataset
python -m src.main --phase train --datasets dataset --model-type logistic_regression
python -m src.main --phase benchmark --models outputs/models
python -m src.main --phase ensemble --models outputs/models
python -m src.main --phase cross_eval --datasets dataset --models outputs/models
```

### Configuration

Key parameters:
- `--model-type`: logistic_regression, linear_svm, naive_bayes, random_forest, xgboost
- `--max-features`: TF-IDF max features (default: 10000)
- `--ngram-range`: N-gram range (default: "1,2")
- `--seed`: Random seed (default: 42)

## Output Structure

```
outputs/
├── datasets/
│   └── datasets_registry.json    # Discovered datasets
├── models/
│   ├── dataset_1/
│   │   ├── metrics.json
│   │   ├── classification_report.txt
│   │   ├── confusion_matrix.png
│   │   ├── model.pkl
│   │   ├── vectorizer.pkl
│   │   └── pipeline_config.json
│   └── dataset_2/
│       └── ...
├── benchmarks/
│   ├── leaderboard.csv
│   ├── leaderboard.json
│   └── comparison_report.md
├── ensemble/
│   └── ensemble_config.json
└── cross_eval/
    ├── cross_dataset_results.csv
    ├── cross_dataset_results.json
    └── cross_dataset_matrix.png
```

## Advantages

### Modularity
- Each dataset is independent
- Easy to add new datasets
- No interference between pipelines

### Scalability
- Parallel training possible
- New models can be added easily
- Flexible configuration

### Research Capability
- Cross-dataset evaluation
- Ensemble methods
- Comprehensive benchmarking

### Maintainability
- Clear separation of concerns
- Easy debugging
- Reusable components

## Future Enhancements

- Transformer-based models (BERT, DistilBERT)
- Distributed training with PySpark
- Hyperparameter optimization
- Real-time API deployment
- Online learning support