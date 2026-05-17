# Multi-Dataset Training System - Usage Guide

## Quick Start

### 1. Discover All Datasets
```bash
python -m src.main --phase discover --datasets dataset
```
This scans all subfolders in `dataset/` and creates a registry at `outputs/datasets/datasets_registry.json`.

### 2. Train Models on All Datasets
```bash
python -m src.main --phase train --datasets dataset --outputs outputs
```

### 3. Benchmark All Models
```bash
python -m src.main --phase benchmark --models outputs/models --outputs outputs
```

### 4. Ensemble Learning
```bash
python -m src.main --phase ensemble --models outputs/models --outputs outputs
```

### 5. Cross-Dataset Evaluation
```bash
python -m src.main --phase cross_eval --datasets dataset --models outputs/models --outputs outputs
```

## All-in-One Execution

Run the complete pipeline:
```bash
python -m src.main --datasets dataset --outputs outputs
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--datasets` | Path to datasets folder | `dataset` |
| `--outputs` | Path to outputs folder | `outputs` |
| `--models` | Path to trained models (for evaluation) | `outputs/models` |
| `--phase` | Phase to execute: all, discover, train, benchmark, ensemble, cross_eval | `all` |
| `--model-type` | Model type: logistic_regression, linear_svm, naive_bayes, random_forest, xgboost | `logistic_regression` |
| `--max-features` | TF-IDF max features | `10000` |
| `--ngram-range` | N-gram range (e.g., "1,2") | `"1,2"` |
| `--seed` | Random seed | `42` |

## Examples

### Train with Different Model Types

```bash
# Logistic Regression (default)
python -m src.main --phase train --datasets dataset --model-type logistic_regression

# XGBoost
python -m src.main --phase train --datasets dataset --model-type xgboost

# Random Forest
python -m src.main --phase train --datasets dataset --model-type random_forest
```

### Custom Preprocessing

```bash
# More features, include trigrams
python -m src.main --phase train --datasets dataset --max-features 20000 --ngram-range "1,3"
```

### Resume from Existing Models

```bash
# Only run benchmarking on existing models
python -m src.main --phase benchmark --models outputs/models
```

## Output Locations

| Output | Location |
|--------|-----------|
| Dataset Registry | `outputs/datasets/datasets_registry.json` |
| Trained Models | `outputs/models/<dataset_name>/` |
| Leaderboard | `outputs/benchmarks/leaderboard.csv` |
| Comparison Report | `outputs/benchmarks/comparison_report.md` |
| Ensemble Config | `outputs/ensemble/ensemble_config.json` |
| Cross-Dataset Results | `outputs/cross_eval/cross_dataset_results.csv` |

## Per-Dataset Output Structure

Each dataset produces:

```
outputs/models/<dataset_name>/
├── metrics.json               # accuracy, precision, recall, F1, ROC-AUC
├── classification_report.txt # sklearn classification report
├── confusion_matrix.png     # confusion matrix visualization
├── model.pkl               # trained model (joblib)
├── vectorizer.pkl          # fitted TF-IDF vectorizer
├── preprocessing_config.json# text cleaning settings
└── pipeline_config.json    # pipeline configuration
```

## Using Trained Models for Prediction

```python
import joblib
import pandas as pd

# Load model and vectorizer
model = joblib.load('outputs/models/dataset_name/model.pkl')
vectorizer = joblib.load('outputs/models/dataset_name/vectorizer.pkl')

# Preprocess and predict
text = "Sample news article text"
X = vectorizer.transform([text])
prediction = model.predict(X)
probability = model.predict_proba(X)
```

## Cross-Dataset Evaluation

The cross-dataset evaluation tests generalization by training on one dataset and testing on others:

```
Train Dataset → Test Dataset → Accuracy
ISOT → WELFake → ~82%
WELFake → ISOT → ~76%
```

Results are saved as a heatmap matrix in `outputs/cross_eval/cross_dataset_matrix.png`.

## Ensemble Learning

To create an ensemble from multiple trained models:

```python
from src.ensemble.voting import EnsembleSystem

ensemble = EnsembleSystem(Path('outputs/models'))
ensemble.load_models(['dataset1', 'dataset2', 'dataset3'])

# Soft voting (average probabilities)
result = ensemble.predict_soft_voting(X_test)

# Hard voting (majority)
result = ensemble.predict_hard_voting(X_test)

# Weighted voting
weights = {'dataset1': 0.5, 'dataset2': 0.3, 'dataset3': 0.2}
result = ensemble.predict_weighted_voting(X_test, weights)
```