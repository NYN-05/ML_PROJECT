# Fake News Detection - Multi-Dataset Independent Training System

A production-grade machine learning system for fake news detection using independent dataset training, model comparison, ensemble learning, and cross-dataset evaluation.

## Architecture Overview

The project implements a **multi-dataset independent training architecture** where each dataset trains its own dedicated model instead of merging all datasets together. This approach:

- Preserves dataset-specific characteristics
- Prevents vocabulary contamination
- Enables cross-dataset generalization testing
- Supports ensemble learning across diverse models

## Tech Stack

| Layer | Technology |
|-------|------------|
| ML Framework | PyTorch 2.12.0+cu130 |
| GPU | NVIDIA CUDA 13.0 (RTX 4050) |
| Backend | Node.js / Express.js |
| Frontend | React 18 + Vite + Tailwind CSS |
| Models | Logistic Regression, Linear SVM, Naive Bayes, Random Forest, XGBoost |

## Project Structure

```
ml_PROJ/
├── src/                         # New modular ML pipeline
│   ├── data/                    # Dataset discovery & loading
│   │   ├── dataset_discovery.py
│   │   └── dataset_loader.py
│   ├── preprocessing/            # Text preprocessing & vectorization
│   │   └── pipeline.py
│   ├── training/                 # Model training
│   │   ├── trainer.py
│   │   └── independent_pipeline.py
│   ├── ensemble/                # Ensemble learning
│   │   └── voting.py
│   ├── evaluation/              # Benchmarking & cross-dataset
│   │   ├── benchmark.py
│   │   └── cross_dataset.py
│   ├── utils/
│   └── main.py                  # Main execution script
│
├── ml-model/                    # Legacy PyTorch pipeline
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── tokenizer_utils.py
│   ├── model_builder.py
│   ├── train.py
│   ├── evaluate.py
│   └── infer.py
│
├── backend/                     # Express.js API server
├── frontend/                    # React UI
├── dataset/                     # Multiple fake news datasets
├── models/                      # Trained model artifacts
├── outputs/                     # New pipeline outputs
│   ├── datasets/               # Dataset registry
│   ├── models/                 # Trained models per dataset
│   ├── benchmarks/             # Model comparisons
│   ├── ensemble/               # Ensemble configs
│   └── cross_eval/             # Cross-dataset results
└── requirements.txt
```

## New Pipeline Features

### Phase 1: Independent Dataset Training
- Automatic dataset discovery (Fake.csv+True.csv, train/test CSVs, single CSVs)
- Isolated preprocessing per dataset
- Individual tokenizer/vectorizer per dataset
- Dedicated model training per dataset
- Model artifacts stored in `outputs/models/<dataset_name>/`

### Phase 2: Model Comparison & Benchmarking
- Automatic leaderboard generation
- Comparison across all trained models
- Metrics: accuracy, precision, recall, F1, ROC-AUC

### Phase 3: Ensemble Learning
- Hard voting, soft voting, weighted voting
- Stacking classifier support

### Phase 4: Cross-Dataset Evaluation
- Train on one dataset, test on others
- Cross-dataset evaluation matrix

## Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt

# Install PyTorch with CUDA 13.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

### Run the Pipeline
```bash
cd ml_PROJ

# Run all phases
python -m src.main --datasets dataset --outputs outputs

# Run specific phase
python -m src.main --phase discover --datasets dataset
python -m src.main --phase train --datasets dataset
python -m src.main --phase benchmark --models outputs/models
python -m src.main --phase ensemble --models outputs/models
python -m src.main --phase cross_eval --datasets dataset --models outputs/models
```

### Training Options
```bash
# Custom model type
python -m src.main --model-type logistic_regression --datasets dataset
python -m src.main --model-type xgboost --datasets dataset

# Custom preprocessing
python -m src.main --max-features 20000 --ngram-range "1,3" --datasets dataset
```

## Supported Datasets

The system automatically detects and processes:

| Dataset | Format | Type |
|---------|--------|------|
| Fake.csv + True.csv | Separate files | fake_true_split |
| train.csv + test.csv | Separate files | train_test_split |
| data.csv with split column | Single file | single_csv |
| fake_or_real_news.csv | Single file | single_csv |
| WELFake_Dataset.csv | Single file | single_csv |

## Output Artifacts

For each dataset, the pipeline produces:

```
outputs/models/<dataset_name>/
├── metrics.json              # Performance metrics
├── classification_report.txt # Detailed classification report
├── confusion_matrix.png      # Confusion matrix visualization
├── model.pkl                # Trained model
├── vectorizer.pkl           # TF-IDF/Count vectorizer
├── preprocessing_config.json # Preprocessing settings
└── pipeline_config.json     # Pipeline configuration
```

## Benchmark Results

Generated in `outputs/benchmarks/`:
- `leaderboard.csv` - Model rankings
- `leaderboard.json` - Rankings in JSON
- `comparison_report.md` - Markdown comparison report

## Cross-Dataset Evaluation

Generated in `outputs/cross_eval/`:
- `cross_dataset_results.csv` - All train→test combinations
- `cross_dataset_results.json` - Results in JSON
- `cross_dataset_matrix.png` - Heatmap visualization

## Configuration

Key configuration options in `src/training/trainer.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_type` | logistic_regression | Model algorithm |
| `test_size` | 0.2 | Test split ratio |
| `max_features` | 10000 | TF-IDF max features |
| `ngram_range` | (1,2) | N-gram range |
| `C` | 1.0 | Regularization |
| `class_weight` | balanced | Class weighting |

## Environment Requirements

- **Python 3.10+**
- **NVIDIA GPU** with CUDA 13.0+
- **Node.js 18+** for backend/frontend

## Legacy Pipeline

The original PyTorch LSTM-based pipeline in `ml-model/` is still available:
```bash
cd ml-model && python train.py
```

## API Endpoints (Backend)

- `POST /api/predict` - Single prediction
- `POST /api/batch` - Batch predictions
- `GET /api/models` - List available models
- `GET /api/datasets` - List discovered datasets