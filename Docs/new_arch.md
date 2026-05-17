# Multi-Dataset Independent Training and Ensemble Learning Architecture for Fake News Detection

## Abstract

This document describes the implemented architecture for fake news detection using multiple heterogeneous datasets. Instead of merging all datasets into a single training corpus, each dataset is processed and trained independently using isolated preprocessing pipelines and dedicated machine learning models. The architecture supports dataset-specific learning, robust evaluation, ensemble learning, and cross-dataset generalization testing.

---

## Implementation Status: COMPLETE

All phases have been implemented in the `src/` directory.

---

## Phase 1 — Independent Dataset Training

### Implemented Components

| Component | File | Status |
|-----------|------|--------|
| Dataset Discovery | `src/data/dataset_discovery.py` | ✅ Complete |
| Dataset Loader | `src/data/dataset_loader.py` | ✅ Complete |
| Text Preprocessing | `src/preprocessing/pipeline.py` | ✅ Complete |
| Independent Pipeline | `src/training/independent_pipeline.py` | ✅ Complete |
| Model Trainer | `src/training/trainer.py` | ✅ Complete |

### Architecture

```
1 Dataset = 1 Pipeline = 1 Model
```

### Supported Dataset Formats

- Fake.csv + True.csv (separate files)
- train.csv + test.csv (pre-split)
- Single CSV with split column
- Single CSV without split (auto-split)
- TSV format support

### Output Storage

Each trained dataset produces:
- `metrics.json` - performance metrics
- `confusion_matrix.png` - error visualization
- `model.pkl` - trained model
- `vectorizer.pkl` - tokenizer/vectorizer
- `preprocessing_config.json` - preprocessing metadata

---

## Phase 2 — Model Comparison and Benchmarking

### Implemented Components

| Component | File | Status |
|-----------|------|--------|
| Model Benchmark | `src/evaluation/benchmark.py` | ✅ Complete |

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correctness |
| Precision | False positive control |
| Recall | False negative control |
| F1 Score | Balanced performance |
| ROC-AUC | Classification robustness |

### Benchmarking Output

- `leaderboard.csv` - Ranked models
- `leaderboard.json` - JSON format
- `comparison_report.md` - Markdown report

---

## Phase 3 — Ensemble Learning

### Implemented Components

| Component | File | Status |
|-----------|------|--------|
| Voting Ensemble | `src/ensemble/voting.py` | ✅ Complete |
| Stacking Classifier | `src/ensemble/voting.py` | ✅ Complete |

### Supported Methods

1. **Hard Voting** - Majority class prediction
2. **Soft Voting** - Averaged probabilities
3. **Weighted Averaging** - Custom weights per model
4. **Stacking** - Meta-learner on base models

---

## Phase 4 — Cross-Dataset Evaluation

### Implemented Components

| Component | File | Status |
|-----------|------|--------|
| Cross-Dataset Evaluator | `src/evaluation/cross_dataset.py` | ✅ Complete |

### Workflow

Train on dataset A → Test on dataset B → Measure generalization

### Output

- `cross_dataset_results.csv` - All train→test combinations
- `cross_dataset_matrix.png` - Heatmap visualization
- `cross_dataset_results.json` - JSON format

---

## System Architecture Diagram

```text
                 MULTIPLE DATASETS
                          │
┌─────────────────────────┼─────────────────────────┐
│                         │                         │
▼                         ▼                         ▼
Dataset A             Dataset B                Dataset C
│                         │                         │
▼                         ▼                         ▼
DatasetDiscovery      DatasetDiscovery         DatasetDiscovery
│                         │                         │
▼                         ▼                         ▼
DatasetLoader         DatasetLoader            DatasetLoader
│                         │                         │
▼                         ▼                         ▼
TextPreprocessor      TextPreprocessor         TextPreprocessor
│                         │                         │
▼                         ▼                         ▼
Trainer               Trainer                  Trainer
│                         │                         │
└───────────────┬────────┴───────────┬───────────┘
                ▼                   ▼
         ModelBenchmark        CrossDatasetEvaluator
                │                   │
                ▼                   ▼
         EnsembleSystem    Evaluation Matrix
                │
                ▼
         Final Predictions
```

---

## Usage

### Quick Start

```bash
# Full pipeline
python -m src.main --datasets dataset --outputs outputs

# Individual phases
python -m src.main --phase discover --datasets dataset
python -m src.main --phase train --datasets dataset
python -m src.main --phase benchmark --models outputs/models
python -m src.main --phase ensemble --models outputs/models
python -m src.main --phase cross_eval --datasets dataset --models outputs/models
```

### Model Selection

```bash
python -m src.main --model-type logistic_regression
python -m src.main --model-type xgboost
python -m src.main --model-type random_forest
```

---

## Project Structure

```
ml_PROJ/
├── src/                         # New modular pipeline
│   ├── data/
│   │   ├── dataset_discovery.py
│   │   └── dataset_loader.py
│   ├── preprocessing/
│   │   └── pipeline.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── independent_pipeline.py
│   ├── ensemble/
│   │   └── voting.py
│   ├── evaluation/
│   │   ├── benchmark.py
│   │   └── cross_dataset.py
│   ├── utils/
│   │   └── helpers.py
│   ├── main.py
│   └── USAGE.md
│
├── ml-model/                    # Legacy PyTorch pipeline
├── dataset/                     # Input datasets
├── outputs/                     # Pipeline outputs
├── Docs/
│   ├── new_arch.md             # This document
│   └── IMPLEMENTATION.md       # Implementation details
└── requirements.txt
```

---

## Advantages Achieved

| Advantage | Status |
|-----------|--------|
| Modularity | ✅ Each dataset has independent pipeline |
| Scalability | ✅ Easy to add new datasets |
| Robustness | ✅ No cross-dataset contamination |
| Research Capability | ✅ Cross-dataset evaluation enabled |
| Ensemble Support | ✅ Multiple ensemble methods |
| Maintainability | ✅ Clean, modular code |

---

## Future Enhancements

Planned improvements:
- Transformer-based models (BERT, DistilBERT)
- Distributed training with PySpark
- Automatic hyperparameter optimization
- Real-time fake news detection API
- Online learning systems
- Multilingual fake news detection

---

## Conclusion

The implemented architecture provides a scalable and research-oriented framework for fake news detection using multiple heterogeneous datasets. By training independent models for each dataset instead of merging them, the system preserves dataset-specific characteristics and avoids feature contamination. The addition of ensemble learning and cross-dataset evaluation significantly improves robustness, generalization, and experimental depth.