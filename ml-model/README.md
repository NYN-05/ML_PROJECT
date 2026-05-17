# ML Model Pipeline (Legacy)

This directory contains the original PyTorch LSTM-based training pipeline. For the new modular multi-dataset system, see `/src`.

## Legacy Scripts

| Script | Description |
|--------|-------------|
| `train.py` | Train LSTM model with CUDA acceleration |
| `evaluate.py` | Evaluate on test split |
| `infer.py` | Single prediction inference |

## Model Architecture

```
Embedding(vocab=30k, dim=128)
    → GlobalAveragePooling1D
    → Dropout(0.3)
    → Linear(128 → 1) + Sigmoid
```

## Quick Start (Legacy)

```bash
cd ml-model
python train.py
python evaluate.py
python infer.py
```

## New Pipeline

For the new independent dataset training system:

```bash
# Run new pipeline
python -m src.main --datasets dataset --outputs outputs
```

See `/src/main.py` for all options.