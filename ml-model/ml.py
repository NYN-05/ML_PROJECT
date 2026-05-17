"""Unified ML module for DistilBERT fake news detection - minimal implementation."""

import json
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Config
MODEL_DIR = Path("artifacts/models_distilbert")
MAX_SAMPLES = 6000
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 3
LR = 2e-5

DATASETS = [
    ("WELFake_Dataset", "../dataset/WELFake_Dataset/WELFake_Dataset.csv"),
    ("FakeNewsDet1", "../dataset/Fake News Detection_1/fake.csv"),
    ("FakeNewsDet2", "../dataset/Fake News Detection Dataset/fake_news_dataset.csv"),
    ("FakeRealNews", "../dataset/Fake or Real News/fake_or_real_news.csv"),
    ("ISOT", "../dataset/ISOT Fake News Dataset/Fake.csv"),
    ("NewsDet", "../dataset/News Detection (Fake or Real) Dataset/fake_and_real_news.csv"),
    ("FakeReal", "../dataset/fake-and-real-news-dataset/Fake.csv"),
]


class DistilBERTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))

    def forward(self, ids, mask):
        out = self.bert(ids, mask).last_hidden_state[:, 0]
        return self.fc(self.dropout(out))


def load_data(path: Path) -> Optional[Tuple[List[str], List[str]]]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        text_col = next((c for c in df.columns if "text" in c.lower()), df.columns[0])
        label_col = next((c for c in df.columns if c.lower() == "label"), df.columns[-1])
        texts = [str(t) for t in df[text_col].tolist()[:MAX_SAMPLES] if len(str(t)) > 10]
        labels = ["REAL" if str(l).upper() in ["1", "TRUE", "REAL"] else "FAKE" for l in df[label_col].tolist()[:MAX_SAMPLES]]
        valid = [(t, l) for t, l in zip(texts, labels) if len(t) > 10]
        return [v[0] for v in valid], [v[1] for v in valid]
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def train_model(texts: List[str], labels: List[str], save_path: Path, epochs: int = EPOCHS) -> Dict:
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBERTClassifier().to(DEVICE)

    train_txt, val_txt, train_lbl, val_lbl = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)

    train_enc = tokenizer(train_txt, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    val_enc = tokenizer(val_txt, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")

    train_ds = TensorDataset(torch.tensor(train_enc["input_ids"]), torch.tensor(train_enc["attention_mask"]), torch.tensor([{"FAKE": 0, "REAL": 1}[l] for l in train_lbl]))
    val_ds = TensorDataset(torch.tensor(val_enc["input_ids"]), torch.tensor(val_enc["attention_mask"]), torch.tensor([{"FAKE": 0, "REAL": 1}[l] for l in val_lbl]))

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for ids, mask, y in train_dl:
            ids, mask, y = ids.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(ids, mask), y)
            loss.backward()
            optimizer.step()

        model.eval()
        preds, true = [], []
        with torch.no_grad():
            for ids, mask, y in val_dl:
                preds.extend(model(ids.to(DEVICE), mask.to(DEVICE)).argmax(dim=1).cpu().numpy())
                true.extend(y.numpy())

        f1 = f1_score(true, preds, average="weighted")
        acc = accuracy_score(true, preds)
        print(f"  Epoch {epoch+1}: Acc={acc:.4f}, F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    save_path.mkdir(parents=True, exist_ok=True)
    torch.save({"fc_state": model.fc.state_dict(), "best_f1": best_f1}, save_path / "model.pt")
    tokenizer.save_pretrained(str(save_path / "tokenizer"))

    return {"accuracy": acc, "f1": best_f1}


def recursive_train() -> None:
    """Train on all datasets recursively."""
    print("\n=== Loading datasets ===")
    all_data = []
    for name, path in DATASETS:
        data = load_data(Path(path))
        if data:
            print(f"  {name}: {len(data[0])} samples")
            all_data.append((name, data))

    if not all_data:
        print("No data loaded!")
        return

    # Initialize
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBERTClassifier().to(DEVICE)
    model.fc.load_state_dict({"0.weight": torch.zeros(256, 768), "0.bias": torch.zeros(256), "2.weight": torch.zeros(2, 256), "2.bias": torch.zeros(2)})

    best_f1 = 0
    all_metrics = []

    for iteration in range(5):
        print(f"\n=== Iteration {iteration + 1} ===")
        improved = False

        for name, (texts, labels) in all_data:
            print(f"\n[{name}]")
            train_txt, val_txt, train_lbl, val_lbl = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)

            train_enc = tokenizer(train_txt, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            val_enc = tokenizer(val_txt, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")

            train_ds = TensorDataset(torch.tensor(train_enc["input_ids"]), torch.tensor(train_enc["attention_mask"]), torch.tensor([{"FAKE": 0, "REAL": 1}[l] for l in train_lbl]))
            train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

            optimizer = torch.optim.AdamW(model.fc.parameters(), lr=LR)
            criterion = nn.CrossEntropyLoss()

            for epoch in range(EPOCHS):
                model.train()
                for ids, mask, y in train_dl:
                    ids, mask, y = ids.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
                    out = model(ids, mask)
                    loss = criterion(out, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # Evaluate
            val_ds = TensorDataset(torch.tensor(val_enc["input_ids"]), torch.tensor(val_enc["attention_mask"]), torch.tensor([{"FAKE": 0, "REAL": 1}[l] for l in val_lbl]))
            val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

            model.eval()
            preds, true = [], []
            with torch.no_grad():
                for ids, mask, y in val_dl:
                    preds.extend(model(ids.to(DEVICE), mask.to(DEVICE)).argmax(dim=1).cpu().numpy())
                    true.extend(y.numpy())

            f1 = f1_score(true, preds, average="weighted")
            print(f"  F1: {f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                improved = True
                torch.save({"fc_state": model.fc.state_dict(), "best_f1": best_f1}, MODEL_DIR / "model.pt")
                tokenizer.save_pretrained(str(MODEL_DIR / "tokenizer"))
                print(f"  ✓ Saved (best: {best_f1:.4f})")

        if not improved:
            print("\n*** Converged ***")
            break

    print(f"\nBest F1: {best_f1:.4f}")


def predict(text: str, threshold: float = 0.5) -> Dict:
    """Predict on text."""
    if not MODEL_DIR.exists():
        return {"error": "Model not found. Run training first."}

    tokenizer = DistilBertTokenizer.from_pretrained(str(MODEL_DIR / "tokenizer"))
    model = DistilBERTClassifier().to(DEVICE)
    checkpoint = torch.load(MODEL_DIR / "model.pt", weights_only=False)
    model.fc.load_state_dict(checkpoint["fc_state"])
    model.eval()

    enc = tokenizer(text, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)), dim=1)
        real_prob = probs[0][1].item()

    pred = "REAL" if real_prob >= threshold else "FAKE"
    return {"prediction": pred, "confidence": round(real_prob if pred == "REAL" else probs[0][0].item(), 4), "method": "distilbert"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DistilBERT Fake News Detection")
    parser.add_argument("--mode", choices=["train", "recursive", "predict"], default="predict")
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.mode == "recursive":
        recursive_train()
    elif args.mode == "predict":
        if args.text:
            print(json.dumps(predict(args.text, args.threshold), indent=2))
        else:
            print("Provide --text for prediction")
    else:
        print("Use --mode recursive for full training")