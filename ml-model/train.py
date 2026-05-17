"""Training pipeline for the fake news detection model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

from config import AppConfig, build_default_config
from data_loader import load_primary_dataset
from logging_utils import get_logger
from metrics import compute_metrics, save_classification_report, save_confusion_matrix_plot
from model_builder import build_lstm_model
from preprocessing import TextPreprocessor
from tokenizer_utils import (
    WordTokenizer,
    build_tokenizer,
    compute_sequence_length,
    save_tokenizer,
    texts_to_padded,
)
from utils import save_metadata

LOGGER = get_logger("ml.train")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.labels[idx]


def _prepare_features(frame: pd.DataFrame, preprocessor: TextPreprocessor) -> pd.DataFrame:
    frame = frame.copy()
    frame["title"] = frame["title"].fillna("")
    frame["text"] = frame["text"].fillna("")
    frame["subject"] = frame["subject"].fillna("Unknown")
    combined = frame["title"].astype(str) + " " + frame["text"].astype(str)
    frame["clean_text"] = combined.apply(preprocessor.preprocess_text)
    return frame


def _encode_labels(labels: pd.Series) -> np.ndarray:
    return labels.map({"REAL": 1, "FAKE": 0}).astype(int).to_numpy()


def _save_processed_splits(
    config: AppConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    config.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(config.paths.processed_dir / "train_dataset.csv", index=False)
    val_df.to_csv(config.paths.processed_dir / "validation_dataset.csv", index=False)
    test_df.to_csv(config.paths.processed_dir / "test_dataset.csv", index=False)


def train_model(config: AppConfig) -> Dict[str, float]:
    torch.manual_seed(config.data.random_seed)
    np.random.seed(config.data.random_seed)

    raw_df = load_primary_dataset(config)
    preprocessor = TextPreprocessor()
    processed_df = _prepare_features(raw_df, preprocessor)

    processed_df = processed_df.dropna(subset=["clean_text"])
    processed_df = processed_df[processed_df["clean_text"].str.len() > 0]
    processed_df = processed_df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)

    labels = _encode_labels(processed_df["label"])
    train_df, temp_df, y_train, y_temp = train_test_split(
        processed_df,
        labels,
        test_size=config.data.validation_split + config.data.test_split,
        stratify=labels,
        random_state=config.data.random_seed,
    )
    val_size = config.data.validation_split / (
        config.data.validation_split + config.data.test_split
    )
    val_df, test_df, y_val, y_test = train_test_split(
        temp_df,
        y_temp,
        test_size=1 - val_size,
        stratify=y_temp,
        random_state=config.data.random_seed,
    )

    _save_processed_splits(config, train_df, val_df, test_df)

    max_len = compute_sequence_length(
        train_df["clean_text"].tolist(),
        min_len=config.preprocess.max_seq_len_min,
        max_len=config.preprocess.max_seq_len_max,
        percentile=config.preprocess.seq_len_percentile,
    )
    LOGGER.info("Using max sequence length: %s", max_len)

    tokenizer = build_tokenizer(train_df["clean_text"].tolist(), config)
    x_train = texts_to_padded(tokenizer, train_df["clean_text"].tolist(), max_len)
    x_val = texts_to_padded(tokenizer, val_df["clean_text"].tolist(), max_len)
    x_test = texts_to_padded(tokenizer, test_df["clean_text"].tolist(), max_len)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y_train,
    )
    class_weight_map = {0: class_weights[0], 1: class_weights[1]}
    weight_tensor = torch.tensor(
        [class_weight_map[0], class_weight_map[1]], dtype=torch.float, device=DEVICE
    )

    train_ds = TextDataset(x_train, y_train)
    val_ds = TextDataset(x_val, y_val)
    test_ds = TextDataset(x_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.training.batch_size)
    test_loader = DataLoader(test_ds, batch_size=config.training.batch_size)

    model = build_lstm_model(config, max_len)
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor[1] / weight_tensor[0])

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(config.training.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for sequences, labels_batch in train_loader:
            sequences, labels_batch = sequences.to(DEVICE), labels_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels_batch)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels_batch).sum().item()
            total += len(labels_batch)

        train_loss = total_loss / total
        train_acc = correct / total

        model.eval()
        val_loss_total = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for sequences, labels_batch in val_loader:
                sequences, labels_batch = sequences.to(DEVICE), labels_batch.to(DEVICE)
                outputs = model(sequences)
                loss = criterion(outputs, labels_batch)

                val_loss_total += loss.item() * len(labels_batch)
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                val_correct += (preds == labels_batch).sum().item()
                val_total += len(labels_batch)

        val_loss = val_loss_total / val_total
        val_acc = val_correct / val_total

        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        LOGGER.info(
            "Epoch %d/%d - loss: %.4f - accuracy: %.4f - val_loss: %.4f - val_accuracy: %.4f",
            epoch + 1, config.training.epochs, train_loss, train_acc, val_loss, val_acc,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_path = config.paths.models_dir / "fake_news_model.pt"
            config.paths.models_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= config.training.early_stopping_patience:
                LOGGER.info("Early stopping triggered after %d epochs", epoch + 1)
                break

    model.load_state_dict(torch.load(config.paths.models_dir / "fake_news_model.pt"))
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for sequences, labels_batch in test_loader:
            sequences = sequences.to(DEVICE)
            outputs = model(sequences)
            preds = (torch.sigmoid(outputs) >= 0.5).cpu().float().numpy().astype(int)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels_batch.numpy().tolist())

    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))

    config.paths.models_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.paths.models_dir / "fake_news_model.pt"
    torch.save(model.state_dict(), model_path)
    save_tokenizer(tokenizer, config.paths.models_dir / "tokenizer.json")
    save_metadata(
        config.paths.models_dir / "metadata.json",
        {
            "max_seq_len": max_len,
            "vocab_size": config.tokenizer.vocab_size,
            "embedding_dim": config.model.embedding_dim,
        },
    )

    save_classification_report(
        np.array(all_labels),
        np.array(all_preds),
        config.paths.eval_dir / "classification_report.txt",
    )
    save_confusion_matrix_plot(
        np.array(all_labels),
        np.array(all_preds),
        config.paths.eval_dir / "confusion_matrix.png",
    )

    history_path = config.paths.eval_dir / "training_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    metrics_path = config.paths.eval_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


def main() -> None:
    LOGGER.info("Using device: %s", DEVICE)
    config = build_default_config()
    metrics = train_model(config)
    LOGGER.info("Training complete: %s", metrics)


if __name__ == "__main__":
    main()
