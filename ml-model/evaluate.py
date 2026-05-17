"""Evaluation script for the saved model."""

from __future__ import annotations

import pandas as pd
import torch

from config import AppConfig, build_default_config
from logging_utils import get_logger
from metrics import compute_metrics, save_classification_report, save_confusion_matrix_plot
from model_builder import build_lstm_model
from preprocessing import TextPreprocessor
from tokenizer_utils import load_tokenizer, texts_to_padded
from utils import load_metadata

LOGGER = get_logger("ml.evaluate")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(config: AppConfig) -> None:
    test_path = config.paths.processed_dir / "test_dataset.csv"
    if not test_path.exists():
        LOGGER.error("Test dataset not found at %s. Train first.", test_path)
        return

    test_df = pd.read_csv(test_path)
    preprocessor = TextPreprocessor()
    test_df["clean_text"] = test_df["clean_text"].fillna("")

    texts = test_df["clean_text"].astype(str).tolist()
    labels = test_df["label"].map({"REAL": 1, "FAKE": 0}).astype(int).to_numpy()

    model_path = config.paths.models_dir / "fake_news_model.pt"
    tokenizer_path = config.paths.models_dir / "tokenizer.json"
    metadata_path = config.paths.models_dir / "metadata.json"

    if not model_path.exists() or not tokenizer_path.exists() or not metadata_path.exists():
        LOGGER.error("Model artifacts missing. Train the model first.")
        return

    tokenizer = load_tokenizer(tokenizer_path)
    metadata = load_metadata(metadata_path)
    max_len = int(metadata.get("max_seq_len", 400))

    x_test = texts_to_padded(tokenizer, texts, max_len)
    model = build_lstm_model(config, max_len)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    test_tensor = torch.tensor(x_test, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        outputs = model(test_tensor)
        predictions = (torch.sigmoid(outputs) >= 0.5).cpu().numpy().astype(int)

    metrics = compute_metrics(labels, predictions)
    LOGGER.info("Evaluation metrics: %s", metrics)

    save_classification_report(
        labels,
        predictions,
        config.paths.eval_dir / "classification_report_eval.txt",
    )
    save_confusion_matrix_plot(
        labels,
        predictions,
        config.paths.eval_dir / "confusion_matrix_eval.png",
    )


def main() -> None:
    config = build_default_config()
    evaluate_model(config)


if __name__ == "__main__":
    main()
