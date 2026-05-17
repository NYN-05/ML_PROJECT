"""Inference utilities for the fake news detection model."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch

from config import AppConfig, build_default_config
from logging_utils import get_logger
from preprocessing import TextPreprocessor
from tokenizer_utils import load_tokenizer, texts_to_padded
from utils import load_metadata

LOGGER = get_logger("ml.infer")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PredictionRequest:
    def __init__(self, title: str = "", text: str = ""):
        self.title = title
        self.text = text


@dataclass
class PredictionResult:
    prediction: str
    confidence: float
    confidence_label: str
    risk_level: str
    processing_time: str


def _confidence_label(confidence: float, config: AppConfig) -> str:
    if confidence >= config.inference.high_confidence_threshold:
        return "High Confidence"
    if confidence >= config.inference.medium_confidence_threshold:
        return "Medium Confidence"
    return "Low Confidence"


def _risk_level(prediction: str) -> str:
    if prediction == "FAKE":
        return "High"
    if prediction == "REAL":
        return "Low"
    return "Unknown"


def _load_model(config: AppConfig):
    from model_builder import build_lstm_model

    model_path = config.paths.models_dir / "fake_news_model.pt"
    metadata_path = config.paths.models_dir / "metadata.json"
    metadata = load_metadata(metadata_path)
    max_len = int(metadata.get("max_seq_len", config.preprocess.max_seq_len_max))
    model = build_lstm_model(config, max_len)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model, max_len


_model_cache = None
_max_len_cache = None


def predict(request: PredictionRequest, config: AppConfig) -> PredictionResult:
    global _model_cache, _max_len_cache

    model_path = config.paths.models_dir / "fake_news_model.pt"
    tokenizer_path = config.paths.models_dir / "tokenizer.json"
    metadata_path = config.paths.models_dir / "metadata.json"

    if not model_path.exists() or not tokenizer_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Model artifacts not found. Train the model first.")

    if _model_cache is None:
        _model_cache, _max_len_cache = _load_model(config)

    preprocessor = TextPreprocessor()
    combined = f"{request.title} {request.text}".strip()
    cleaned = preprocessor.preprocess_text(combined)

    if len(cleaned.split()) < config.preprocess.min_tokens:
        return PredictionResult(
            prediction="UNCERTAIN",
            confidence=0.5,
            confidence_label="Low Confidence",
            risk_level="Unknown",
            processing_time="0ms",
        )

    tokenizer = load_tokenizer(tokenizer_path)
    padded = texts_to_padded(tokenizer, [cleaned], _max_len_cache)

    input_tensor = torch.tensor(padded, dtype=torch.long, device=DEVICE)
    start = time.perf_counter()
    with torch.no_grad():
        output = _model_cache(input_tensor)
        probability = float(torch.sigmoid(output[0]).cpu().item())
    duration = (time.perf_counter() - start) * 1000

    confidence = max(probability, 1 - probability)
    if confidence < config.inference.medium_confidence_threshold:
        prediction = "UNCERTAIN"
    else:
        prediction = "REAL" if probability >= 0.5 else "FAKE"

    return PredictionResult(
        prediction=prediction,
        confidence=round(confidence, 2),
        confidence_label=_confidence_label(confidence, config),
        risk_level=_risk_level(prediction),
        processing_time=f"{int(duration)}ms",
    )


def main() -> None:
    config = build_default_config()
    request = PredictionRequest(
        title="Government announces new education reform",
        text="The government officially announced a nationwide reform focused on digital learning.",
    )
    result = predict(request, config)
    LOGGER.info("Prediction: %s", result)


if __name__ == "__main__":
    main()
