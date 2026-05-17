"""Dataset loading and validation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from config import AppConfig
from logging_utils import get_logger

LOGGER = get_logger("ml.data_loader")

REQUIRED_COLUMNS = {"title", "text", "subject", "date"}


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [col.strip().lower() for col in frame.columns]
    return frame


def _load_fake_true_pairs(root: Path) -> pd.DataFrame:
    frames = []
    for fake_path in root.rglob("Fake.csv"):
        true_path = fake_path.parent / "True.csv"
        if not true_path.exists():
            continue
        src = fake_path.parent.name
        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)
        fake_df = _normalize_columns(fake_df)
        true_df = _normalize_columns(true_df)
        fake_df["label"] = "FAKE"
        true_df["label"] = "REAL"
        combined = pd.concat([fake_df, true_df], ignore_index=True)
        LOGGER.info("Loaded Fake/True pair: %s (%d rows)", src, len(combined))
        frames.append(combined)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_single_file_datasets(root: Path) -> pd.DataFrame:
    frames = []
    seen = set()
    for fpath in root.rglob("*.csv"):
        name = fpath.name.lower()
        if name in ("fake.csv", "true.csv", "train_dataset.csv", "test_dataset.csv",
                     "validation_dataset.csv", "processed"):
            continue
        if fpath.parent.name == "processed":
            continue
        if fpath in seen:
            continue
        seen.add(fpath)
        try:
            df = pd.read_csv(fpath)
        except Exception:
            continue
        df = _normalize_columns(df)
        if "label" not in df.columns:
            continue
        has_title = "title" in df.columns
        has_text = "text" in df.columns or "text" in df.columns
        if not (has_title or has_text):
            continue
        if "text" not in df.columns and "text" in df.columns:
            df = df.rename(columns={"text": "text"})
        if "text" not in df.columns:
            continue
        if "title" not in df.columns:
            df["title"] = ""
        if "subject" not in df.columns:
            df["subject"] = ""
        if "date" not in df.columns:
            df["date"] = ""
        if df["label"].dtype == "object":
            df["label"] = df["label"].str.strip().str.upper().map(
                lambda x: "REAL" if x in ("1", "TRUE", "REAL") else "FAKE"
            )
        else:
            df["label"] = df["label"].map(lambda x: "REAL" if x == 1 else "FAKE")
        cols = ["title", "text", "subject", "date", "label"]
        df = df[[c for c in cols if c in df.columns]]
        LOGGER.info("Loaded single-file dataset: %s (%d rows)", fpath.parent.name, len(df))
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_all_datasets(config: AppConfig) -> pd.DataFrame:
    root = config.paths.dataset_dir
    LOGGER.info("Loading all datasets from: %s", root)

    pair_df = _load_fake_true_pairs(root)
    single_df = _load_single_file_datasets(root)

    if pair_df.empty and single_df.empty:
        raise FileNotFoundError("No usable datasets found under %s" % root)

    parts = []
    if not pair_df.empty:
        parts.append(pair_df)
    if not single_df.empty:
        parts.append(single_df)

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=config.data.random_seed).reset_index(drop=True)
    LOGGER.info("Total combined dataset: %d rows", len(combined))
    return combined


def load_primary_dataset(config: AppConfig) -> pd.DataFrame:
    return load_all_datasets(config)
