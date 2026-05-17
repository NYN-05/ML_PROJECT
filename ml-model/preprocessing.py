"""Text preprocessing utilities."""

from __future__ import annotations

import re
from typing import Iterable, List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from logging_utils import get_logger

LOGGER = get_logger("ml.preprocessing")


def ensure_nltk_assets() -> None:
    """Ensure required NLTK resources are available."""
    resources = ["stopwords", "wordnet", "omw-1.4"]
    for resource in resources:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource)


class TextPreprocessor:
    """Text preprocessing pipeline with cleaning and normalization."""

    def __init__(self) -> None:
        ensure_nltk_assets()
        self._stopwords = set(stopwords.words("english"))
        self._lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        """Clean raw text input.

        Args:
            text: Raw input text.

        Returns:
            Cleaned text.
        """
        text = text.lower()
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def normalize_tokens(self, tokens: Iterable[str]) -> List[str]:
        """Remove stopwords and lemmatize tokens.

        Args:
            tokens: Token sequence.

        Returns:
            Normalized token list.
        """
        normalized = []
        for token in tokens:
            if token in self._stopwords:
                continue
            normalized.append(self._lemmatizer.lemmatize(token))
        return normalized

    def preprocess_text(self, text: str) -> str:
        """Apply full preprocessing pipeline.

        Args:
            text: Raw input text.

        Returns:
            Preprocessed text.
        """
        cleaned = self.clean_text(text)
        tokens = cleaned.split()
        normalized = self.normalize_tokens(tokens)
        return " ".join(normalized)
