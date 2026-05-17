"""Tokenizer helpers for sequence preparation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from config import AppConfig
from logging_utils import get_logger

LOGGER = get_logger("ml.tokenizer")


class WordTokenizer:
    def __init__(self, vocab_size: int = 30000, oov_token: str = "<OOV>"):
        self.vocab_size = vocab_size
        self.oov_token = oov_token
        self.word_index: Dict[str, int] = {}
        self.index_word: Dict[int, str] = {}
        self.oov_index: int = 1

    def fit_on_texts(self, texts: Iterable[str]) -> None:
        counter = Counter()
        for text in texts:
            counter.update(text.split())
        most_common = counter.most_common(self.vocab_size - 2)
        self.word_index = {self.oov_token: 1}
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word_index[word] = idx
        self.index_word = {v: k for k, v in self.word_index.items()}

    def texts_to_sequences(self, texts: Iterable[str]) -> List[List[int]]:
        max_idx = self.vocab_size - 1
        sequences = []
        for text in texts:
            seq = []
            for word in text.split():
                idx = self.word_index.get(word, self.oov_index)
                seq.append(min(idx, max_idx))
            sequences.append(seq)
        return sequences

    def to_json(self) -> str:
        data = {
            "vocab_size": self.vocab_size,
            "oov_token": self.oov_token,
            "word_index": self.word_index,
        }
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> WordTokenizer:
        data = json.loads(json_str)
        tok = cls(vocab_size=data["vocab_size"], oov_token=data["oov_token"])
        tok.word_index = data["word_index"]
        tok.index_word = {int(v): k for k, v in tok.word_index.items()}
        return tok


def build_tokenizer(texts: Iterable[str], config: AppConfig) -> WordTokenizer:
    tokenizer = WordTokenizer(
        vocab_size=config.tokenizer.vocab_size,
        oov_token=config.tokenizer.oov_token,
    )
    tokenizer.fit_on_texts(texts)
    return tokenizer


def save_tokenizer(tokenizer: WordTokenizer, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(tokenizer.to_json(), encoding="utf-8")


def load_tokenizer(input_path: Path) -> WordTokenizer:
    return WordTokenizer.from_json(input_path.read_text(encoding="utf-8"))


def compute_sequence_length(
    texts: Iterable[str],
    min_len: int,
    max_len: int,
    percentile: int,
) -> int:
    lengths = [len(text.split()) for text in texts]
    if not lengths:
        return min_len
    value = int(np.percentile(lengths, percentile))
    return max(min_len, min(max_len, value))


def texts_to_padded(
    tokenizer: WordTokenizer,
    texts: Iterable[str],
    max_len: int,
) -> np.ndarray:
    sequences = tokenizer.texts_to_sequences(texts)
    padded = np.zeros((len(sequences), max_len), dtype=np.int32)
    for i, seq in enumerate(sequences):
        trunc = seq[:max_len] if len(seq) > max_len else seq
        padded[i, : len(trunc)] = trunc
    return padded
