"""Model architecture builder."""

from __future__ import annotations

import torch
import torch.nn as nn

from config import AppConfig


class FakeNewsClassifier(nn.Module):
    def __init__(self, config: AppConfig, max_len: int):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=config.tokenizer.vocab_size,
            embedding_dim=config.model.embedding_dim,
        )
        self.dropout = nn.Dropout(config.model.dropout)
        self.classifier = nn.Linear(config.model.embedding_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x.squeeze(1)


def build_lstm_model(config: AppConfig, max_len: int) -> FakeNewsClassifier:
    return FakeNewsClassifier(config, max_len)
