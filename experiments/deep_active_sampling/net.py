"""Classifier used by every query rule.

The forward pass returns ``(logits, embedding)``, the same pair the deep
active learning strategies read. Dropout for BALD is applied outside the
module, on the embedding, so training itself stays deterministic.
"""

from __future__ import annotations

import torch
from torch import nn


class EmbeddingNet(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):
        embedding = self.backbone(x)
        return self.head(embedding), embedding

    def get_embedding_dim(self) -> int:
        return int(self.head.in_features)
