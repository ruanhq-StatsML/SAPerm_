"""Shared digestion helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Digested:
    """Output of one dataset digestion script."""

    dataset: str
    Z: np.ndarray | None          # embeddings (n, p_z) or None if meta-only
    M: np.ndarray                 # meta features (n, p_m)
    y: np.ndarray                 # labels
    texts: list[str]
    meta_names: list[str]
    schema: dict[str, Any]
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.y)

    def X(self, pca_z: np.ndarray | None = None) -> np.ndarray:
        """Build X=[Z_or_pca, M]."""
        if pca_z is not None:
            return np.hstack([pca_z, self.M])
        if self.Z is None:
            return self.M
        return np.hstack([self.Z, self.M])
