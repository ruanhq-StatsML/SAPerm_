"""Shared digestion helpers (paths, cache, PCA, Digested)."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PY_ROOT = Path(__file__).resolve().parents[1]   # .../python
REPO = Path(__file__).resolve().parents[4]      # SAPerm repo root
OUT = PY_ROOT / "results"

if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))


@dataclass
class Digested:
    dataset: str
    Z: np.ndarray | None  # PCA embed, or None if meta-only
    M: np.ndarray
    y: np.ndarray
    texts: list[str]
    meta_names: list[str]
    schema: dict[str, Any]
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(len(self.y))

    def X(self) -> np.ndarray:
        return self.M if self.Z is None else np.hstack([self.Z, self.M])

    def to_bundle(self) -> dict[str, Any]:
        pca_k = int(self.schema.get("pca_k", 0 if self.Z is None else self.Z.shape[1]))
        return {
            "Zpca": self.Z,
            "M": self.M,
            "y": self.y,
            "texts": self.texts,
            "schema": self.schema,
            "x_names": [f"pca_{i}" for i in range(pca_k)] + list(self.meta_names),
            "dataset": self.dataset,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            Z=self.Z if self.Z is not None else np.array([]),
            M=self.M,
            y=self.y,
            meta_names=np.array(self.meta_names, dtype=object),
            texts=np.array(self.texts, dtype=object),
        )


def stratified_cap(y: np.ndarray, n_max: int | None, seed: int = 0) -> np.ndarray:
    """Return sorted indices; identity if n_max is None / large enough."""
    n = len(y)
    if n_max is None or n_max >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    k0, k1 = n_max // 2, n_max - n_max // 2
    take = np.concatenate(
        [
            rng.choice(i0, min(k0, len(i0)), replace=False),
            rng.choice(i1, min(k1, len(i1)), replace=False),
        ]
    )
    take.sort()
    return take


def pca_embed(Z: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    return PCA(k, random_state=seed).fit_transform(StandardScaler().fit_transform(Z))


def cached_text_meta(texts: list[str], cache: Path | None = None) -> tuple[np.ndarray, list[str]]:
    """spaCy+VADER matrix in FEATURE_NAMES order, with optional npz cache."""
    from extract_features import FEATURE_NAMES, build_matrix

    if cache is not None and cache.exists():
        data = np.load(cache, allow_pickle=True)
        names = list(data["names"])
        if names == FEATURE_NAMES:
            return data["M"], names
    M_list, names = build_matrix(texts, "meta")
    M = np.asarray(M_list, dtype=float)
    j = {n: i for i, n in enumerate(names)}
    M = M[:, [j[n] for n in FEATURE_NAMES]]
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, M=M, names=np.array(FEATURE_NAMES, dtype=object))
    return M, list(FEATURE_NAMES)
