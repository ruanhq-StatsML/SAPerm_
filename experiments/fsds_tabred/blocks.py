"""Feature-block constructors for TabReD FSDS arms."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def contiguous_quarters(p: int, n_blocks: int = 4) -> list[list[int]]:
    """Split ``0..p-1`` into ``n_blocks`` contiguous index blocks."""
    if n_blocks < 1:
        raise ValueError("n_blocks must be >= 1")
    edges = np.linspace(0, p, n_blocks + 1, dtype=int)
    blocks = [list(range(int(edges[i]), int(edges[i + 1]))) for i in range(n_blocks)]
    return [b for b in blocks if b]


def vimp_cluster_blocks(
    X: np.ndarray,
    y: np.ndarray,
    n_blocks: int = 4,
    seed: int = 2026,
    max_rows: int = 4000,
) -> list[list[int]]:
    """Cluster features by a global RF importance rank into equal-size blocks.

    Used for Sberbank / Homesite where hand grouping is impractical. A cheap
    RF importance stands in for a full permuCATE_vimp pass at block-construction
    time; FSDS still reweights with PO / VIMP at each batch.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()
    n, p = X.shape
    rng = np.random.default_rng(seed)
    if n > max_rows:
        idx = rng.choice(n, size=max_rows, replace=False)
        X, y = X[idx], y[idx]
    clf = RandomForestClassifier(
        n_estimators=80,
        max_depth=max(3, int(round(np.sqrt(p)))),
        min_samples_leaf=max(1, int(round(np.sqrt(len(X)) / 2))),
        n_jobs=1,
        random_state=seed,
    )
    clf.fit(X, y.astype(int))
    order = np.argsort(-clf.feature_importances_)
    blocks: list[list[int]] = [[] for _ in range(n_blocks)]
    for rank, j in enumerate(order):
        blocks[rank % n_blocks].append(int(j))
    return [sorted(b) for b in blocks if b]


def build_blocks(
    X: np.ndarray,
    y: np.ndarray,
    mode: str = "contiguous_quarters",
    n_blocks: int = 4,
    seed: int = 2026,
) -> list[list[int]]:
    p = X.shape[1]
    if mode == "contiguous_quarters":
        return contiguous_quarters(p, n_blocks=n_blocks)
    if mode == "vimp_cluster":
        return vimp_cluster_blocks(X, y, n_blocks=n_blocks, seed=seed)
    raise ValueError(f"unknown block mode: {mode}")


def summarize_blocks(blocks: Sequence[Sequence[int]], feature_cols: Sequence[str] | None = None) -> list[dict]:
    out = []
    for i, cols in enumerate(blocks):
        entry = {"arm": i, "size": len(cols), "cols": list(cols)}
        if feature_cols is not None:
            entry["names"] = [feature_cols[j] for j in cols[:8]]
        out.append(entry)
    return out
