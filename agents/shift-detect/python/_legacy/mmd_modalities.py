#!/usr/bin/env python3
"""Ready software: per-modality RBF-MMD² on two batches.

Input
-----
Each modality is just a feature matrix (meta and/or embedding already
concatenated upstream if you want). Split rows into batch0 / batch1
(or pass two matrices).

Output
------
``{modality_name: mmd2}``

Example
-------
```python
from mmd_modalities import mmd_by_modality

# modality -> (n, p_m) array  (meta, embed, or concat — your choice)
mods = {
    "text":  X_text,    # e.g. M_text or [Z_text, M_text]
    "image": X_image,
    "audio": X_audio,
}
# same row index: batch0 idx / batch1 idx
out = mmd_by_modality(mods, idx0, idx1)
# -> {"text": 0.10, "image": 0.02, "audio": 0.03}
```

Or two clouds per modality:

```python
out = mmd_by_modality_pairs({
    "text": (X_text[a], X_text[b]),
    "image": (X_image[a], X_image[b]),
})
```
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


def rbf_mmd2(X: np.ndarray, Y: np.ndarray, *, max_median: int = 400, seed: int = 0) -> float:
    """Unbiased-looking mean-embedding MMD² with median heuristic bandwidth."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X, Y must be 2-d arrays")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"dim mismatch {X.shape} vs {Y.shape}")
    if len(X) == 0 or len(Y) == 0:
        return float("nan")

    rng = np.random.default_rng(seed)
    n = len(X) + len(Y)
    if n > max_median:
        ix = rng.choice(len(X), min(len(X), max_median // 2), replace=False)
        iy = rng.choice(len(Y), min(len(Y), max_median // 2), replace=False)
        Z = np.vstack([X[ix], Y[iy]])
    else:
        Z = np.vstack([X, Y])
    sq = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=-1)
    pos = sq[sq > 0]
    gamma = 1.0 / float(np.median(pos)) if pos.size else 1.0

    def kmean(A: np.ndarray, B: np.ndarray) -> float:
        d = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
        return float(np.mean(np.exp(-gamma * d)))

    return float(kmean(X, X) + kmean(Y, Y) - 2.0 * kmean(X, Y))


def mmd_by_modality_pairs(
    pairs: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    standardize: bool = True,
    max_median: int = 400,
    seed: int = 0,
) -> dict[str, float]:
    """``pairs[name] = (X0, X1)`` → ``{name: mmd2}``."""
    out: dict[str, float] = {}
    for i, (name, (X0, X1)) in enumerate(pairs.items()):
        X0 = np.asarray(X0, dtype=float)
        X1 = np.asarray(X1, dtype=float)
        if standardize:
            # fit on pooled batches so scale is shared
            mu = np.vstack([X0, X1]).mean(axis=0)
            sd = np.vstack([X0, X1]).std(axis=0) + 1e-8
            X0 = (X0 - mu) / sd
            X1 = (X1 - mu) / sd
        out[name] = rbf_mmd2(X0, X1, max_median=max_median, seed=seed + i)
    return out


def mmd_by_modality(
    modalities: Mapping[str, np.ndarray],
    idx0: np.ndarray,
    idx1: np.ndarray,
    *,
    standardize: bool = True,
    max_median: int = 400,
    seed: int = 0,
) -> dict[str, float]:
    """``modalities[name] = array (n, p)`` + row index splits → per-mod MMD²."""
    idx0 = np.asarray(idx0, dtype=int)
    idx1 = np.asarray(idx1, dtype=int)
    pairs = {
        name: (np.asarray(X, dtype=float)[idx0], np.asarray(X, dtype=float)[idx1])
        for name, X in modalities.items()
    }
    return mmd_by_modality_pairs(
        pairs, standardize=standardize, max_median=max_median, seed=seed
    )


def mmd_from_indices(
    X: np.ndarray,
    modality_indices: Mapping[str, list[int] | np.ndarray],
    idx0: np.ndarray,
    idx1: np.ndarray,
    **kwargs,
) -> dict[str, float]:
    """One big ``X`` + ``{mod: column indices}`` (meta/embed slices)."""
    X = np.asarray(X, dtype=float)
    mods = {name: X[:, np.asarray(cols, dtype=int)] for name, cols in modality_indices.items()}
    return mmd_by_modality(mods, idx0, idx1, **kwargs)


if __name__ == "__main__":
    # tiny self-check
    rng = np.random.default_rng(0)
    n, p = 80, 8
    text = rng.normal(size=(n, p))
    image = rng.normal(size=(n, p))
    image[40:] += 1.5  # planted image shift on second half
    idx0, idx1 = np.arange(40), np.arange(40, 80)
    print(mmd_by_modality({"text": text, "image": image}, idx0, idx1))
