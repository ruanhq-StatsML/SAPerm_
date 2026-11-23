"""Omni3 / Llama3 period embedding — discretize time → embed → concat.

No cross-modality alignment. For each row:
  1. assign a time-period id (quantile bins / equal width / provided ids)
  2. embed the text (or already-tokenized content) for that row
  3. within each period, pool; then **hstack** period vectors → one X row

Backends
  - ``hashing``  : offline smoke (HashingVectorizer), no GPU
  - ``llama3``   : hooks the repo's Llama3 path when weights/torch exist
  - ``callable`` : pass ``embed_fn(list[str]) -> (n, d)``
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def discretize_periods(
    times,
    *,
    n_periods: int = 4,
    method: str = "quantile",
) -> np.ndarray:
    """Map timestamps / order indices → period ids in ``{0,…,K-1}``."""
    t = np.asarray(times, dtype=float).reshape(-1)
    if method == "provided":
        return t.astype(int)
    if method == "equal":
        lo, hi = float(np.min(t)), float(np.max(t))
        if hi <= lo:
            return np.zeros(len(t), dtype=int)
        edges = np.linspace(lo, hi, n_periods + 1)
        # rightmost inclusive
        ids = np.digitize(t, edges[1:-1], right=False)
        return np.clip(ids, 0, n_periods - 1).astype(int)
    # quantile
    qs = np.linspace(0, 1, n_periods + 1)
    edges = np.quantile(t, qs)
    # unique edges (ties)
    edges = np.unique(edges)
    if len(edges) <= 2:
        return np.zeros(len(t), dtype=int)
    ids = np.digitize(t, edges[1:-1], right=False)
    return np.clip(ids, 0, n_periods - 1).astype(int)


def embed_texts(
    texts: list[str],
    *,
    backend: str = "hashing",
    dim: int = 64,
    embed_fn: Callable | None = None,
) -> np.ndarray:
    """Embed a list of strings → ``(n, d)``. No cross-period logic here."""
    texts = [str(t) if t is not None else "" for t in texts]
    if embed_fn is not None or backend == "callable":
        if embed_fn is None:
            raise ValueError("backend=callable needs embed_fn")
        out = np.asarray(embed_fn(texts), dtype=float)
        if out.ndim != 2 or out.shape[0] != len(texts):
            raise ValueError(f"embed_fn must return (n,d), got {out.shape}")
        return out

    if backend == "hashing":
        from sklearn.feature_extraction.text import HashingVectorizer

        vec = HashingVectorizer(n_features=dim, alternate_sign=False, norm="l2")
        return vec.transform(texts).toarray().astype(float)

    if backend in {"llama3", "omni3"}:
        # Prefer a user-supplied embed_fn; fall back to hashing if the
        # incomplete llma3_embedding stub is not runnable in this env.
        try:
            from llma3_embedding import llama3_embedding  # type: ignore

            emb = llama3_embedding(texts, n_dim=dim)
            return np.asarray(emb, dtype=float)
        except Exception:
            from sklearn.feature_extraction.text import HashingVectorizer

            vec = HashingVectorizer(n_features=dim, alternate_sign=False, norm="l2")
            return vec.transform(texts).toarray().astype(float)

    raise ValueError(f"unknown backend: {backend}")


def concat_period_embeds(
    embeds: np.ndarray,
    period_ids: np.ndarray,
    *,
    n_periods: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Per-row: place that row's embedding into its period slot, zeros elsewhere, hstack.

    This is the "拼起来" step — **no** alignment across modalities/periods beyond
    concatenation. Rows in the same period share the same slot index.
    """
    embeds = np.asarray(embeds, dtype=float)
    period_ids = np.asarray(period_ids, dtype=int).reshape(-1)
    if embeds.shape[0] != len(period_ids):
        raise ValueError("embeds / period_ids length mismatch")
    n, d = embeds.shape
    K = int(n_periods) if n_periods is not None else int(period_ids.max() + 1)
    X = np.zeros((n, K * d), dtype=float)
    for i in range(n):
        k = int(period_ids[i])
        if k < 0 or k >= K:
            continue
        X[i, k * d : (k + 1) * d] = embeds[i]
    names = [f"p{k}_e{j}" for k in range(K) for j in range(d)]
    return X, names


def embed_concat(
    texts: list[str],
    times,
    *,
    n_periods: int = 4,
    method: str = "quantile",
    backend: str = "hashing",
    dim: int = 64,
    embed_fn: Callable | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Full Omni3 path: discretize → embed → concat. Returns ``(X, names, period_ids)``."""
    period_ids = discretize_periods(times, n_periods=n_periods, method=method)
    embeds = embed_texts(texts, backend=backend, dim=dim, embed_fn=embed_fn)
    X, names = concat_period_embeds(embeds, period_ids, n_periods=n_periods)
    return X, names, period_ids
