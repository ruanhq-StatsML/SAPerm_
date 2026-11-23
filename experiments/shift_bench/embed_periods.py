"""Omni3 / Llama3 period embedding — discretize → embed → align head → concat.

Pipeline (no cross-modality fusion beyond a shared head):
  1. discretize time → period id
  2. embed each row (Omni3 / Llama3 / hashing)
  3. **AlignHead**: shared linear map so every period lives in one space
  4. place the aligned vector into its time-window slot and **concatenate**
     the K windows → one X row

Backends: ``hashing`` | ``llama3`` | ``omni3`` | ``callable``
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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
        ids = np.digitize(t, edges[1:-1], right=False)
        return np.clip(ids, 0, n_periods - 1).astype(int)
    qs = np.linspace(0, 1, n_periods + 1)
    edges = np.unique(np.quantile(t, qs))
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
    """Embed a list of strings → ``(n, d)``."""
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
        try:
            from llma3_embedding import llama3_embedding  # type: ignore

            emb = llama3_embedding(texts, n_dim=dim)
            return np.asarray(emb, dtype=float)
        except Exception:
            from sklearn.feature_extraction.text import HashingVectorizer

            vec = HashingVectorizer(n_features=dim, alternate_sign=False, norm="l2")
            return vec.transform(texts).toarray().astype(float)

    raise ValueError(f"unknown backend: {backend}")


class AlignHead:
    """Shared linear head: scale → project to ``out_dim`` → optional L2 norm.

    Fits once on all period embeds so every time window sits in the same
    coordinates before we concatenate the windows.
    """

    def __init__(self, out_dim: int = 32, *, normalize: bool = True, seed: int = 0):
        self.out_dim = int(out_dim)
        self.normalize = bool(normalize)
        self.seed = int(seed)
        self.scaler_: StandardScaler | None = None
        self.pca_: PCA | None = None
        self.in_dim_: int | None = None

    def fit(self, embeds: np.ndarray) -> "AlignHead":
        E = np.asarray(embeds, dtype=float)
        if E.ndim != 2 or E.shape[0] == 0:
            raise ValueError("embeds must be (n, d) with n>0")
        self.in_dim_ = int(E.shape[1])
        self.scaler_ = StandardScaler().fit(E)
        Z = self.scaler_.transform(E)
        k = max(1, min(self.out_dim, Z.shape[0], Z.shape[1]))
        self.pca_ = PCA(n_components=k, random_state=self.seed).fit(Z)
        self.out_dim = k
        return self

    def transform(self, embeds: np.ndarray) -> np.ndarray:
        if self.scaler_ is None or self.pca_ is None:
            raise RuntimeError("AlignHead is not fitted")
        E = np.asarray(embeds, dtype=float)
        Z = self.pca_.transform(self.scaler_.transform(E))
        if Z.shape[1] < self.out_dim:
            # pad if PCA rank-deficient relative to requested dim
            pad = np.zeros((Z.shape[0], self.out_dim - Z.shape[1]), dtype=float)
            Z = np.hstack([Z, pad])
        if self.normalize:
            nrm = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
            Z = Z / nrm
        return Z.astype(float)

    def fit_transform(self, embeds: np.ndarray) -> np.ndarray:
        return self.fit(embeds).transform(embeds)


def align_period_embeds(
    embeds: np.ndarray,
    *,
    align_dim: int = 32,
    normalize: bool = True,
    seed: int = 0,
    head: AlignHead | None = None,
) -> tuple[np.ndarray, AlignHead]:
    """Run the shared AlignHead. Returns aligned embeds and the fitted head."""
    h = head if head is not None else AlignHead(out_dim=align_dim, normalize=normalize, seed=seed)
    if head is None:
        aligned = h.fit_transform(embeds)
    else:
        aligned = h.transform(embeds)
    return aligned, h


def concat_period_embeds(
    embeds: np.ndarray,
    period_ids: np.ndarray,
    *,
    n_periods: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Concatenate time windows: slot ``k`` gets the (aligned) embed for that period.

    Per row: zeros everywhere except period ``k``, then hstack the K slots.
    That is literally concatenate-the-time-windows.
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
        if 0 <= k < K:
            X[i, k * d : (k + 1) * d] = embeds[i]
    names = [f"p{k}_a{j}" for k in range(K) for j in range(d)]
    return X, names


def pool_entity_windows(
    embeds: np.ndarray,
    period_ids: np.ndarray,
    entity_ids: np.ndarray,
    *,
    n_periods: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """One row per entity: mean-pool embeds inside each period, then concat K windows.

    Use when the same entity appears in several time periods.
    """
    embeds = np.asarray(embeds, dtype=float)
    period_ids = np.asarray(period_ids, dtype=int).reshape(-1)
    entity_ids = np.asarray(entity_ids)
    entities = np.unique(entity_ids)
    d = embeds.shape[1]
    X = np.zeros((len(entities), n_periods * d), dtype=float)
    for r, ent in enumerate(entities):
        mask = entity_ids == ent
        for k in range(n_periods):
            mk = mask & (period_ids == k)
            if not np.any(mk):
                continue
            X[r, k * d : (k + 1) * d] = embeds[mk].mean(axis=0)
    names = [f"p{k}_a{j}" for k in range(n_periods) for j in range(d)]
    return X, entities, names


def embed_concat(
    texts: list[str],
    times,
    *,
    n_periods: int = 4,
    method: str = "quantile",
    backend: str = "hashing",
    dim: int = 64,
    embed_fn: Callable | None = None,
    align: bool = True,
    align_dim: int = 32,
    normalize: bool = True,
    seed: int = 0,
    entity_ids=None,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """discretize → embed → AlignHead → concatenate time windows.

    Returns ``(X, names, period_ids)``. If ``entity_ids`` is given, ``X`` has
    one row per entity (pooled per window then concat).
    """
    period_ids = discretize_periods(times, n_periods=n_periods, method=method)
    embeds = embed_texts(texts, backend=backend, dim=dim, embed_fn=embed_fn)
    head = None
    if align:
        embeds, head = align_period_embeds(
            embeds, align_dim=align_dim, normalize=normalize, seed=seed
        )
    if entity_ids is not None:
        X, _ents, names = pool_entity_windows(
            embeds, period_ids, np.asarray(entity_ids), n_periods=n_periods
        )
        return X, names, period_ids
    X, names = concat_period_embeds(embeds, period_ids, n_periods=n_periods)
    return X, names, period_ids
