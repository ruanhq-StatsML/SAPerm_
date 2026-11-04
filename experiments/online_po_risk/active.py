"""Tabular query rules corresponding to deep active learning strategies.

The pool is the candidate recent slice. Each rule returns ``budget`` indices
into that pool.

- ``recent`` keeps the latest rows, which is the plain online window.
- ``random`` is RandomSampling.
- ``entropy`` ranks a logistic domain classifier by binary entropy. For a
  binary batch label this is the same order as least confidence and margin
  sampling, so those three are one rule here.
- ``kcenter`` is KCenterGreedy, seeded with the reference rows.
- ``badge`` builds the regression gradient embedding ``x * residual`` from a
  reference outcome fit and draws a k-means++ batch, as in BADGE.
- ``learning_loss`` takes the pool rows with the largest absolute residual
  from that same reference fit.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

STRATEGIES = ("recent", "random", "entropy", "kcenter", "badge", "learning_loss")


def _top_k(scores: np.ndarray, budget: int, rng: np.random.Generator) -> np.ndarray:
    noise = rng.uniform(0.0, 1e-8, size=scores.shape[0])
    order = np.argsort(-(np.asarray(scores, dtype=float) + noise))
    return order[:budget].astype(int)


def _domain_probability(ref_x, pool_x, seed: int) -> np.ndarray:
    x = np.vstack([np.asarray(ref_x, dtype=float), np.asarray(pool_x, dtype=float)])
    y = np.concatenate(
        [
            np.zeros(len(ref_x), dtype=int),
            np.ones(len(pool_x), dtype=int),
        ]
    )
    if np.unique(y).size < 2:
        return np.full(len(pool_x), 0.5)
    model = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=250, random_state=int(seed) % (2**31 - 1)
    )
    model.fit(x, y)
    proba = model.predict_proba(np.asarray(pool_x, dtype=float))
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(pool_x), dtype=float)
    return np.asarray(proba[:, classes.index(1)], dtype=float)


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def _kcenter(ref_x, pool_x, budget: int, rng: np.random.Generator) -> np.ndarray:
    pool = np.asarray(pool_x, dtype=float)
    centers = np.asarray(ref_x, dtype=float)
    diff = pool[:, None, :] - centers[None, :, :]
    min_dist = np.sqrt(np.sum(diff * diff, axis=2)).min(axis=1)
    available = np.ones(len(pool), dtype=bool)
    chosen = []
    for _ in range(min(budget, len(pool))):
        masked = np.where(available, min_dist, -np.inf)
        peak = masked.max()
        ties = np.flatnonzero(available & (min_dist >= peak - 1e-12))
        j = int(rng.choice(ties))
        chosen.append(j)
        available[j] = False
        step = np.linalg.norm(pool - pool[j], axis=1)
        min_dist = np.minimum(min_dist, step)
    return np.asarray(chosen, dtype=int)


def _badge(ref_x, ref_y, pool_x, pool_y, budget: int, rng: np.random.Generator) -> np.ndarray:
    ref_x = np.asarray(ref_x, dtype=float)
    pool = np.asarray(pool_x, dtype=float)
    model = Ridge(alpha=1.0)
    model.fit(ref_x, np.asarray(ref_y, dtype=float).ravel())
    residual = np.asarray(pool_y, dtype=float).ravel() - model.predict(pool)
    embedding = pool * residual[:, None]
    d2 = np.sum(embedding * embedding, axis=1)
    available = np.ones(len(pool), dtype=bool)
    chosen = []
    weights = np.maximum(d2, 0.0)
    if float(weights.sum()) <= 0:
        return rng.choice(len(pool), size=min(budget, len(pool)), replace=False).astype(int)
    first = int(rng.choice(len(pool), p=weights / weights.sum()))
    chosen.append(first)
    available[first] = False
    closest = np.sum((embedding - embedding[first]) ** 2, axis=1)
    for _ in range(min(budget, len(pool)) - 1):
        masked = np.where(available, closest, 0.0)
        if float(masked.sum()) <= 0:
            rest = np.flatnonzero(available)
            j = int(rng.choice(rest))
        else:
            j = int(rng.choice(len(pool), p=masked / masked.sum()))
        chosen.append(j)
        available[j] = False
        dist = np.sum((embedding - embedding[j]) ** 2, axis=1)
        closest = np.minimum(closest, dist)
    return np.asarray(chosen, dtype=int)


def select_window(
    strategy: str,
    ref_x,
    ref_y,
    pool_x,
    pool_y,
    budget: int,
    rng: np.random.Generator,
    seed: int,
) -> np.ndarray:
    """Return ``budget`` indices into the pool for one query strategy."""
    pool_x = np.asarray(pool_x, dtype=float)
    n_pool = pool_x.shape[0]
    budget = min(int(budget), n_pool)
    if strategy == "recent":
        return np.arange(n_pool - budget, n_pool)
    if strategy == "random":
        return rng.choice(n_pool, size=budget, replace=False).astype(int)
    if strategy == "entropy":
        p = _domain_probability(ref_x, pool_x, seed)
        return _top_k(_binary_entropy(p), budget, rng)
    if strategy == "kcenter":
        return _kcenter(ref_x, pool_x, budget, rng)
    if strategy == "badge":
        return _badge(ref_x, ref_y, pool_x, pool_y, budget, rng)
    if strategy == "learning_loss":
        ref_x = np.asarray(ref_x, dtype=float)
        model = Ridge(alpha=1.0)
        model.fit(ref_x, np.asarray(ref_y, dtype=float).ravel())
        residual = np.abs(np.asarray(pool_y, dtype=float).ravel() - model.predict(pool_x))
        return _top_k(residual, budget, rng)
    raise ValueError(f"unknown strategy: {strategy}")
