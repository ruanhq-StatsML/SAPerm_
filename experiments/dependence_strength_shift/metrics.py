"""Ranking-recovery scores for signed dependence strength shift."""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, spearmanr


def _finite(a, b):
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask], mask


def spearman(a, b) -> float:
    x, y, _ = _finite(a, b)
    if x.size < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(spearmanr(x, y).statistic)


def kendall(a, b) -> float:
    x, y, _ = _finite(a, b)
    if x.size < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(kendalltau(x, y).statistic)


def precision_at_k(oracle, score, k: int) -> float:
    """Fraction of the top-k |oracle| features recovered in top-k |score|."""
    oracle = np.asarray(oracle, dtype=float)
    score = np.asarray(score, dtype=float)
    k = int(min(k, oracle.size))
    if k <= 0:
        return float("nan")
    true_top = set(np.argsort(-np.abs(oracle))[:k].tolist())
    pred_top = set(np.argsort(-np.abs(score))[:k].tolist())
    return float(len(true_top & pred_top) / k)


def sign_agreement(oracle, score, mask=None) -> float:
    """Share of coordinates with matching sign (zeros ignored)."""
    oracle = np.asarray(oracle, dtype=float)
    score = np.asarray(score, dtype=float)
    if mask is None:
        mask = np.ones(oracle.size, dtype=bool)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(oracle) & np.isfinite(score)
    mask &= oracle != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.sign(oracle[mask]) == np.sign(score[mask])))


def ranking_recovery(oracle_shift, estimated_shift, k: int = 3, active_mask=None) -> dict:
    """Spearman / Kendall / Precision@k / sign agreement against the oracle."""
    return {
        "spearman": spearman(oracle_shift, estimated_shift),
        "kendall": kendall(oracle_shift, estimated_shift),
        "spearman_abs": spearman(np.abs(oracle_shift), np.abs(estimated_shift)),
        "precision_at_k": precision_at_k(oracle_shift, estimated_shift, k=k),
        "sign_agreement": sign_agreement(oracle_shift, estimated_shift, mask=active_mask),
    }
