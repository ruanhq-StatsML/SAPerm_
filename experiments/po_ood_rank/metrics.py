"""Rank agreement between a score and the planted OOD magnitude."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def spearman(score: np.ndarray, target: np.ndarray) -> float:
    """Spearman correlation. Constant inputs have no rank to recover."""
    from scipy.stats import spearmanr

    left = np.asarray(score, dtype=float).reshape(-1)
    right = np.asarray(target, dtype=float).reshape(-1)
    mask = np.isfinite(left) & np.isfinite(right)
    left, right = left[mask], right[mask]
    if left.size < 5 or np.nanstd(left) < 1e-12 or np.nanstd(right) < 1e-12:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def _binary_rank_scores(score: np.ndarray, high: np.ndarray) -> tuple[float, float]:
    score = np.asarray(score, dtype=float).reshape(-1)
    high = np.asarray(high, dtype=int).reshape(-1)
    mask = np.isfinite(score)
    score, high = score[mask], high[mask]
    if high.size < 5 or len(np.unique(high)) < 2 or np.nanstd(score) < 1e-12:
        return float("nan"), float("nan")
    return float(roc_auc_score(high, score)), float(average_precision_score(high, score))


def high_ood_mask(target: np.ndarray) -> np.ndarray:
    """Upper half of the planted magnitude, including the median tie."""
    target = np.asarray(target, dtype=float).reshape(-1)
    return (target >= np.median(target)).astype(int)


def powered_score(risk: np.ndarray, var_po: np.ndarray, mag_power: float) -> np.ndarray:
    """``r ** mag_power * Var(r)``. Magnitude 0 ranks by the variance alone."""
    risk = np.asarray(risk, dtype=float)
    weighted = np.power(np.maximum(risk, 0.0), float(mag_power))
    return weighted * np.asarray(var_po, dtype=float)


def rank_row(risk: np.ndarray, var_po: np.ndarray, var_tau: np.ndarray, designed: np.ndarray, realized: np.ndarray, mag_power: float) -> dict:
    """Spearman and high-OOD detection for one magnitude power."""
    score = powered_score(risk, var_po, mag_power)
    designed_high = high_ood_mask(designed)
    realized_high = high_ood_mask(realized)
    auroc_index, ap_index = _binary_rank_scores(score, designed_high)
    auroc_realized, ap_realized = _binary_rank_scores(score, realized_high)
    auroc_risk, _ = _binary_rank_scores(risk, designed_high)
    return {
        "spearman_index": spearman(score, designed),
        "spearman_realized": spearman(score, realized),
        "spearman_risk_index": spearman(risk, designed),
        "spearman_risk_realized": spearman(risk, realized),
        "spearman_var_index": spearman(var_po, designed),
        "spearman_vartau_index": spearman(var_tau, designed),
        "auroc_index": auroc_index,
        "ap_index": ap_index,
        "auroc_realized": auroc_realized,
        "ap_realized": ap_realized,
        "auroc_risk_index": auroc_risk,
    }
