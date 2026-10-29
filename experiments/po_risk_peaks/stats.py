"""Second-order summaries of a PO-risk proxy.

The proxy on a classifier is the true-class residual ``1 - p(y|x)``.
Its mean moves under any shift that hurts calibration or accuracy.
A peak is a different claim: a few environments, or a few examples, carry
a much larger residual than the center of the batch.

Averaging half the corpus, thirty times, does not see that. Those
sub-batch means share one expectation, and the gap between their max and
their median is Monte Carlo error of order ``sigma / sqrt(n/2)``.
The gap has to be computed on small sub-batches, or on the example
residuals themselves (CVaR minus the mean).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis


def true_class_residual(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """``1 - p(y|x)`` for each row. ``probs`` is (n, n_classes)."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if probs.ndim != 2:
        raise ValueError("probs must have shape (n, n_classes)")
    if len(labels) != len(probs):
        raise ValueError("labels and probs must have the same length")
    return 1.0 - probs[np.arange(len(labels)), labels]


def _cvar(values: np.ndarray, level: float = 95.0) -> float:
    if len(values) == 0:
        return float("nan")
    cut = np.percentile(values, level)
    tail = values[values >= cut]
    if len(tail) == 0:
        return float("nan")
    return float(tail.mean())


def _excess_kurtosis(values: np.ndarray) -> float:
    if len(values) < 4 or float(np.std(values)) < 1e-12:
        return 0.0
    return float(kurtosis(values, fisher=True, bias=False))


def subbatch_means(risks: np.ndarray, sub_size: int, n_sub: int, seed: int) -> np.ndarray:
    risks = np.asarray(risks, dtype=float)
    n = len(risks)
    if n == 0:
        return np.zeros(0, dtype=float)
    size = min(int(sub_size), n)
    rng = np.random.default_rng(seed)
    out = np.empty(n_sub, dtype=float)
    for i in range(n_sub):
        idx = rng.choice(n, size=size, replace=False)
        out[i] = float(risks[idx].mean())
    return out


def summarize_risks(
    risks: np.ndarray,
    sub_size: int = 32,
    n_sub: int = 40,
    seed: int = 0,
) -> dict[str, float | np.ndarray]:
    """Mean, tail excess, and peak gap at two sub-batch scales.

    ``peak_gap`` is max minus median of the example residuals.
    ``sub_peak_gap`` is the same functional on means of ``sub_size`` examples.
    ``half_peak_gap`` uses sub-batches of size ``n/2`` and should collapse
    toward zero whenever the residual distribution has a finite variance.
    """
    risks = np.asarray(risks, dtype=float)
    mean = float(risks.mean()) if len(risks) else float("nan")
    cvar = _cvar(risks)
    small = subbatch_means(risks, sub_size, n_sub, seed)
    half = subbatch_means(risks, max(1, len(risks) // 2), n_sub, seed + 1)
    return {
        "n": float(len(risks)),
        "mean": mean,
        "cvar95": cvar,
        "tail_excess": cvar - mean,
        "kurtosis": _excess_kurtosis(risks),
        "peak_gap": float(risks.max() - np.median(risks)) if len(risks) else float("nan"),
        "sub_peak_gap": float(small.max() - np.median(small)) if len(small) else float("nan"),
        "half_peak_gap": float(half.max() - np.median(half)) if len(half) else float("nan"),
        "sub_kurtosis": _excess_kurtosis(small),
        "sub_means": small,
    }


def fpr_threshold(human_scores: np.ndarray, alpha: float = 0.05) -> float:
    """Split-conformal threshold for ``predict AI if p(AI) > t``.

    ``human_scores`` are ``p(AI)`` on calibration texts that are human.
    The index ``k = ceil((n+1)(1-alpha))`` makes the calibration false
    positive rate at most alpha when scores are compared with a strict
    inequality.
    """
    scores = np.sort(np.asarray(human_scores, dtype=float))
    n = len(scores)
    if n == 0:
        raise ValueError("calibration set is empty")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(scores[k - 1])


def classification_rates(human_scores: np.ndarray, ai_scores: np.ndarray, threshold: float) -> dict[str, float]:
    human_scores = np.asarray(human_scores, dtype=float)
    ai_scores = np.asarray(ai_scores, dtype=float)
    return {
        "fpr": float(np.mean(human_scores > threshold)) if len(human_scores) else float("nan"),
        "recall": float(np.mean(ai_scores > threshold)) if len(ai_scores) else float("nan"),
        "threshold": float(threshold),
    }
