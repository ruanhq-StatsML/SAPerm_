"""Rank blocks, then features, by distance to the conditional counterfactual."""

from __future__ import annotations

import numpy as np

from .metrics import METRIC_FN


def rank_blocks(X_real, X_cf, blocks, metric="mean_diff"):
    fn = METRIC_FN[metric]
    scores = {b: fn(X_real, X_cf, cols) for b, cols in blocks.items()}
    ranked = dict(sorted(scores.items(), key=lambda x: -x[1] if np.isfinite(x[1]) else 0.0))
    total = sum(v for v in scores.values() if np.isfinite(v)) + 1e-8
    prop = {b: (v / total if np.isfinite(v) else 0.0) for b, v in ranked.items()}
    return ranked, prop


def rank_features(X_real, X_cf, col_names):
    delta = np.abs(X_real - X_cf)
    vimp = delta.mean(axis=0)
    ranked = dict(sorted(
        zip(col_names, vimp.tolist()), key=lambda x: -x[1]
    ))
    return ranked
