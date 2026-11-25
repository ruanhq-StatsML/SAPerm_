"""Y does not use X under either null."""

from __future__ import annotations

import numpy as np

from experiments.null_noise_auc.dgp import DGPS
from experiments.null_noise_auc.methods import selection_auc


def _max_corr(x, y) -> float:
    y = y - y.mean()
    num = (x - x.mean(axis=0)).T @ y
    den = x.std(axis=0) * y.std() * len(y)
    return float(np.max(np.abs(num / den)))


def test_outcome_does_not_use_features():
    for name, draw in DGPS.items():
        x0, y0, x1, y1 = draw(4000, 8, np.random.default_rng(0))
        assert x0.shape == (4000, 8), name
        assert _max_corr(x0, y0) < 0.08
        assert _max_corr(x1, y1) < 0.08
        assert abs(float(y1.mean() - y0.mean())) > 0.15


def test_selection_auc_coin_flip_and_perfect():
    support = np.array([0, 1])
    assert selection_auc(np.zeros(6), support) == 0.5
    assert selection_auc(np.array([2.0, 1.0, 0.0, 0.0, 0.0, 0.0]), support) == 1.0
