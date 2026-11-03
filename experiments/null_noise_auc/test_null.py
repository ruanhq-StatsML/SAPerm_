"""Checks that the null design really leaves Y independent of X."""

from __future__ import annotations

import numpy as np

from experiments.null_noise_auc.dgp import decoy_support, draw_null_pair
from experiments.null_noise_auc.methods import selection_auc


def _max_corr(x: np.ndarray, y: np.ndarray) -> float:
    y = y - y.mean()
    y_sd = y.std()
    num = (x - x.mean(axis=0)).T @ y
    den = x.std(axis=0) * y_sd * len(y)
    return float(np.max(np.abs(num / den)))


def test_outcome_does_not_use_features():
    rng = np.random.default_rng(0)
    x0, y0, x1, y1 = draw_null_pair(5000, 12, rng)
    assert x0.shape == (5000, 12)
    assert x1.shape == x0.shape
    assert _max_corr(x0, y0) < 0.08
    assert _max_corr(x1, y1) < 0.08
    assert abs(float(y1.mean() - y0.mean())) > 0.15


def test_batches_share_the_covariate_center():
    rng = np.random.default_rng(1)
    x0, _, x1, _ = draw_null_pair(4000, 10, rng)
    gap = np.max(np.abs(x0.mean(axis=0) - x1.mean(axis=0)))
    assert gap < 0.25


def test_selection_auc_coin_flip_and_perfect():
    support = np.array([0, 1])
    assert selection_auc(np.zeros(6), support) == 0.5
    perfect = np.array([2.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    assert selection_auc(perfect, support) == 1.0


def test_decoy_is_a_proper_subset():
    rng = np.random.default_rng(2)
    support = decoy_support(15, 4, rng)
    assert support.shape == (4,)
    assert len(set(support.tolist())) == 4
