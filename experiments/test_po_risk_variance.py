"""Checks for the PO-risk variance written against the two generators."""

from __future__ import annotations

import numpy as np

from experiments.po_risk_variance import (
    IJ_estimator,
    generate_concept_drift_dgp,
    generate_covariate_shift_dgp,
)


def test_each_batch_has_half_the_rows():
    X, Y, W = generate_concept_drift_dgp(n=2000, p=20, delta_beta=1.0, seed=0)
    assert X.shape == (2000, 20)
    assert int((W == 0).sum()) == 1000
    assert int((W == 1).sum()) == 1000
    X2, Y2, W2 = generate_covariate_shift_dgp(n=2000, p=20, gamma=1.0, seed=1)
    assert X2.shape == (2000, 20)
    assert Y2.shape == (2000,)
    assert int((W2 == 0).sum()) == 1000
    assert int((W2 == 1).sum()) == 1000


def test_concept_drift_moves_only_the_second_batch():
    null_x, null_y, _ = generate_concept_drift_dgp(n=400, p=6, delta_beta=0.0, seed=4)
    alt_x, alt_y, _ = generate_concept_drift_dgp(n=400, p=6, delta_beta=1.5, seed=4)
    assert np.allclose(null_x, alt_x)
    assert np.allclose(null_y[:200], alt_y[:200])
    beta = np.zeros(6)
    beta[:4] = np.arange(4) * 1.5
    assert np.allclose(alt_y[200:] - null_y[200:], alt_x[200:] @ beta)


def test_ij_wager_subtracts_tree_variance_over_b():
    outputs = np.array(
        [
            [1.0, 3.0, 5.0],
            [2.0, 2.0, 2.0],
        ]
    )
    inbag = np.array(
        [
            [0.0, 1.0, 2.0],
            [2.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    raw = IJ_estimator(outputs, inbag, bias_correction="none")
    t_centered = outputs - outputs.mean(axis=1, keepdims=True)
    n_centered = inbag - inbag.mean(axis=1, keepdims=True)
    expected = np.sum(((t_centered @ n_centered.T) / 3.0) ** 2, axis=1)
    np.testing.assert_allclose(raw, expected)
    corrected = IJ_estimator(outputs, inbag, bias_correction="Wager")
    naive = np.var(outputs, ddof=1, axis=1) / 3.0
    np.testing.assert_allclose(corrected, np.maximum(expected - naive, 0.0))
