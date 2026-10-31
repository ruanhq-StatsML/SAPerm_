"""Checks for the PO-risk variance estimators and the two shift generators."""

from __future__ import annotations

import numpy as np

from experiments.po_risk_variance import (
    IJ_estimator,
    delta_method_po_variance,
    generate_concept_drift_dgp,
    generate_covariate_shift_dgp,
    ij_components,
    planted_features,
    true_cate,
    variance_estimator_po_risk,
)


def test_each_batch_has_half_the_rows():
    X, Y, W = generate_concept_drift_dgp(n=2000, p=20, delta_beta=1.0, seed=0)
    assert X.shape == (2000, 20)
    assert Y.shape == (2000,)
    assert int((W == 0).sum()) == 1000
    assert int((W == 1).sum()) == 1000
    X2, Y2, W2 = generate_covariate_shift_dgp(n=2000, p=20, gamma=1.0, seed=1)
    assert X2.shape == (2000, 20)
    assert int((W2 == 0).sum()) == 1000
    assert int((W2 == 1).sum()) == 1000
    assert Y2.shape == (2000,)


def test_concept_drift_moves_only_the_second_batch():
    null_x, null_y, _ = generate_concept_drift_dgp(n=400, p=6, delta_beta=0.0, seed=4)
    alt_x, alt_y, _ = generate_concept_drift_dgp(n=400, p=6, delta_beta=1.5, seed=4)
    assert np.allclose(null_x, alt_x)
    assert np.allclose(null_y[:200], alt_y[:200])
    beta = np.zeros(6)
    beta[:4] = np.arange(4) * 1.5
    assert np.allclose(alt_y[200:] - null_y[200:], alt_x[200:] @ beta)


def test_covariate_shift_keeps_the_regression_and_moves_the_first_batch():
    base_x, base_y, _ = generate_covariate_shift_dgp(n=400, p=6, gamma=0.0, rho=0.0, seed=5)
    # gamma = 0 draws the same X and then adds pure noise, so redraw the noise
    # path by calling the generator and checking the conditional mean directly.
    X, Y, W = generate_covariate_shift_dgp(n=400, p=6, gamma=0.8, rho=0.0, seed=6)
    assert np.std(X[:200, 0]) > np.std(X[200:, 0])
    beta = np.zeros(6)
    beta[:4] = 0.8
    residual = Y - X @ beta
    assert abs(float(np.mean(residual))) < 0.2
    assert int((W == 0).sum()) == 200
    assert base_x.shape == (400, 6)
    assert base_y.shape == (400,)


def test_population_cate_matches_the_generators():
    X = np.ones((5, 6))
    cate = true_cate("concept", X, strength=1.0)
    assert np.allclose(cate, np.full(5, 0.0 + 1.0 + 2.0 + 3.0))
    assert planted_features("concept").tolist() == [1, 2, 3]
    assert np.allclose(true_cate("covariate", X, strength=2.0), 0.0)
    assert planted_features("covariate").tolist() == [0, 1, 2, 3]


def test_ij_centers_trees_and_inbag_counts():
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
    covariance = (t_centered @ n_centered.T) / 3.0
    expected = np.sum(covariance ** 2, axis=1)
    np.testing.assert_allclose(raw, expected)
    pieces = ij_components(outputs, inbag, bias_correction="Wager")
    n_var = float(np.mean(np.mean(inbag ** 2, axis=1) - np.mean(inbag, axis=1) ** 2))
    tree_var = np.sum(t_centered ** 2, axis=1) / 3.0
    corrected = np.maximum(expected - inbag.shape[0] * n_var * tree_var / 3.0, 0.0)
    np.testing.assert_allclose(pieces["variance"], corrected)
    assert np.all(pieces["variance"] <= raw + 1e-12)


def test_delta_method_is_four_times_residual_times_variance():
    pseudo = np.array([1.0, -1.0, 0.5])
    tau = np.array([0.0, 0.0, 0.5])
    var_tau = np.array([0.25, 0.0, 4.0])
    residual_sq, variance = delta_method_po_variance(pseudo, tau, var_tau)
    np.testing.assert_allclose(residual_sq, (pseudo - tau) ** 2)
    np.testing.assert_allclose(variance, 4.0 * residual_sq * var_tau)


def test_po_risk_variance_runs_on_a_small_concept_draw():
    X, Y, W = generate_concept_drift_dgp(n=180, p=6, delta_beta=1.0, rho=0.0, seed=0)
    result = variance_estimator_po_risk(
        X,
        Y,
        W,
        cate_model="RF_IJ",
        n_estimators=12,
        min_leaf_size=6,
        test_size=0.5,
        seed=0,
        n_jobs=1,
        n_perm=0,
    )
    assert result["var_estimator"].shape == result["tau"].shape
    assert np.all(np.isfinite(result["var_estimator"]))
    assert np.all(result["var_tau"] >= -1e-12)
    assert result["summary"]["po_risk"] > 0.0
    assert result["method"] == "RF_IJ"
