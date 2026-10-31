"""The two DGP cases. Each batch has 1000 rows. The estimator is called as written."""

from __future__ import annotations

import numpy as np

from experiments.po_risk_variance import (
    generate_concept_drift_dgp,
    generate_covariate_shift_dgp,
    make_registry,
    permuCATE_PO,
    variance_estimator_po_risk,
)

N = 2000
P = 20
N_PER_BATCH = 1000
TREES = 30


def _fit_variance(X, Y, W, cate_model):
    registry = make_registry(ntree=TREES, n_jobs=1)
    return variance_estimator_po_risk(
        X, Y, W,
        cate_model=cate_model,
        model_registry=registry,
        n_estimators=TREES,
        seed=0,
        n_jobs=1,
        bias_correction="Wager",
    )


def _fit_risk(X, Y, W):
    _, _, po_risk = permuCATE_PO(
        X, Y, W,
        model_registry=make_registry(ntree=TREES, n_jobs=1),
        seed=0,
    )
    return po_risk


def _assert_variance(var_estimator):
    assert var_estimator.shape == (N,)
    assert np.all(np.isfinite(var_estimator))
    assert np.all(var_estimator >= -1e-8)


def test_concept_drift_dgp_po_risk_variance():
    X0, Y0, W0 = generate_concept_drift_dgp(n=N, p=P, delta_beta=0.0, rho=0.0, seed=0)
    X1, Y1, W1 = generate_concept_drift_dgp(n=N, p=P, delta_beta=1.0, rho=0.0, seed=0)
    assert int((W0 == 0).sum()) == N_PER_BATCH and int((W0 == 1).sum()) == N_PER_BATCH
    assert int((W1 == 0).sum()) == N_PER_BATCH and int((W1 == 1).sum()) == N_PER_BATCH
    assert np.allclose(X0, X1)

    var_null = _fit_variance(X0, Y0, W0, "RF_IJ")
    var_alt = _fit_variance(X1, Y1, W1, "RF_IJ")
    var_delta = _fit_variance(X1, Y1, W1, "RF_delta")
    _assert_variance(var_null)
    _assert_variance(var_alt)
    _assert_variance(var_delta)
    assert np.mean(var_alt) > np.mean(var_null)
    assert _fit_risk(X1, Y1, W1) > _fit_risk(X0, Y0, W0)


def test_covariate_shift_dgp_po_risk_variance():
    X0, Y0, W0 = generate_covariate_shift_dgp(n=N, p=P, gamma=0.0, rho=0.0, seed=0)
    X1, Y1, W1 = generate_covariate_shift_dgp(n=N, p=P, gamma=1.0, rho=0.0, seed=0)
    assert int((W0 == 0).sum()) == N_PER_BATCH and int((W0 == 1).sum()) == N_PER_BATCH
    assert int((W1 == 0).sum()) == N_PER_BATCH and int((W1 == 1).sum()) == N_PER_BATCH
    beta = np.zeros(P)
    beta[:4] = 1.0
    assert abs(float(np.mean(Y1 - X1 @ beta))) < 0.15

    var_null = _fit_variance(X0, Y0, W0, "RF_IJ")
    var_alt = _fit_variance(X1, Y1, W1, "RF_IJ")
    var_delta = _fit_variance(X1, Y1, W1, "RF_delta")
    _assert_variance(var_null)
    _assert_variance(var_alt)
    _assert_variance(var_delta)
    assert np.mean(var_alt) > np.mean(var_null)
    assert _fit_risk(X1, Y1, W1) > _fit_risk(X0, Y0, W0)
