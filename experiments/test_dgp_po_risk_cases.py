"""Test cases for the two shift generators.

Each draw uses 1000 rows in each batch. The variance estimator is called as it
stands; these cases do not change it.

Concept drift keeps P(X) fixed and adds a CATE on the second batch.
Covariate shift keeps the regression fixed and moves the first batch.
"""

from __future__ import annotations

import numpy as np

from experiments.po_risk_variance import (
    generate_concept_drift_dgp,
    generate_covariate_shift_dgp,
    variance_estimator_po_risk,
)

N = 2000
P = 20
N_PER_BATCH = 1000


def _batches(treatment) -> tuple[int, int]:
    treatment = np.asarray(treatment).astype(int)
    return int(np.sum(treatment == 0)), int(np.sum(treatment == 1))


def _fit(X, Y, W, cate_model: str):
    return variance_estimator_po_risk(
        X,
        Y,
        W,
        cate_model=cate_model,
        n_estimators=40,
        min_leaf_size=20,
        test_size=0.5,
        seed=0,
        n_jobs=1,
        n_perm=0,
        bias_correction="Wager",
    )


def _assert_delta_method(result) -> None:
    """The returned sample-wise variance is ``4 (phi - tau)^2 Var(tau)``."""
    residual_sq = (result["pseudo"] - result["tau"]) ** 2
    np.testing.assert_allclose(result["var_estimator"], 4.0 * residual_sq * result["var_tau"])
    assert result["var_estimator"].shape == result["tau"].shape
    assert result["var_estimator"].shape[0] == N_PER_BATCH
    assert np.all(np.isfinite(result["var_estimator"]))
    assert np.all(result["var_tau"] >= -1e-12)
    assert result["summary"]["po_risk"] > 0.0


def test_concept_drift_dgp_po_risk_variance():
    """Same covariates, CATE appears on the second batch of 1000 rows."""
    X_null, Y_null, W_null = generate_concept_drift_dgp(
        n=N, p=P, delta_beta=0.0, rho=0.0, seed=0
    )
    X_alt, Y_alt, W_alt = generate_concept_drift_dgp(
        n=N, p=P, delta_beta=1.0, rho=0.0, seed=0
    )
    assert _batches(W_null) == (N_PER_BATCH, N_PER_BATCH)
    assert _batches(W_alt) == (N_PER_BATCH, N_PER_BATCH)
    assert np.allclose(X_null, X_alt)
    assert np.allclose(Y_null[:N_PER_BATCH], Y_alt[:N_PER_BATCH])

    null = _fit(X_null, Y_null, W_null, "RF_IJ")
    alt = _fit(X_alt, Y_alt, W_alt, "RF_IJ")
    delta = _fit(X_alt, Y_alt, W_alt, "RF_delta")
    _assert_delta_method(null)
    _assert_delta_method(alt)
    _assert_delta_method(delta)
    assert null["method"] == "RF_IJ"
    assert alt["method"] == "RF_IJ"
    assert delta["method"] == "RF_delta"
    assert alt["summary"]["po_risk"] > null["summary"]["po_risk"]
    assert alt["summary"]["mean_var_tau"] > null["summary"]["mean_var_tau"]


def test_covariate_shift_dgp_po_risk_variance():
    """Same regression, the first batch of 1000 rows is shifted on columns 0-3."""
    X_null, Y_null, W_null = generate_covariate_shift_dgp(
        n=N, p=P, gamma=0.0, rho=0.0, seed=0
    )
    X_alt, Y_alt, W_alt = generate_covariate_shift_dgp(
        n=N, p=P, gamma=1.0, rho=0.0, seed=0
    )
    assert _batches(W_null) == (N_PER_BATCH, N_PER_BATCH)
    assert _batches(W_alt) == (N_PER_BATCH, N_PER_BATCH)
    beta = np.zeros(P)
    beta[:4] = 1.0
    assert abs(float(np.mean(Y_alt - X_alt @ beta))) < 0.15

    null = _fit(X_null, Y_null, W_null, "RF_IJ")
    alt = _fit(X_alt, Y_alt, W_alt, "RF_IJ")
    delta = _fit(X_alt, Y_alt, W_alt, "RF_delta")
    _assert_delta_method(null)
    _assert_delta_method(alt)
    _assert_delta_method(delta)
    assert null["method"] == "RF_IJ"
    assert alt["method"] == "RF_IJ"
    assert delta["method"] == "RF_delta"
    assert alt["summary"]["po_risk"] > null["summary"]["po_risk"]
    assert alt["summary"]["mean_var_tau"] > null["summary"]["mean_var_tau"]
