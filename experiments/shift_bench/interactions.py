"""High-order interaction / shift quantities + PO-risk Shapley.

One module for:
  - PO-risk (+ DRPerm)
  - R-risk / power-loss (RFPerm)
  - RBF-MMD²
  - τ-Shapley / ood_vimp (three-way)
  - window_bundle → one-row dict for the benchmark DataFrame

Heavy Shapley/online helpers live in ``experiments.online_po_risk.score``;
this file is the testing surface.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from experiments.shift_bench.utils import (
    as_1d,
    batch_xy,
    brier_losses,
    make_folds,
    rf_classifier,
    rf_regressor,
)

# Re-export the PO-Shapley three-way from the online score module.
from experiments.online_po_risk.score import (  # noqa: F401
    MMD_LOCO,
    RF_domain_classifier,
    fit_dr_tau,
    ood_vimp,
    ood_vimp_wrapper,
    tau_shapley_importance,
)


def r_risk(X, Y, W, *, seed: int = 0, n_estimators: int = 60) -> float:
    Y, W = as_1d(Y), as_1d(W).astype(int)
    Xw = np.column_stack([X, W])
    m = rf_regressor(seed, n_estimators=n_estimators)
    m.fit(Xw, Y)
    x0 = np.column_stack([X, np.zeros(len(X))])
    x1 = np.column_stack([X, np.ones(len(X))])
    return float(np.mean((m.predict(x1) - m.predict(x0)) ** 2))


def power_loss(X_exist, y_exist, X_new, y_new, *, seed: int = 0, n_estimators: int = 80) -> float:
    y_exist, y_new = as_1d(y_exist), as_1d(y_new)
    clf = rf_classifier(seed, n_estimators=n_estimators, oob_score=True)
    clf.fit(X_exist, y_exist.astype(int))
    oob = clf.oob_decision_function_
    classes = list(clf.classes_)
    if oob.ndim == 2:
        y_hat_oob = oob[:, classes.index(1)] if 1 in classes else 1.0 - oob[:, 0]
    else:
        y_hat_oob = oob
    proba = clf.predict_proba(X_new)
    y_hat_new = proba[:, classes.index(1)] if 1 in classes else 1.0 - proba[:, 0]
    return float(np.mean(brier_losses(y_new, y_hat_new)) - np.mean(brier_losses(y_exist, y_hat_oob)))


def rbf_mmd2(X, Y, *, max_median: int = 400, seed: int = 0) -> float:
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    rng = np.random.default_rng(seed)
    if len(X) + len(Y) > max_median:
        ix = rng.choice(len(X), min(len(X), max_median // 2), replace=False)
        iy = rng.choice(len(Y), min(len(Y), max_median // 2), replace=False)
        Z = np.vstack([X[ix], Y[iy]])
    else:
        Z = np.vstack([X, Y])
    sq = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=-1)
    pos = sq[sq > 0]
    gamma = 1.0 / float(np.median(pos)) if pos.size else 1.0

    def kmean(A, B):
        d = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
        return float(np.mean(np.exp(-gamma * d)))

    return float(kmean(X, X) + kmean(Y, Y) - 2.0 * kmean(X, Y))


def drperm_pvalue(X, Y, W, *, n_perm: int = 40, seed: int = 0, clip_e: float = 0.05, n_estimators: int = 60) -> dict:
    X = np.asarray(X, float)
    Y, W = as_1d(Y), as_1d(W).astype(int)
    n = X.shape[0]
    folds = make_folds(n, 2, seed)
    mu_hat = np.zeros(n)
    e_hat = np.zeros(n)
    for k, te in enumerate(folds):
        tr = np.setdiff1d(np.arange(n), te)
        m = rf_regressor(seed + k, n_estimators=n_estimators)
        m.fit(X[tr], Y[tr])
        mu_hat[te] = m.predict(X[te])
        if len(np.unique(W[tr])) < 2:
            e_hat[te] = float(np.mean(W[tr]))
        else:
            e = rf_classifier(seed + 100 + k, n_estimators=max(20, n_estimators // 2))
            e.fit(X[tr], W[tr])
            classes = list(e.classes_)
            proba = e.predict_proba(X[te])
            e_hat[te] = proba[:, classes.index(1)] if 1 in classes else 1.0 - proba[:, 0]
    e_hat = np.clip(e_hat, clip_e, 1 - clip_e)
    residual = Y - mu_hat

    def _po(Wb, e_b, seed_tau):
        po_b = residual * (Wb - e_b)
        tau = rf_regressor(seed_tau, n_estimators=n_estimators)
        tau.fit(X, po_b)
        return float(np.mean(tau.predict(X) ** 2))

    observed = _po(W, e_hat, seed + 200)
    if n_perm <= 0:
        return {"method": "DRPerm", "statistic": observed, "p_value": float("nan"), "po_risk": observed, "reject": False}
    rng = np.random.default_rng(seed + 7)
    null = np.zeros(n_perm)
    for b in range(n_perm):
        Wb = rng.permutation(W)
        # reuse e_hat shape under perm of W (cheap null)
        null[b] = _po(Wb, e_hat, seed + 400 + b)
    p = (1.0 + np.sum(null >= observed - 1e-15)) / (1 + n_perm)
    return {"method": "DRPerm", "statistic": observed, "p_value": float(p), "po_risk": observed, "reject": p < 0.05}


def rfperm_pvalue(X_exist, y_exist, X_new, y_new, *, n_perm: int = 40, seed: int = 0, n_estimators: int = 80) -> dict:
    obs = power_loss(X_exist, y_exist, X_new, y_new, seed=seed, n_estimators=n_estimators)
    y_exist, y_new = as_1d(y_exist), as_1d(y_new)
    clf = rf_classifier(seed, n_estimators=n_estimators, oob_score=True)
    clf.fit(X_exist, y_exist.astype(int))
    oob = clf.oob_decision_function_
    classes = list(clf.classes_)
    if oob.ndim == 2:
        y_hat_oob = oob[:, classes.index(1)] if 1 in classes else 1.0 - oob[:, 0]
    else:
        y_hat_oob = oob
    proba = clf.predict_proba(X_new)
    y_hat_new = proba[:, classes.index(1)] if 1 in classes else 1.0 - proba[:, 0]
    loss_exist = brier_losses(y_exist, y_hat_oob)
    loss_new = brier_losses(y_new, y_hat_new)
    rng = np.random.default_rng(seed + 11)
    pooled = np.concatenate([loss_exist, loss_new])
    n_e = len(loss_exist)
    exceed = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        dif = float(np.mean(perm[n_e:]) - np.mean(perm[:n_e]))
        if dif >= obs - 1e-15:
            exceed += 1
    p = (1 + exceed) / (1 + n_perm)
    return {"method": "RFPerm", "statistic": float(obs), "power_loss": float(obs), "p_value": float(p), "reject": p < 0.05}


def mmd_pvalue(A, B, *, n_perm: int = 40, seed: int = 0) -> dict:
    A, B = np.asarray(A, float), np.asarray(B, float)
    obs = rbf_mmd2(A, B, seed=seed)
    pooled = np.vstack([A, B])
    n0 = len(A)
    rng = np.random.default_rng(seed + 19)
    exceed = 0
    for i in range(n_perm):
        perm = rng.permutation(len(pooled))
        m = rbf_mmd2(pooled[perm[:n0]], pooled[perm[n0:]], seed=seed + 100 + i)
        if m >= obs - 1e-15:
            exceed += 1
    p = (1 + exceed) / (1 + n_perm)
    return {"method": "MMD", "statistic": float(obs), "p_value": float(p), "reject": p < 0.05}


def window_bundle(
    X0,
    y0,
    X1,
    y1,
    *,
    names: list[str] | None = None,
    drivers: list[int] | None = None,
    n_perm: int = 40,
    seed: int = 0,
    n_estimators: int = 40,
    with_shapley: bool = True,
    tau_kind: str = "rf",
) -> dict:
    """One window → RF / DR(PO) / MMD / R-risk / optional PO-Shapley three-way."""
    X, Y, W = batch_xy(X0, y0, X1, y1)
    rf = rfperm_pvalue(X0, y0, X1, y1, n_perm=n_perm, seed=seed, n_estimators=n_estimators)
    dr = drperm_pvalue(X, Y, W, n_perm=n_perm, seed=seed + 3, n_estimators=n_estimators)
    mmd = mmd_pvalue(X0, X1, n_perm=n_perm, seed=seed + 5)
    out: dict = {
        "RFPerm_p": float(rf["p_value"]),
        "RFPerm_reject": bool(rf["reject"]),
        "DRPerm_p": float(dr["p_value"]),
        "DRPerm_reject": bool(dr["reject"]),
        "MMD_p": float(mmd["p_value"]),
        "MMD_reject": bool(mmd["reject"]),
        "po_risk": float(dr["po_risk"]),
        "power_loss": float(rf["power_loss"]),
        "r_risk": float(r_risk(X, Y, W, seed=seed + 7, n_estimators=n_estimators)),
        "mmd2": float(mmd["statistic"]),
    }
    if with_shapley:
        vim = ood_vimp(X0, y0, X1, y1, n_perm=max(8, n_perm // 3), seed=seed + 11, tau_kind=tau_kind)
        shap = np.asarray(vim["po_shapley"], float)
        rf_v = np.asarray(vim["rf_vimp"], float)
        mmd_v = np.asarray(vim["MMD_vimp"], float)
        names = list(names) if names is not None else [f"x{j}" for j in range(len(shap))]
        out["tau2"] = float(vim["tau2"])
        out["top_shap"] = names[int(np.nanargmax(shap))] if np.any(np.isfinite(shap)) else ""
        out["top_rf"] = names[int(np.nanargmax(rf_v))] if np.any(np.isfinite(rf_v)) else ""
        out["top_mmd"] = names[int(np.nanargmax(mmd_v))] if np.any(np.isfinite(mmd_v)) else ""
        if drivers:
            dset = set(drivers)
            out["top_shap_is_driver"] = int(int(np.nanargmax(shap)) in dset) if np.any(np.isfinite(shap)) else float("nan")
            out["top_rf_is_driver"] = int(int(np.nanargmax(rf_v)) in dset) if np.any(np.isfinite(rf_v)) else float("nan")
            out["top_mmd_is_driver"] = int(int(np.nanargmax(mmd_v)) in dset) if np.any(np.isfinite(mmd_v)) else float("nan")
            out["driver_shap"] = float(np.mean(shap[list(dset)]))
            noise = [i for i in range(len(shap)) if i not in dset]
            out["noise_shap"] = float(np.mean(shap[noise])) if noise else float("nan")
    return out
