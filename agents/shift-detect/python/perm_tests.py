#!/usr/bin/env python3
"""Portable RFPerm / DRPerm / CFPerm (RF outcome) for meta-feature power study.

Faithful to the SAPerm / RFPerm / CFPerm methodology, trimmed for a single
script: RandomForest nuisances, permutation p-values, no external registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def _as_1d(a) -> np.ndarray:
    return np.asarray(a, dtype=float).reshape(-1)


def brier_losses(y_true, y_prob) -> np.ndarray:
    y = _as_1d(y_true)
    p = np.clip(_as_1d(y_prob), 1e-4, 1 - 1e-4)
    return (p - y) ** 2


def make_folds(n: int, n_folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return [part for part in np.array_split(idx, n_folds) if len(part)]


def rf_regressor(seed: int, n_estimators: int = 80, max_depth: int = 8):
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=1,
        random_state=seed,
    )


def rf_classifier(
    seed: int,
    n_estimators: int = 80,
    max_depth: int = 8,
    *,
    oob_score: bool = False,
):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=1,
        random_state=seed,
        oob_score=oob_score,
        bootstrap=True,
    )


# ---------------------------------------------------------------------------
# RFPerm
# ---------------------------------------------------------------------------

def rfperm_pvalue(
    X_exist: np.ndarray,
    y_exist: np.ndarray,
    X_new: np.ndarray,
    y_new: np.ndarray,
    *,
    n_perm: int = 120,
    seed: int = 0,
    n_estimators: int = 80,
) -> dict:
    """One RFPerm p-value: OOB losses on exist vs predictive losses on new."""
    y_exist = _as_1d(y_exist)
    y_new = _as_1d(y_new)
    clf = rf_classifier(seed, n_estimators=n_estimators, oob_score=True)
    clf.fit(X_exist, y_exist.astype(int))
    # OOB probabilities for class 1
    if getattr(clf, "oob_decision_function_", None) is None:
        raise RuntimeError("RFPerm needs OOB; increase n_estimators / ensure bootstrap")
    oob = clf.oob_decision_function_
    if oob.ndim == 2:
        classes = list(clf.classes_)
        if 1 in classes:
            y_hat_oob = oob[:, classes.index(1)]
        else:
            y_hat_oob = 1.0 - oob[:, 0]
    else:
        y_hat_oob = oob
    y_hat_new = clf.predict_proba(X_new)[:, list(clf.classes_).index(1) if 1 in list(clf.classes_) else 0]
    if 1 not in list(clf.classes_):
        y_hat_new = 1.0 - clf.predict_proba(X_new)[:, 0]

    loss_exist = brier_losses(y_exist, y_hat_oob)
    loss_new = brier_losses(y_new, y_hat_new)
    obs = float(np.mean(loss_new) - np.mean(loss_exist))

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
    return {
        "method": "RFPerm",
        "statistic": float(obs),
        "p_value": float(p),
        "reject": bool(p < 0.05),
    }


# ---------------------------------------------------------------------------
# DRPerm (PO-risk)
# ---------------------------------------------------------------------------

def drperm_pvalue(
    X: np.ndarray,
    Y: np.ndarray,
    W: np.ndarray,
    *,
    n_perm: int = 120,
    n_splits: int = 2,
    seed: int = 0,
    clip_e: float = 0.05,
    n_estimators: int = 60,
) -> dict:
    """
    Doubly-robust style PO-risk permutation test.

    Pseudo-outcome: (Y - μ(X)) * (W - e(X)).
    Statistic: mean τ(X)^2 where τ regresses the pseudo-outcome on X.
    Null: permute W (batch indicator), refit e and τ.

    Interpretation
    --------------
    Under no concept/batch effect, E[(Y-μ)(W-e) | X] ≈ 0, so PO-risk ≈ 0.
    Large PO-risk means residual outcome variation still aligns with batch
    after adjusting for X — i.e. P(Y|X) differs across batches (concept),
    or unmodeled shift remains in the outcome channel.
    """
    X = np.asarray(X, dtype=float)
    Y = _as_1d(Y)
    W = _as_1d(W).astype(int)
    n = X.shape[0]
    folds = make_folds(n, n_splits, seed)

    mu_hat = np.zeros(n)
    e_hat = np.zeros(n)
    for k, te in enumerate(folds):
        tr = np.setdiff1d(np.arange(n), te)
        m = rf_regressor(seed + k, n_estimators=n_estimators)
        m.fit(X[tr], Y[tr])
        mu_hat[te] = m.predict(X[te])
        # if a fold misses a class, fall back to global rate
        if len(np.unique(W[tr])) < 2:
            e_hat[te] = float(np.mean(W[tr]))
        else:
            e = rf_classifier(seed + 100 + k, n_estimators=max(20, n_estimators // 2))
            e.fit(X[tr], W[tr])
            classes = list(e.classes_)
            proba = e.predict_proba(X[te])
            e_hat[te] = proba[:, classes.index(1)] if 1 in classes else 1.0 - proba[:, 0]
    e_hat = np.clip(e_hat, clip_e, 1 - clip_e)
    residual_y = Y - mu_hat

    # Propensity can be lighter than the outcome RF; keep identical for
    # observed and null so Type-I stays calibrated.
    n_est_e = max(20, n_estimators // 2)

    def _crossfit_e(Wb: np.ndarray, seed_e: int) -> np.ndarray:
        out = np.zeros(n)
        for k, te in enumerate(folds):
            tr = np.setdiff1d(np.arange(n), te)
            if len(np.unique(Wb[tr])) < 2:
                out[te] = float(np.mean(Wb[tr]))
                continue
            e = rf_classifier(seed_e + k, n_estimators=n_est_e)
            e.fit(X[tr], Wb[tr])
            classes = list(e.classes_)
            proba = e.predict_proba(X[te])
            out[te] = proba[:, classes.index(1)] if 1 in classes else 1.0 - proba[:, 0]
        return np.clip(out, clip_e, 1.0 - clip_e)

    def _po_risk(Wb: np.ndarray, e_hat_b: np.ndarray, seed_tau: int) -> float:
        po_b = residual_y * (Wb - e_hat_b)
        tau_b = rf_regressor(seed_tau, n_estimators=n_estimators)
        tau_b.fit(X, po_b)
        return float(np.mean(tau_b.predict(X) ** 2))

    observed = _po_risk(W, e_hat, seed + 200)

    rng = np.random.default_rng(seed + 7)
    null = np.zeros(n_perm)
    for b in range(n_perm):
        Wb = rng.permutation(W)
        # Cross-fit e under the permuted batch label — same procedure as observed.
        e_hat_b = _crossfit_e(Wb, seed_e=seed + 300 + 17 * b)
        null[b] = _po_risk(Wb, e_hat_b, seed_tau=seed + 400 + b)

    p = (1.0 + np.sum(null >= observed - 1e-15)) / (1 + n_perm)
    return {
        "method": "DRPerm",
        "statistic": observed,
        "p_value": float(p),
        "reject": p < 0.05,
        "po_risk": observed,
        "mu_hat_mean": float(np.mean(mu_hat)),
        "e_hat_mean": float(np.mean(e_hat)),
        "interpretation": (
            "PO-risk = E[τ(X)^2] with τ≈(Y-μ)(W-e). "
            "Large values ⇒ batch still predicts residual outcome after X-adjustment "
            "(concept / label-channel shift), not merely covariate shift in X."
        ),
    }


# ---------------------------------------------------------------------------
# CFPerm (LOCO on a feature subset — meta block vs matched embed dims)
# ---------------------------------------------------------------------------

def _r_risk_score(X, Y, W, seed: int, n_estimators: int = 60) -> float:
    """Scalar R-risk style score: E[(μ1-μ0)^2] via RF outcome with W as feature."""
    Y = _as_1d(Y)
    W = _as_1d(W).astype(int)
    Xw = np.column_stack([X, W])
    m = rf_regressor(seed, n_estimators=n_estimators)
    m.fit(Xw, Y)
    # predict under W=0 and W=1
    x0 = np.column_stack([X, np.zeros(len(X))])
    x1 = np.column_stack([X, np.ones(len(X))])
    return float(np.mean((m.predict(x1) - m.predict(x0)) ** 2))


def cfperm_loco_pvalue(
    X: np.ndarray,
    Y: np.ndarray,
    W: np.ndarray,
    *,
    test_indices: np.ndarray,
    n_perm: int = 120,
    seed: int = 0,
    n_estimators: int = 50,
) -> dict:
    """
    CFPerm-LOCO style test on a *block* of coordinates `test_indices`.

    R-risk here is E[(μ_1(X)-μ_0(X))^2] (apparent batch effect on Y given X).
    Shift-explaining features *shrink* that score when included, so LOCO
    importance = R_minus_block - R_full (how much residual batch→Y grows
    after dropping the block). Permute W for the null; upper-tail p-value.
    """
    X = np.asarray(X, dtype=float)
    Y = _as_1d(Y)
    W = _as_1d(W).astype(int)
    idx = np.asarray(test_indices, dtype=int)
    keep = np.setdiff1d(np.arange(X.shape[1]), idx)

    r_full = _r_risk_score(X, Y, W, seed=seed, n_estimators=n_estimators)
    if keep.size == 0:
        r_minus = r_full
    else:
        r_minus = _r_risk_score(X[:, keep], Y, W, seed=seed + 1, n_estimators=n_estimators)
    imp = max(0.0, r_minus - r_full)

    rng = np.random.default_rng(seed + 17)
    null = np.zeros(n_perm)
    for b in range(n_perm):
        Wb = rng.permutation(W)
        rf = _r_risk_score(X, Y, Wb, seed=seed + 100 + b, n_estimators=n_estimators)
        if keep.size == 0:
            rm = rf
        else:
            rm = _r_risk_score(
                X[:, keep], Y, Wb, seed=seed + 200 + b, n_estimators=n_estimators
            )
        null[b] = max(0.0, rm - rf)

    p = (1.0 + np.sum(null >= imp - 1e-15)) / (1 + n_perm)
    return {
        "method": "CFPerm_loco_block",
        "statistic": float(imp),
        "p_value": float(p),
        "reject": p < 0.05,
        "n_test_features": int(len(idx)),
    }


@dataclass
class MethodResult:
    method: str
    feature_set: str
    scenario: str
    p_value: float
    reject: bool
    statistic: float
