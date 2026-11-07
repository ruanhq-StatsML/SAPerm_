"""Cross-fit PO risk and ``Var(tau)`` on the held-out fold.

Same split as the cross-fit PermuCATE: stratify on the batch indicator, fit
the outcome, propensity, and CATE on the training fold, and score the test
fold only. Feature-permutation importance is a vector over columns, so it
does not enter the observation ranking. The ranking uses the held-out
residual magnitude and the held-out variance of tau.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from experiments.po_ood_rank.variance import _as_1d, _po_forest, ij_variance_at


def _dr_phi(Y, W, mu0, mu1, pi) -> np.ndarray:
    Y = _as_1d(Y)
    W = _as_1d(W)
    treated = W == 1
    residual = Y - np.where(treated, mu1, mu0)
    return (mu1 - mu0) + residual * (W - pi) / (pi * (1.0 - pi))


def _fit_nuisance(X, Y, W, *, n_estimators, min_leaf, seed, n_jobs, clip_e):
    X = np.asarray(X, dtype=float)
    Y = _as_1d(Y)
    W = _as_1d(W).astype(int)
    leaf = max(2, min(int(min_leaf), max(2, int((W == 0).sum()) // 5)))
    forest_kw = dict(
        n_estimators=int(n_estimators),
        min_samples_leaf=leaf,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=int(n_jobs),
    )
    mu0 = RandomForestRegressor(random_state=seed, **forest_kw)
    mu1 = RandomForestRegressor(random_state=seed + 1, **forest_kw)
    if np.unique(Y[W == 0]).size < 1 or (W == 0).sum() < leaf + 2:
        raise RuntimeError("control arm is too small to fit")
    mu0.fit(X[W == 0], Y[W == 0])
    mu1.fit(X[W == 1], Y[W == 1])
    propensity = LogisticRegression(C=1.0, solver="lbfgs", max_iter=400, random_state=seed + 2)
    propensity.fit(X, W)

    def phi_on(X_new, Y_new, W_new):
        mu0_hat = mu0.predict(X_new)
        mu1_hat = mu1.predict(X_new)
        pi = np.clip(propensity.predict_proba(X_new)[:, 1], clip_e, 1.0 - clip_e)
        return _dr_phi(Y_new, W_new, mu0_hat, mu1_hat, pi)

    return phi_on


def _tau_var_po(X_tr, phi_tr, X_te, *, n_estimators, min_leaf, seed, n_jobs):
    forest = _po_forest(n_estimators, min_leaf, seed, n_jobs)
    forest.fit(X_tr, _as_1d(phi_tr))
    tau = forest.predict(X_te)
    var_tau = ij_variance_at(forest, X_te, bias_correction="Wager")
    return tau, var_tau


def _tau_var_cf(X_tr, Y_tr, W_tr, X_te, *, n_estimators, min_leaf, seed, n_jobs):
    from econml.dml import CausalForestDML

    trees = int(n_estimators)
    subforest = next(size for size in (4, 3, 2, 1) if trees % size == 0)
    forest = CausalForestDML(
        n_estimators=trees,
        min_samples_leaf=max(2, int(min_leaf)),
        subforest_size=subforest,
        discrete_treatment=True,
        random_state=seed,
        n_jobs=int(n_jobs),
        verbose=0,
    )
    forest.fit(_as_1d(Y_tr), _as_1d(W_tr).astype(int), X=np.asarray(X_tr, dtype=float))
    query = np.asarray(X_te, dtype=float)
    tau = np.asarray(forest.effect(query), dtype=float).reshape(-1)
    stderr = np.asarray(forest.effect_inference(query).stderr, dtype=float).reshape(-1)
    return tau, np.nan_to_num(stderr, nan=0.0) ** 2


def _tau_var_orf(X_tr, Y_tr, W_tr, X_te, *, n_estimators, min_leaf, subsample_ratio, seed, n_jobs):
    from econml.orf import DMLOrthoForest

    forest = DMLOrthoForest(
        n_trees=int(n_estimators),
        min_leaf_size=max(2, int(min_leaf)),
        subsample_ratio=float(subsample_ratio),
        discrete_treatment=True,
        random_state=seed,
        n_jobs=int(n_jobs),
        verbose=0,
    )
    forest.fit(
        _as_1d(Y_tr),
        _as_1d(W_tr).astype(int),
        X=np.asarray(X_tr, dtype=float),
        inference="blb",
    )
    query = np.asarray(X_te, dtype=float)
    tau = np.asarray(forest.effect(query, T0=0, T1=1), dtype=float).reshape(-1)
    stderr = np.asarray(forest.effect_inference(query, T0=0, T1=1).stderr, dtype=float).reshape(-1)
    return tau, np.nan_to_num(stderr, nan=0.0) ** 2


def po_risk_cf(
    X, Y, W,
    *,
    n_folds: int = 5,
    n_estimators: int = 80,
    min_leaf: int = 20,
    seed: int = 0,
    n_jobs: int = 1,
    clip_e: float = 0.01,
) -> dict:
    """Held-out PO risk, the cross-fit used in place of an in-sample residual.

    ``po_risk`` is already ``(phi - tau)^2`` on the test fold. ``var_tau`` is
    the jackknife of that same fold's pseudo-outcome forest, scored on the
    held-out rows.
    """
    fit = crossfit_magnitude_variance(
        X, Y, W, "PO",
        n_folds=n_folds,
        n_estimators=n_estimators,
        min_leaf=min_leaf,
        seed=seed,
        n_jobs=n_jobs,
        clip_e=clip_e,
    )
    return {
        "pseudo_outcome": fit["phi"],
        "tau_est": fit["tau"],
        "po_risk": fit["magnitude"],
        "var_tau": fit["var_tau"],
    }


def crossfit_magnitude_variance(
    X, Y, W,
    cate_model: str,
    *,
    n_folds: int = 5,
    n_estimators: int = 80,
    min_leaf: int = 20,
    subsample_ratio: float = 0.7,
    seed: int = 0,
    n_jobs: int = 1,
    clip_e: float = 0.01,
    nuisance_estimators: int | None = None,
) -> dict:
    """Out-of-fold residual magnitude and ``Var(tau)`` for every row."""
    X = np.asarray(X, dtype=float)
    Y = _as_1d(Y)
    W = _as_1d(W).astype(int)
    n = X.shape[0]
    if cate_model not in {"PO", "CF", "orf"}:
        raise ValueError(cate_model)
    phi = np.full(n, np.nan)
    tau = np.full(n, np.nan)
    var_tau = np.full(n, np.nan)
    nuisance_trees = int(n_estimators if nuisance_estimators is None else nuisance_estimators)
    splitter = StratifiedKFold(n_splits=int(n_folds), shuffle=True, random_state=seed)
    for k, (train_idx, test_idx) in enumerate(splitter.split(X, W)):
        fold_seed = seed + k * 1000
        X_tr, Y_tr, W_tr = X[train_idx], Y[train_idx], W[train_idx]
        X_te, Y_te, W_te = X[test_idx], Y[test_idx], W[test_idx]
        phi_on = _fit_nuisance(
            X_tr, Y_tr, W_tr,
            n_estimators=nuisance_trees,
            min_leaf=min_leaf,
            seed=fold_seed,
            n_jobs=n_jobs,
            clip_e=clip_e,
        )
        phi[test_idx] = phi_on(X_te, Y_te, W_te)
        if cate_model == "PO":
            tau_te, var_te = _tau_var_po(
                X_tr, phi_on(X_tr, Y_tr, W_tr), X_te,
                n_estimators=n_estimators,
                min_leaf=min_leaf,
                seed=fold_seed + 3,
                n_jobs=n_jobs,
            )
        elif cate_model == "CF":
            tau_te, var_te = _tau_var_cf(
                X_tr, Y_tr, W_tr, X_te,
                n_estimators=n_estimators,
                min_leaf=min_leaf,
                seed=fold_seed + 3,
                n_jobs=n_jobs,
            )
        else:
            tau_te, var_te = _tau_var_orf(
                X_tr, Y_tr, W_tr, X_te,
                n_estimators=n_estimators,
                min_leaf=min_leaf,
                subsample_ratio=subsample_ratio,
                seed=fold_seed + 3,
                n_jobs=n_jobs,
            )
        tau[test_idx] = tau_te
        var_tau[test_idx] = np.maximum(var_te, 0.0)
    magnitude = (phi - tau) ** 2
    return {
        "phi": phi,
        "tau": tau,
        "var_tau": var_tau,
        "magnitude": magnitude,
        "product": magnitude * var_tau,
    }
