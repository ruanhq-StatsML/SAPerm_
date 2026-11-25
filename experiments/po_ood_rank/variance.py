"""Observation-level variance of the doubly robust PO risk.

The pointwise pseudo-outcome residual is ``r = (phi - tau)^2``. Differentiating
in the CATE estimate gives the delta-method variance

    Var(r) ≈ 4 * r * Var(tau).

``Var(tau)`` is the infinitesimal jackknife of the pseudo-outcome forest, or
the squared standard error from a causal forest or an orthogonal forest.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression


def _as_1d(values) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def dr_pseudo_outcome(
    X, Y, W, *, n_estimators: int, min_leaf: int, seed: int, n_jobs: int, clip_e: float
) -> np.ndarray:
    """In-sample doubly robust pseudo-outcome for the batch contrast."""
    X = np.asarray(X, dtype=float)
    Y = _as_1d(Y)
    W = _as_1d(W).astype(int)
    leaf = max(2, min(int(min_leaf), max(2, (W == 0).sum() // 5)))
    forest_kw = dict(
        n_estimators=int(n_estimators),
        min_samples_leaf=leaf,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=int(n_jobs),
    )
    mu0 = RandomForestRegressor(random_state=seed, **forest_kw)
    mu1 = RandomForestRegressor(random_state=seed + 1, **forest_kw)
    mu0.fit(X[W == 0], Y[W == 0])
    mu1.fit(X[W == 1], Y[W == 1])
    mu0_hat = mu0.predict(X)
    mu1_hat = mu1.predict(X)
    propensity = LogisticRegression(C=1.0, solver="lbfgs", max_iter=400, random_state=seed + 2)
    propensity.fit(X, W)
    pi = np.clip(propensity.predict_proba(X)[:, 1], clip_e, 1.0 - clip_e)
    treated = W == 1
    residual = Y - np.where(treated, mu1_hat, mu0_hat)
    return mu1_hat - mu0_hat + residual * (W - pi) / (pi * (1.0 - pi))


def inbag_presence(forest) -> np.ndarray:
    """Bootstrap presence. A row drawn more than once still counts as one."""
    samples = forest.estimators_samples_
    n_trees = len(samples)
    n = int(getattr(forest, "n_samples_in_", samples[0].shape[0]))
    presence = np.zeros((n, n_trees), dtype=float)
    for b, idx in enumerate(samples):
        presence[np.asarray(idx, dtype=int), b] = 1.0
    return presence


def ij_variance_at(forest, X_new, bias_correction: str = "Wager") -> np.ndarray:
    """Jackknife variance of a fitted forest at new rows.

    Training in-bag indicators stay on the fit sample. Predictions are at
    ``X_new``, so this is the variance used on a held-out fold.
    """
    X_new = np.asarray(X_new, dtype=float)
    tree_preds = np.column_stack([tree.predict(X_new) for tree in forest.estimators_])
    inbag = inbag_presence(forest)
    n_trees = tree_preds.shape[1]
    centered_pred = tree_preds - np.mean(tree_preds, axis=1, keepdims=True)
    centered_count = inbag - np.mean(inbag, axis=1, keepdims=True)
    covariance = (centered_pred @ centered_count.T) / n_trees
    variance = np.sum(covariance ** 2, axis=1)
    if str(bias_correction).lower() == "wager":
        monte_carlo = np.var(tree_preds, ddof=1, axis=1) / n_trees
        variance = np.maximum(variance - monte_carlo, 0.0)
    return variance


def ij_variance(model_outputs: np.ndarray, inbag_times: np.ndarray, bias_correction: str = "Wager") -> np.ndarray:
    """Infinitesimal jackknife. The Wager branch subtracts ``Var(T_b) / B``."""
    outputs = np.asarray(model_outputs, dtype=float)
    inbag = np.asarray(inbag_times, dtype=float)
    if outputs.ndim != 2 or inbag.shape != outputs.shape:
        raise ValueError(f"expected matching (n, B) arrays, got {outputs.shape} and {inbag.shape}")
    n_trees = outputs.shape[1]
    centered_pred = outputs - np.mean(outputs, axis=1, keepdims=True)
    centered_count = inbag - np.mean(inbag, axis=1, keepdims=True)
    covariance = (centered_pred @ centered_count.T) / n_trees
    variance = np.sum(covariance ** 2, axis=1)
    if str(bias_correction).lower() == "wager":
        monte_carlo = np.var(outputs, ddof=1, axis=1) / n_trees
        variance = np.maximum(variance - monte_carlo, 0.0)
    return variance


def _po_forest(n_estimators: int, min_leaf: int, seed: int, n_jobs: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=int(n_estimators),
        min_samples_leaf=max(2, int(min_leaf)),
        max_features="sqrt",
        bootstrap=True,
        n_jobs=int(n_jobs),
        random_state=seed,
    )


def tau_variance_po(X, phi, *, n_estimators: int, min_leaf: int, seed: int, n_jobs: int) -> tuple[np.ndarray, np.ndarray]:
    """Pseudo-outcome forest. Variance is the bias-corrected jackknife."""
    forest = _po_forest(n_estimators, min_leaf, seed, n_jobs)
    target = _as_1d(phi)
    forest.fit(X, target)
    tau = forest.predict(X)
    tree_preds = np.column_stack([tree.predict(X) for tree in forest.estimators_])
    variance = ij_variance(tree_preds, inbag_presence(forest), bias_correction="Wager")
    return tau, variance


def tau_variance_cf(
    X, Y, W, *, n_estimators: int, min_leaf: int, seed: int, n_jobs: int
) -> tuple[np.ndarray, np.ndarray]:
    """Causal forest CATE and its squared standard error."""
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
    forest.fit(np.asarray(Y, dtype=float), np.asarray(W, dtype=int), X=np.asarray(X, dtype=float))
    tau = np.asarray(forest.effect(X), dtype=float).reshape(-1)
    stderr = np.asarray(forest.effect_inference(X).stderr, dtype=float).reshape(-1)
    return tau, np.nan_to_num(stderr, nan=0.0) ** 2


def tau_variance_orf(
    X, Y, W, *, n_estimators: int, min_leaf: int, subsample_ratio: float, seed: int, n_jobs: int
) -> tuple[np.ndarray, np.ndarray]:
    """Orthogonal forest CATE and its squared bootstrap standard error."""
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
    forest.fit(np.asarray(Y, dtype=float), np.asarray(W, dtype=int), X=np.asarray(X, dtype=float), inference="blb")
    query = np.asarray(X, dtype=float)
    tau = np.asarray(forest.effect(query, T0=0, T1=1), dtype=float).reshape(-1)
    stderr = np.asarray(forest.effect_inference(query, T0=0, T1=1).stderr, dtype=float).reshape(-1)
    return tau, np.nan_to_num(stderr, nan=0.0) ** 2


def fit_tau_variance(
    cate_model: str,
    X, Y, W, phi, *,
    n_estimators: int,
    min_leaf: int,
    subsample_ratio: float,
    seed: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    if cate_model == "PO":
        return tau_variance_po(X, phi, n_estimators=n_estimators, min_leaf=min_leaf, seed=seed, n_jobs=n_jobs)
    if cate_model == "CF":
        return tau_variance_cf(X, Y, W, n_estimators=n_estimators, min_leaf=min_leaf, seed=seed, n_jobs=n_jobs)
    if cate_model == "orf":
        return tau_variance_orf(
            X, Y, W,
            n_estimators=n_estimators,
            min_leaf=min_leaf,
            subsample_ratio=subsample_ratio,
            seed=seed,
            n_jobs=n_jobs,
        )
    raise ValueError(cate_model)


def pointwise_quantities(phi: np.ndarray, tau: np.ndarray, var_tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise PO risk and its delta-method variance ``4 * r * Var(tau)``."""
    residual = _as_1d(phi) - _as_1d(tau)
    risk = residual ** 2
    variance = 4.0 * risk * np.maximum(_as_1d(var_tau), 0.0)
    return risk, variance
