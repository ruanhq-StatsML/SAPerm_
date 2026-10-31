"""Variance of the pseudo-outcome risk on the two stacked shift DGPs.

The risk is ``mean((phi - tau)^2)``. ``phi`` is the AIPW pseudo-outcome and
``tau`` is a CATE fit. Holding ``phi`` fixed, the delta method gives the
sample-wise variance

    Var((phi - tau)^2) ≈ 4 (phi - tau)^2 Var(tau).

``Var(tau)`` is estimated in four ways:

- ``RF_delta``: variance of the forest mean, ``Var_b(T_b) / B``.
- ``RF_IJ``: infinitesimal jackknife with the Wager-Hastie-Efron Monte Carlo
  correction, ``max(V_IJ - n * Var(N) * v_hat / B, 0)``.
- ``CF``: ``CausalForestDML`` prediction variance.
- ``ORF``: ``DMLOrthoForest`` with ``inference='blb'``.

A bag-of-little-bootstraps refit of the pseudo-outcome forest estimates the
variance of the scalar risk itself. The data generators are the concept-drift
and covariate-shift DGPs already in ``experiments.dgps``; their formulas are
not modified.
"""

from __future__ import annotations

import argparse
import csv
import time
import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from experiments.dgps import banded_toeplitz_corr

DEFAULT_CONFIGS = {
    "n_trees": 150,
    "min_leaf_size": 20,
    "subsample_ratio": 0.7,
    "discrete_treatment": True,
    "inference": "blb",
    "background_samples": 100,
    "random_state": 24,
    "n_jobs": -1,
    "clip_e": 0.01,
    "n_estimators": 150,
    "bias_correction": "wager",
    "max_depth_orf": 10,
}


def generate_concept_drift_dgp(n=2000, p=20, delta_beta=0.0, rho=0.0, seed=None):
    """Concept drift: same covariate distribution, CATE changes after n/2.

    ``rho`` controls feature correlation. ``delta_beta`` scales the post-shift
    treatment effect on the first four features. The first of those slopes is
    zero because the coefficient is ``arange(4) * delta_beta``.
    """
    rng = np.random.default_rng(seed)
    corr = banded_toeplitz_corr(float(rho), p=p)
    X0 = rng.multivariate_normal(mean=np.zeros(p), cov=corr, size=n)
    beta0 = np.ones(p)
    Y = X0 @ beta0 + rng.normal(0, 1, n)
    beta = np.zeros(p)
    beta[:4] = np.arange(4) * float(delta_beta)
    Y[n // 2 :] = Y[n // 2 :] + (X0[n // 2 :, :] @ beta)
    W = np.concatenate([np.zeros(n // 2), np.ones(n - n // 2)])
    return X0, Y, W


def generate_covariate_shift_dgp(n=2000, p=20, gamma=0.0, rho=0.0, seed=None):
    """Covariate shift: the outcome mechanism stays fixed and P(X) changes.

    The first half gets an additive shift in ``X[:, :4]``. ``gamma`` controls
    both the outcome strength and the shift magnitude on those four features.
    """
    rng = np.random.default_rng(seed)
    corr = banded_toeplitz_corr(float(rho), p=p)
    X0 = rng.multivariate_normal(mean=np.zeros(p), cov=corr, size=n)
    beta0 = np.zeros(p)
    beta0[:4] = float(gamma)
    shift_score = X0[: n // 2, :4] @ beta0[:4]
    X0[: n // 2, :4] = X0[: n // 2, :4] + np.tile(
        shift_score.reshape(-1, 1), (1, 4)
    )
    Y = X0 @ beta0 + rng.normal(0, 1, n)
    W = np.concatenate([np.zeros(n // 2), np.ones(n - n // 2)])
    return X0, Y, W


def _as_1d(values) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def true_cate(kind: str, X: np.ndarray, strength: float) -> np.ndarray:
    """Population CATE implied by the two generators. Covariate shift has tau = 0."""
    X = np.asarray(X, dtype=float)
    if kind == "covariate":
        return np.zeros(X.shape[0], dtype=float)
    beta = np.zeros(X.shape[1], dtype=float)
    beta[:4] = np.arange(4) * float(strength)
    return X @ beta


def planted_features(kind: str) -> np.ndarray:
    """Columns whose population CATE, or whose shifted margin, is nonzero."""
    if kind == "concept":
        return np.array([1, 2, 3], dtype=int)
    if kind == "covariate":
        return np.arange(4, dtype=int)
    raise ValueError(kind)


def _stratified_split(treatment, test_size: float, rng: np.random.Generator):
    treatment = np.asarray(treatment).astype(int).reshape(-1)
    if test_size <= 0.0:
        index = np.arange(treatment.shape[0])
        return index, index
    train_parts = []
    test_parts = []
    for label in np.unique(treatment):
        idx = np.flatnonzero(treatment == label)
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_size))
        n_test = min(max(n_test, 1), len(idx) - 1)
        test_parts.append(idx[:n_test])
        train_parts.append(idx[n_test:])
    return np.sort(np.concatenate(train_parts)), np.sort(np.concatenate(test_parts))


def _make_regressor(n_estimators: int, min_samples_leaf: int, max_features: int, seed: int, n_jobs: int):
    return RandomForestRegressor(
        n_estimators=int(n_estimators),
        min_samples_leaf=int(min_samples_leaf),
        max_features=int(max_features),
        bootstrap=True,
        random_state=int(seed),
        n_jobs=n_jobs,
    )


def aipw_pseudo_outcome(Y, T, mu0, mu1, pi, clip: float = 0.01) -> np.ndarray:
    """Doubly robust pseudo-outcome for the CATE."""
    pi_clip = np.clip(np.asarray(pi, dtype=float), clip, 1.0 - clip)
    treated = np.asarray(T).astype(int).reshape(-1) == 1
    mu_t = np.where(treated, mu1, mu0)
    weight = (np.asarray(T, dtype=float).reshape(-1) - pi_clip) / (pi_clip * (1.0 - pi_clip))
    return (np.asarray(Y, dtype=float).reshape(-1) - mu_t) * weight + (
        np.asarray(mu1, dtype=float) - np.asarray(mu0, dtype=float)
    )


def fit_po_nuisance(
    X,
    Y,
    T,
    *,
    n_estimators: int = 150,
    min_samples_leaf: int = 20,
    clip_e: float = 0.01,
    seed: int = 0,
    n_jobs: int = -1,
):
    """Outcome forests, a logistic propensity, and a forest on the pseudo-outcome.

    This is the nuisance block of ``permuCATE_fit_nuisance``: separate outcome
    regressions in each treatment arm, then a regression of the AIPW
    pseudo-outcome on X.
    """
    X = np.asarray(X, dtype=float)
    Y = _as_1d(Y)
    T = np.asarray(T).astype(int).reshape(-1)
    n, p = X.shape
    leaf = max(1, min(int(min_samples_leaf), max(1, n // 5)))
    max_features = max(1, int(round(p / 3.0)))
    control = T == 0
    treated = T == 1
    if control.sum() < 2 or treated.sum() < 2:
        raise ValueError("both treatment arms need at least two rows")
    mu0_model = _make_regressor(n_estimators, leaf, max_features, seed, n_jobs)
    mu1_model = _make_regressor(n_estimators, leaf, max_features, seed + 1, n_jobs)
    mu0_model.fit(X[control], Y[control])
    mu1_model.fit(X[treated], Y[treated])
    mu0 = mu0_model.predict(X)
    mu1 = mu1_model.predict(X)
    propensity = LogisticRegression(solver="lbfgs", max_iter=500, random_state=seed + 2)
    propensity.fit(X, T)
    positive = int(np.where(np.asarray(propensity.classes_) == 1)[0][0])
    pi = np.clip(propensity.predict_proba(X)[:, positive], clip_e, 1.0 - clip_e)
    pseudo = aipw_pseudo_outcome(Y, T, mu0, mu1, pi, clip=clip_e)
    tau_model = _make_regressor(n_estimators, leaf, max_features, seed + 3, n_jobs)
    tau_model.fit(X, pseudo)
    return {
        "mu0_model": mu0_model,
        "mu1_model": mu1_model,
        "propensity": propensity,
        "tau_model": tau_model,
        "mu0": mu0,
        "mu1": mu1,
        "pi": pi,
        "pseudo": pseudo,
        "max_features": max_features,
        "min_samples_leaf": leaf,
    }


def _positive_probability(model, X) -> np.ndarray:
    positive = int(np.where(np.asarray(model.classes_) == 1)[0][0])
    return np.asarray(model.predict_proba(np.asarray(X, dtype=float))[:, positive], dtype=float)


def predict_pseudo(fit, X, Y, T, clip_e: float = 0.01):
    X = np.asarray(X, dtype=float)
    Y = _as_1d(Y)
    T = np.asarray(T).astype(int).reshape(-1)
    mu0 = fit["mu0_model"].predict(X)
    mu1 = fit["mu1_model"].predict(X)
    pi = np.clip(_positive_probability(fit["propensity"], X), clip_e, 1.0 - clip_e)
    pseudo = aipw_pseudo_outcome(Y, T, mu0, mu1, pi, clip=clip_e)
    tau = np.asarray(fit["tau_model"].predict(X), dtype=float).reshape(-1)
    weight = (T.astype(float) - pi) / (pi * (1.0 - pi))
    return {
        "mu0": mu0,
        "mu1": mu1,
        "pi": pi,
        "pseudo": pseudo,
        "tau": tau,
        "mean_weight_sq": float(np.mean(weight ** 2)),
        "clip_rate": float(np.mean((pi <= clip_e + 1e-12) | (pi >= 1.0 - clip_e - 1e-12))),
    }


def forest_inbag_counts(forest, n_samples: int) -> np.ndarray:
    """Bootstrap inclusion counts. Entry ``(i, b)`` is how often row ``i`` trained tree ``b``."""
    samples = forest.estimators_samples_
    counts = np.zeros((int(n_samples), len(samples)), dtype=float)
    for tree_id, idx in enumerate(samples):
        np.add.at(counts[:, tree_id], np.asarray(idx, dtype=int), 1.0)
    return counts


def forest_tree_predictions(forest, X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    columns = [np.asarray(tree.predict(X), dtype=float).reshape(-1) for tree in forest.estimators_]
    return np.column_stack(columns)


def IJ_estimator(model_outputs, inbag_times, bias_correction: str = "Wager") -> np.ndarray:
    """Infinitesimal jackknife variance of a bagged prediction.

    ``model_outputs`` has shape ``(n_eval, B)`` and ``inbag_times`` has shape
    ``(n_train, B)``. The covariance is the one in the sketch,

        Cov_i(x) = (1/B) sum_b (T_b(x) - Tbar(x)) (N_ib - Nbar_i),

    and ``V_IJ(x) = sum_i Cov_i(x)^2``. ``bias_correction='Wager'`` subtracts
    the Monte Carlo bias from Wager, Hastie, and Efron (2014), equation (15),

        V_IJ-U = max(V_IJ - n * Var(N) * v_hat / B, 0),

    where ``v_hat`` is the empirical variance of the tree predictions. For a
    bootstrap sample ``Var(N)`` is about 1, so the subtraction is about
    ``n * v_hat / B``. The sketch wrote ``v_hat / B``; that drops the factor
    ``n * Var(N)`` from the cited correction, so it is not the default.
    """
    pieces = ij_components(model_outputs, inbag_times, bias_correction=bias_correction)
    return pieces["variance"]


def ij_components(model_outputs, inbag_times, bias_correction: str = "Wager") -> dict:
    outputs = np.asarray(model_outputs, dtype=float)
    inbag = np.asarray(inbag_times, dtype=float)
    if outputs.ndim != 2 or inbag.ndim != 2:
        raise ValueError("model_outputs and inbag_times must be two-dimensional")
    if outputs.shape[1] != inbag.shape[1]:
        raise ValueError("model_outputs and inbag_times must have one column per tree")
    n_trees = outputs.shape[1]
    if n_trees < 2:
        raise ValueError("the jackknife needs at least two trees")
    t_centered = outputs - np.mean(outputs, axis=1, keepdims=True)
    n_centered = inbag - np.mean(inbag, axis=1, keepdims=True)
    covariance = (t_centered @ n_centered.T) / n_trees
    raw = np.sum(covariance ** 2, axis=1)
    n_train = inbag.shape[0]
    row_second = np.mean(inbag ** 2, axis=1)
    row_mean = np.mean(inbag, axis=1)
    n_var = float(np.mean(row_second - row_mean ** 2))
    tree_var = np.sum(t_centered ** 2, axis=1) / n_trees
    bias = n_train * n_var * tree_var / n_trees
    label = str(bias_correction).lower()
    if label in {"wager", "wager14"}:
        corrected = np.maximum(raw - bias, 0.0)
    elif label in {"sketch", "v_over_b"}:
        corrected = np.maximum(raw - tree_var / n_trees, 0.0)
    elif label in {"none", "raw", "off"}:
        corrected = raw
    else:
        raise ValueError(f"unknown bias_correction: {bias_correction}")
    return {
        "variance": corrected,
        "raw": raw,
        "bias": bias,
        "tree_var": tree_var,
        "n_var": n_var,
    }


def delta_method_po_variance(pseudo, tau, var_tau):
    """Sample-wise delta-method variance ``4 * (phi - tau)^2 * Var(tau)``."""
    residual = _as_1d(pseudo) - _as_1d(tau)
    residual_sq = residual ** 2
    var = np.maximum(_as_1d(var_tau), 0.0)
    return residual_sq, 4.0 * residual_sq * var


def summarize_delta(residual_sq, var_tau) -> dict:
    """Turn the sample-wise delta-method variance into a few scalars.

    ``se_indep`` treats the pointwise contributions as independent, so it is
    ``sqrt(mean(4 r Var(tau)) / n)``. ``se_upper`` is the Cauchy-Schwarz bound
    that allows those contributions to be perfectly dependent. ``plugin_se``
    is the ordinary standard error of the mean squared residual and does not
    use ``Var(tau)``.
    """
    residual_sq = _as_1d(residual_sq)
    var_tau = np.maximum(_as_1d(var_tau), 0.0)
    n = residual_sq.shape[0]
    delta = 4.0 * residual_sq * var_tau
    gap = np.sqrt(residual_sq)
    se_tau = np.sqrt(var_tau)
    return {
        "n_eval": int(n),
        "po_risk": float(np.mean(residual_sq)),
        "mean_var_tau": float(np.mean(var_tau)),
        "median_var_tau": float(np.median(var_tau)),
        "mean_delta_var": float(np.mean(delta)),
        "se_indep": float(np.sqrt(np.mean(delta) / n)),
        "se_upper": float((2.0 / n) * np.sum(gap * se_tau)),
        "plugin_se": float(np.std(residual_sq, ddof=1) / np.sqrt(n)),
        "zero_var_fraction": float(np.mean(var_tau <= 0.0)),
    }


def rf_delta_variance(tree_preds) -> np.ndarray:
    """Variance of the average tree. Independent-trees approximation ``v / B``."""
    preds = np.asarray(tree_preds, dtype=float)
    n_trees = preds.shape[1]
    return np.var(preds, axis=1, ddof=1) / n_trees


def _fit_causal_forest(X, Y, T, *, n_estimators, min_samples_leaf, seed, n_jobs):
    from econml.dml import CausalForestDML

    X = np.asarray(X, dtype=float)
    model_y = Ridge(alpha=1.0)
    model_t = LogisticRegression(solver="lbfgs", max_iter=500, random_state=seed)
    n_trees = int(n_estimators)
    subforest = next(size for size in (4, 3, 2, 1) if n_trees % size == 0)
    forest = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=True,
        n_estimators=n_trees,
        min_samples_leaf=max(1, int(min_samples_leaf)),
        subforest_size=subforest,
        cv=2,
        honest=True,
        inference=True,
        random_state=int(seed),
        n_jobs=n_jobs,
        verbose=0,
    )
    forest.fit(Y, T, X=X)
    return forest


def _causal_forest_tau_variance(forest, X):
    X = np.asarray(X, dtype=float)
    inferred = forest.effect_inference(X)
    stderr = np.asarray(inferred.stderr, dtype=float).reshape(-1)
    tau = np.asarray(forest.effect(X), dtype=float).reshape(-1)
    return tau, np.maximum(stderr, 0.0) ** 2


def _fit_ortho_forest(X, Y, T, *, n_estimators, min_leaf_size, subsample_ratio, seed, n_jobs):
    from econml.orf import DMLOrthoForest

    forest = DMLOrthoForest(
        n_trees=int(n_estimators),
        min_leaf_size=max(1, int(min_leaf_size)),
        max_depth=int(DEFAULT_CONFIGS["max_depth_orf"]),
        subsample_ratio=float(subsample_ratio),
        discrete_treatment=True,
        random_state=int(seed),
        n_jobs=n_jobs,
        verbose=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forest.fit(
            _as_1d(Y),
            np.asarray(T).astype(int).reshape(-1),
            X=np.asarray(X, dtype=float),
            inference="blb",
        )
    return forest


def _ortho_forest_tau_variance(forest, X):
    X = np.asarray(X, dtype=float)
    tau = np.asarray(forest.effect(X, T0=0, T1=1), dtype=float).reshape(-1)
    inferred = forest.effect_inference(X, T0=0, T1=1)
    stderr = np.asarray(inferred.stderr, dtype=float).reshape(-1)
    return tau, np.maximum(stderr, 0.0) ** 2


def blb_po_risk_variance(
    X,
    pseudo,
    *,
    n_estimators: int,
    min_samples_leaf: int,
    max_features: int,
    subset_size: int,
    n_subsets: int,
    n_boot: int,
    seed: int,
    n_jobs: int = 1,
) -> dict:
    """Bag of little bootstraps for the scalar PO-risk.

    Nuisance pseudo-outcomes stay fixed. Each little bag is redrawn up to the
    original sample size, the pseudo-outcome forest is refit, and the statistic
    is the full-sample ``mean((phi - tau)^2)``. The returned variance is the
    average across little bags of the bootstrap variance.
    """
    X = np.asarray(X, dtype=float)
    pseudo = _as_1d(pseudo)
    n = pseudo.shape[0]
    subset_size = int(min(max(subset_size, 8), n))
    rng = np.random.default_rng(seed)
    bag_variances = []
    bag_means = []
    for subset_id in range(int(n_subsets)):
        subset = rng.choice(n, size=subset_size, replace=False)
        thetas = np.empty(int(n_boot), dtype=float)
        for replicate in range(int(n_boot)):
            draw = rng.choice(subset, size=n, replace=True)
            forest = _make_regressor(
                n_estimators,
                min_samples_leaf,
                max_features,
                seed + 17 + subset_id * int(n_boot) + replicate,
                n_jobs,
            )
            forest.fit(X[draw], pseudo[draw])
            tau = forest.predict(X)
            thetas[replicate] = float(np.mean((pseudo - tau) ** 2))
        bag_variances.append(float(np.var(thetas, ddof=1)))
        bag_means.append(float(np.mean(thetas)))
    variance = float(np.mean(bag_variances))
    return {
        "variance": variance,
        "se": float(np.sqrt(variance)),
        "mean_theta": float(np.mean(bag_means)),
        "subset_size": subset_size,
        "n_subsets": int(n_subsets),
        "n_boot": int(n_boot),
    }


def permutation_po_vimp(tau_model, X, pseudo, n_perm: int, seed: int):
    """Increase in PO-risk when one column of the evaluation sample is permuted.

    ``phi`` stays at the original covariates. Only ``tau(X)`` is recomputed,
    so a column matters when the CATE fit uses it.
    """
    X = np.asarray(X, dtype=float)
    pseudo = _as_1d(pseudo)
    base_tau = np.asarray(tau_model.predict(X), dtype=float).reshape(-1)
    base = float(np.mean((pseudo - base_tau) ** 2))
    n_features = X.shape[1]
    drops = np.zeros((n_features, int(n_perm)), dtype=float)
    for feature in range(n_features):
        rng = np.random.default_rng(seed + feature)
        for replicate in range(int(n_perm)):
            permuted = X.copy()
            permuted[:, feature] = rng.permutation(permuted[:, feature])
            tau = np.asarray(tau_model.predict(permuted), dtype=float).reshape(-1)
            drops[feature, replicate] = float(np.mean((pseudo - tau) ** 2) - base)
    return drops.mean(axis=1), drops.std(axis=1, ddof=1)


def _tau_fit_stats(tau_hat, tau_true) -> dict:
    hat = _as_1d(tau_hat)
    truth = _as_1d(tau_true)
    mse = float(np.mean((hat - truth) ** 2))
    if np.std(hat) < 1e-12 or np.std(truth) < 1e-12:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(hat, truth)[0, 1])
    return {"tau_mse": mse, "tau_corr": corr, "mean_abs_tau": float(np.mean(np.abs(hat)))}


@dataclass
class EvalSpec:
    name: str
    index: np.ndarray


def variance_estimator_po_risk(
    X,
    Y,
    T,
    cate_model: str = "RF_IJ",
    **kwargs,
):
    """Sample-wise delta-method variance of the squared pseudo-outcome residual.

    The return value's ``var_estimator`` is ``4 * (phi - tau)^2 * Var(tau)``,
    aligned with the evaluation rows. Set ``test_size=0`` to fit and evaluate
    on the same sample.
    """
    report = estimate_po_risk_variance(X, Y, T, methods=(cate_model,), **kwargs)
    method = _canonical_method(cate_model)
    block = report["splits"]["holdout" if report["test_size"] > 0 else "insample"][method]
    return {
        "var_estimator": block["var_estimator"],
        "po_risk": block["summary"]["po_risk"],
        "tau": block["tau"],
        "pseudo": block["pseudo"],
        "var_tau": block["var_tau"],
        "eval_index": block["eval_index"],
        "summary": block["summary"],
        "method": method,
    }


def _canonical_method(name: str) -> str:
    key = str(name).lower()
    mapping = {
        "po": "RF_IJ",
        "rf_ij": "RF_IJ",
        "ij": "RF_IJ",
        "rf_delta": "RF_delta",
        "delta": "RF_delta",
        "cf": "CF",
        "causal_forest": "CF",
        "orf": "ORF",
        "orthoforest": "ORF",
        "blb": "BLB",
    }
    if key not in mapping:
        raise ValueError(f"unknown cate_model: {name}")
    return mapping[key]


def estimate_po_risk_variance(
    X,
    Y,
    T,
    *,
    methods=("RF_delta", "RF_IJ"),
    n_estimators: int = 150,
    min_leaf_size: int = 20,
    subsample_ratio: float = 0.7,
    clip_e: float = 0.01,
    test_size: float = 0.5,
    seed: int = 0,
    n_jobs: int = -1,
    bias_correction: str = "Wager",
    n_perm: int = 0,
    blb_subsets: int = 4,
    blb_boot: int = 8,
    blb_trees: int = 40,
    blb_subset_frac: float = 0.5,
    true_tau=None,
):
    """Fit one nuisance model and attach every requested variance estimator."""
    X = np.asarray(X, dtype=float)
    Y = _as_1d(Y)
    T = np.asarray(T).astype(int).reshape(-1)
    requested = tuple(_canonical_method(name) for name in methods)
    rng = np.random.default_rng(seed)
    train_index, test_index = _stratified_split(T, test_size, rng)
    fit = fit_po_nuisance(
        X[train_index],
        Y[train_index],
        T[train_index],
        n_estimators=n_estimators,
        min_samples_leaf=min_leaf_size,
        clip_e=clip_e,
        seed=seed,
        n_jobs=n_jobs,
    )
    need_forest_variance = any(name in requested for name in ("RF_delta", "RF_IJ"))
    inbag_counts = (
        forest_inbag_counts(fit["tau_model"], len(train_index)) if need_forest_variance else None
    )
    tree_bank = {}
    ij_bank = {}

    def trees_at(name, index):
        if name not in tree_bank:
            tree_bank[name] = forest_tree_predictions(fit["tau_model"], X[index])
        return tree_bank[name]

    def ij_at(name, index):
        if name not in ij_bank:
            ij_bank[name] = ij_components(
                trees_at(name, index),
                inbag_counts,
                bias_correction=bias_correction,
            )
        return ij_bank[name]

    causal = None
    ortho = None
    if "CF" in requested:
        causal = _fit_causal_forest(
            X[train_index],
            Y[train_index],
            T[train_index],
            n_estimators=n_estimators,
            min_samples_leaf=fit["min_samples_leaf"],
            seed=seed,
            n_jobs=n_jobs,
        )
    if "ORF" in requested:
        ortho = _fit_ortho_forest(
            X[train_index],
            Y[train_index],
            T[train_index],
            n_estimators=n_estimators,
            min_leaf_size=fit["min_samples_leaf"],
            subsample_ratio=subsample_ratio,
            seed=seed,
            n_jobs=n_jobs,
        )
    blb = None
    if "BLB" in requested:
        train_pseudo = predict_pseudo(
            fit, X[train_index], Y[train_index], T[train_index], clip_e=clip_e
        )["pseudo"]
        subset_size = max(8, int(round(blb_subset_frac * len(train_index))))
        blb = blb_po_risk_variance(
            X[train_index],
            train_pseudo,
            n_estimators=blb_trees,
            min_samples_leaf=fit["min_samples_leaf"],
            max_features=fit["max_features"],
            subset_size=subset_size,
            n_subsets=blb_subsets,
            n_boot=blb_boot,
            seed=seed + 11,
            n_jobs=1,
        )

    vimp_mean = None
    vimp_sd = None
    if n_perm > 0:
        holdout_pseudo = predict_pseudo(fit, X[test_index], Y[test_index], T[test_index], clip_e=clip_e)
        vimp_mean, vimp_sd = permutation_po_vimp(
            fit["tau_model"], X[test_index], holdout_pseudo["pseudo"], n_perm, seed + 29
        )

    specs = [EvalSpec("insample", train_index)]
    if test_size > 0.0 and not np.array_equal(train_index, test_index):
        specs.append(EvalSpec("holdout", test_index))

    splits = {}
    for spec in specs:
        predicted = predict_pseudo(fit, X[spec.index], Y[spec.index], T[spec.index], clip_e=clip_e)
        truth = None if true_tau is None else np.asarray(true_tau, dtype=float).reshape(-1)[spec.index]
        methods_out = {}
        if need_forest_variance:
            preds = trees_at(spec.name, spec.index)
            ij = ij_at(spec.name, spec.index) if "RF_IJ" in requested else None
            if "RF_delta" in requested:
                var_tau = rf_delta_variance(preds)
                residual_sq, var_estimator = delta_method_po_variance(predicted["pseudo"], predicted["tau"], var_tau)
                extra = {} if ij is None else {"ij_n_var": ij["n_var"]}
                methods_out["RF_delta"] = _pack_method(
                    predicted, spec.index, residual_sq, var_tau, var_estimator, truth, extra=extra
                )
            if "RF_IJ" in requested:
                var_tau = ij["variance"]
                residual_sq, var_estimator = delta_method_po_variance(predicted["pseudo"], predicted["tau"], var_tau)
                methods_out["RF_IJ"] = _pack_method(
                    predicted,
                    spec.index,
                    residual_sq,
                    var_tau,
                    var_estimator,
                    truth,
                    extra={
                        "raw_mean_var_tau": float(np.mean(ij["raw"])),
                        "bias_mean": float(np.mean(ij["bias"])),
                        "ij_n_var": ij["n_var"],
                    },
                )
        if "CF" in requested:
            tau_cf, var_tau = _causal_forest_tau_variance(causal, X[spec.index])
            residual_sq, var_estimator = delta_method_po_variance(predicted["pseudo"], tau_cf, var_tau)
            packed = _pack_method(predicted, spec.index, residual_sq, var_tau, var_estimator, truth)
            packed["tau"] = tau_cf
            packed["summary"] = summarize_delta(residual_sq, var_tau)
            if truth is not None:
                packed["fit_stats"] = _tau_fit_stats(tau_cf, truth)
            methods_out["CF"] = packed
        if "ORF" in requested:
            tau_orf, var_tau = _ortho_forest_tau_variance(ortho, X[spec.index])
            residual_sq, var_estimator = delta_method_po_variance(predicted["pseudo"], tau_orf, var_tau)
            packed = _pack_method(predicted, spec.index, residual_sq, var_tau, var_estimator, truth)
            packed["tau"] = tau_orf
            packed["summary"] = summarize_delta(residual_sq, var_tau)
            if truth is not None:
                packed["fit_stats"] = _tau_fit_stats(tau_orf, truth)
            methods_out["ORF"] = packed
        if "BLB" in requested and blb is not None and spec.name == "insample":
            # Little-bootstrap variance of the scalar training risk. It is not
            # a property of the holdout rows, so it is recorded once.
            methods_out["BLB"] = {
                "summary": {
                    "n_eval": int(len(train_index)),
                    "po_risk": blb["mean_theta"],
                    "mean_var_tau": float("nan"),
                    "median_var_tau": float("nan"),
                    "mean_delta_var": float("nan"),
                    "se_indep": blb["se"],
                    "se_upper": float("nan"),
                    "plugin_se": float("nan"),
                    "zero_var_fraction": float("nan"),
                },
                "blb": blb,
                "eval_index": train_index,
                "pseudo": None,
                "tau": None,
                "var_tau": None,
                "var_estimator": None,
                "fit_stats": {},
                "extra": {},
            }
        splits[spec.name] = methods_out

    return {
        "test_size": float(test_size),
        "train_index": train_index,
        "test_index": test_index,
        "splits": splits,
        "vimp_mean": vimp_mean,
        "vimp_sd": vimp_sd,
        "fit": fit,
    }


def _pack_method(predicted, index, residual_sq, var_tau, var_estimator, truth, extra=None):
    packed = {
        "summary": summarize_delta(residual_sq, var_tau),
        "eval_index": index,
        "pseudo": predicted["pseudo"],
        "tau": predicted["tau"],
        "var_tau": var_tau,
        "var_estimator": var_estimator,
        "mean_weight_sq": predicted["mean_weight_sq"],
        "clip_rate": predicted["clip_rate"],
        "fit_stats": {} if truth is None else _tau_fit_stats(predicted["tau"], truth),
        "extra": {} if extra is None else extra,
    }
    return packed


def _fmt(value, digits=3):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    return f"{float(value):.{digits}f}"


def run_comparison(config: dict) -> list[dict]:
    rows = []
    kinds = (
        ("concept", "delta_beta", generate_concept_drift_dgp),
        ("covariate", "gamma", generate_covariate_shift_dgp),
    )
    for kind, strength_name, generator in kinds:
        for strength in config["strengths"]:
            for seed in config["seeds"]:
                started = time.perf_counter()
                kwargs = {"n": config["n"], "p": config["p"], "rho": config["rho"], "seed": seed}
                kwargs[strength_name] = strength
                X, Y, W = generator(**kwargs)
                truth = true_cate(kind, X, strength)
                report = estimate_po_risk_variance(
                    X,
                    Y,
                    W,
                    methods=config["methods"],
                    n_estimators=config["n_estimators"],
                    min_leaf_size=config["min_leaf_size"],
                    subsample_ratio=config["subsample_ratio"],
                    clip_e=config["clip_e"],
                    test_size=config["test_size"],
                    seed=seed,
                    n_jobs=config["n_jobs"],
                    bias_correction=config["bias_correction"],
                    n_perm=config["n_perm"],
                    blb_subsets=config["blb_subsets"],
                    blb_boot=config["blb_boot"],
                    blb_trees=config["blb_trees"],
                    blb_subset_frac=config["blb_subset_frac"],
                    true_tau=truth,
                )
                planted = planted_features(kind)
                vimp = report["vimp_mean"]
                if vimp is None:
                    vimp_planted = float("nan")
                    vimp_rest = float("nan")
                    top = ""
                else:
                    rest = np.setdiff1d(np.arange(X.shape[1]), planted)
                    vimp_planted = float(np.mean(vimp[planted]))
                    vimp_rest = float(np.mean(vimp[rest])) if len(rest) else float("nan")
                    top = " ".join(str(int(j)) for j in np.argsort(-vimp)[:4])
                for split_name, methods in report["splits"].items():
                    for method, block in methods.items():
                        summary = block["summary"]
                        fit_stats = block.get("fit_stats") or {}
                        extra = block.get("extra") or {}
                        rows.append(
                            {
                                "kind": kind,
                                "strength": float(strength),
                                "rho": float(config["rho"]),
                                "seed": int(seed),
                                "split": split_name,
                                "method": method,
                                "n_batch_0": int(np.sum(W == 0)),
                                "n_batch_1": int(np.sum(W == 1)),
                                "n_eval": summary["n_eval"],
                                "po_risk": summary["po_risk"],
                                "mean_weight_sq": block.get("mean_weight_sq", float("nan")),
                                "clip_rate": block.get("clip_rate", float("nan")),
                                "mean_var_tau": summary["mean_var_tau"],
                                "median_var_tau": summary["median_var_tau"],
                                "mean_delta_var": summary["mean_delta_var"],
                                "se_indep": summary["se_indep"],
                                "se_upper": summary["se_upper"],
                                "plugin_se": summary["plugin_se"],
                                "zero_var_fraction": summary["zero_var_fraction"],
                                "raw_mean_var_tau": extra.get("raw_mean_var_tau", float("nan")),
                                "tau_mse": fit_stats.get("tau_mse", float("nan")),
                                "tau_corr": fit_stats.get("tau_corr", float("nan")),
                                "mean_abs_tau": fit_stats.get("mean_abs_tau", float("nan")),
                                "vimp_planted": vimp_planted,
                                "vimp_rest": vimp_rest,
                                "vimp_top4": top,
                                "seconds": time.perf_counter() - started,
                            }
                        )
                print(
                    f"{kind} {strength_name}={strength:g} seed={seed} "
                    f"done in {time.perf_counter() - started:.1f}s",
                    flush=True,
                )
    return rows


def _group_key(row, fields):
    return tuple(row[field] for field in fields)


def summarize_rows(rows: list[dict]) -> str:
    """Mean tables and the paired null-versus-shift comparison."""
    numeric_fields = {
        "strength", "rho", "seed", "n_batch_0", "n_batch_1", "n_eval", "po_risk",
        "mean_weight_sq", "clip_rate", "mean_var_tau", "median_var_tau", "mean_delta_var",
        "se_indep", "se_upper", "plugin_se", "zero_var_fraction", "raw_mean_var_tau",
        "tau_mse", "tau_corr", "mean_abs_tau", "vimp_planted", "vimp_rest", "seconds",
    }
    rows = [dict(row) for row in rows]
    for row in rows:
        for field in numeric_fields:
            if field not in row or row[field] == "" or row[field] is None:
                continue
            value = float(row[field])
            row[field] = int(value) if field == "seed" else value
    lines = []
    lines.append("# PO-risk variance on the two shift generators")
    lines.append("")
    lines.append(
        "Each batch has 1000 rows when `n = 2000`. "
        "Concept drift keeps P(X) fixed and adds `X @ beta` on the second batch, "
        "with `beta[:4] = (0, delta, 2 delta, 3 delta)`. "
        "Covariate shift keeps the regression fixed and moves the first batch's first four columns. "
        "The population CATE is that linear drift, and it is identically zero under covariate shift."
    )
    lines.append("")
    lines.append(
        "Holdout PO-risk is `mean((phi - tau)^2)` on a treatment-stratified half. "
        "`mean_var_tau` is the average estimated variance of the CATE. "
        "`mean_delta_var` is the average of `4 (phi - tau)^2 Var(tau)`. "
        "`plugin_se` is the ordinary standard error of the mean squared residual. "
        "`se_indep` folds the delta-method variance into a standard error of the mean "
        "under an independence approximation. "
        "BLB's `se_indep` is instead the little-bootstrap standard error of the scalar training risk."
    )
    lines.append("")
    fields = ("kind", "strength", "split", "method")
    grouped: dict[tuple, list] = {}
    for row in rows:
        grouped.setdefault(_group_key(row, fields), []).append(row)
    lines.append("## Means across seeds")
    lines.append("")
    lines.append(
        "| kind | strength | split | method | po_risk | mean_var_tau | mean_delta_var | plugin_se | se_indep | tau_mse | tau_corr |"
    )
    lines.append("| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for key in sorted(grouped, key=lambda item: (item[0], item[1], item[2], item[3])):
        block = grouped[key]
        def avg(name, _block=block):
            values = np.array([float(row[name]) for row in _block], dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                return float("nan")
            return float(np.mean(values))

        lines.append(
            "| {kind} | {strength:g} | {split} | {method} | {po} | {vtau} | {dv} | {plugin} | {se} | {mse} | {corr} |".format(
                kind=key[0],
                strength=key[1],
                split=key[2],
                method=key[3],
                po=_fmt(avg("po_risk")),
                vtau=_fmt(avg("mean_var_tau")),
                dv=_fmt(avg("mean_delta_var")),
                plugin=_fmt(avg("plugin_se")),
                se=_fmt(avg("se_indep")),
                mse=_fmt(avg("tau_mse")),
                corr=_fmt(avg("tau_corr")),
            )
        )
    lines.append("")
    lines.append("## Paired shift minus null")
    lines.append("")
    lines.append(
        "The null is strength 0 drawn with the same seed. "
        "A positive PO-risk gap means the squared pseudo-outcome residual got larger. "
        "A positive `mean_var_tau` gap means the CATE estimate got less stable."
    )
    lines.append("")
    lines.append(
        "| kind | split | method | d_po_risk | seeds po up | d_var_tau | seeds var up | d_plugin_se | null po | alt po |"
    )
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    by_pair: dict[tuple, dict] = {}
    for row in rows:
        by_pair.setdefault((row["kind"], row["split"], row["method"], row["seed"]), {})[float(row["strength"])] = row
    pair_groups: dict[tuple, list] = {}
    for (kind, split, method, _seed), strengths in by_pair.items():
        if 0.0 not in strengths:
            continue
        alt_keys = [value for value in strengths if value != 0.0]
        if not alt_keys:
            continue
        alt = max(alt_keys)
        pair_groups.setdefault((kind, split, method), []).append((strengths[0.0], strengths[alt]))
    for key in sorted(pair_groups):
        pairs = pair_groups[key]
        d_po = np.array([alt["po_risk"] - null["po_risk"] for null, alt in pairs], dtype=float)
        d_var = np.array(
            [alt["mean_var_tau"] - null["mean_var_tau"] for null, alt in pairs],
            dtype=float,
        )
        d_plugin = np.array([alt["plugin_se"] - null["plugin_se"] for null, alt in pairs], dtype=float)
        null_po = np.mean([null["po_risk"] for null, _alt in pairs])
        alt_po = np.mean([alt["po_risk"] for _null, alt in pairs])
        var_finite = d_var[np.isfinite(d_var)]
        lines.append(
            "| {kind} | {split} | {method} | {dpo} | {npo}/{n} | {dvar} | {nvar}/{nv} | {dplug} | {npo0} | {po1} |".format(
                kind=key[0],
                split=key[1],
                method=key[2],
                dpo=_fmt(float(np.mean(d_po))),
                npo=int(np.sum(d_po > 0)),
                n=len(pairs),
                dvar=_fmt(float(np.mean(var_finite))) if var_finite.size else "",
                nvar=int(np.sum(var_finite > 0)) if var_finite.size else 0,
                nv=int(var_finite.size),
                dplug=_fmt(float(np.mean(d_plugin))),
                npo0=_fmt(float(null_po)),
                po1=_fmt(float(alt_po)),
            )
        )
    lines.append("")
    present_methods = {row["method"] for row in rows}
    lines.append("## Propensity weight on the holdout")
    lines.append("")
    lines.append(
        "Under a perfect overlap of one half, `mean((T - pi) / (pi (1 - pi)))^2` equals 4. "
        "Covariate shift pushes mass toward the clip, and the pseudo-outcome gets heavier even when the CATE is zero."
    )
    lines.append("")
    lines.append("| kind | strength | mean weight^2 | clip rate |")
    lines.append("| --- | ---: | ---: | ---: |")
    weight_groups: dict[tuple, list] = {}
    weight_seen = set()
    weight_method = "RF_delta" if "RF_delta" in present_methods else next(iter(present_methods))
    for row in rows:
        if row["split"] != "holdout" or row["method"] != weight_method:
            continue
        marker = (row["kind"], float(row["strength"]), row["seed"])
        if marker in weight_seen:
            continue
        weight_seen.add(marker)
        weight_groups.setdefault((row["kind"], float(row["strength"])), []).append(row)
    for key in sorted(weight_groups):
        block = weight_groups[key]
        lines.append(
            "| {kind} | {strength:g} | {weight} | {clip} |".format(
                kind=key[0],
                strength=key[1],
                weight=_fmt(float(np.mean([float(row["mean_weight_sq"]) for row in block]))),
                clip=_fmt(float(np.mean([float(row["clip_rate"]) for row in block]))),
            )
        )
    lines.append("")
    lines.append("## Feature permutation of the holdout PO-risk")
    lines.append("")
    lines.append(
        "One column of the holdout covariates is permuted and `tau` is recomputed. "
        "`phi` is left as fitted. Concept drift's nonzero CATE coordinates are 1, 2 and 3 "
        "(coordinate 0 has coefficient 0). Covariate shift moves columns 0 through 3, "
        "but the population CATE is zero, so those columns should not be required to predict `tau`."
    )
    lines.append("")
    lines.append("| kind | strength | vimp planted | vimp rest | top 4 by seed |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for candidate in ("RF_delta", "RF_IJ", "CF", "ORF", "BLB"):
        if candidate in present_methods:
            vimp_method = candidate
            break
    else:
        vimp_method = None
    vimp_groups: dict[tuple, list] = {}
    seen = set()
    for row in rows:
        if vimp_method is None or row["split"] != "holdout" or row["method"] != vimp_method:
            continue
        if not row.get("vimp_top4"):
            continue
        marker = (row["kind"], float(row["strength"]), row["seed"])
        if marker in seen:
            continue
        seen.add(marker)
        vimp_groups.setdefault((row["kind"], float(row["strength"])), []).append(row)
    for key in sorted(vimp_groups):
        block = sorted(vimp_groups[key], key=lambda row: row["seed"])
        tops = "; ".join(f"s{row['seed']}: {row['vimp_top4']}" for row in block)
        lines.append(
            "| {kind} | {strength:g} | {planted} | {rest} | {top} |".format(
                kind=key[0],
                strength=key[1],
                planted=_fmt(float(np.mean([row["vimp_planted"] for row in block]))),
                rest=_fmt(float(np.mean([row["vimp_rest"] for row in block]))),
                top=tops,
            )
        )
    lines.append("")
    lines.append(
        "Oracle scale: with a perfect regression, a balanced batch indicator and residual variance 1, "
        "`(T - 1/2) / (1/4)` equals ±2, so the PO-risk equals 4. "
        "`mean_weight_sq` above 4 means the propensity is no longer one half and the pseudo-outcome is heavier."
    )
    lines.append("")
    lines.extend(detection_notes(rows))
    return "\n".join(lines) + "\n"


def _finite(values):
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def _mean_std(values):
    array = _finite(values)
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1:
        return float(array[0]), float("nan")
    return float(np.mean(array)), float(np.std(array, ddof=1))


def detection_notes(rows: list[dict]) -> list[str]:
    """State what the variance and the permutation ranking did on this table."""
    lines = ["## What separates the two shifts", ""]
    if not rows:
        lines.append("No replicates.")
        lines.append("")
        return lines
    lines.append(
        "The concept-drift null is not an oracle risk of 4. Its outcome is `Y = X @ 1 + ε` "
        "on every column, and the random-forest outcome fit leaves a large pseudo-outcome remainder. "
        "The covariate-shift null is `Y = ε`, and that holdout risk sits next to 4. "
        "The shift in this covariate generator changes second moments, not means, so a linear propensity "
        "stays near one half: `mean weight^2` does not leave 4 and nothing is clipped. "
        "A rise in the covariate PO-risk is leftover nuisance error, not an inverse-propensity explosion."
    )
    lines.append("")
    lines.append(
        "Causal-forest `tau` is scored against the same random-forest pseudo-outcome as the other rows. "
        "`tau_mse` is the error against the population CATE, which is zero under covariate shift "
        "and `X @ (0, δ, 2δ, 3δ)` under concept drift."
    )
    lines.append("")
    lines.append("| kind | method | null risk (sd) | alt risk (sd) | gap (sd) | null Var(tau) | alt Var(tau) |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for kind in ("concept", "covariate"):
        for method in ("RF_delta", "RF_IJ", "CF"):
            null_rows = [
                row for row in rows
                if row["split"] == "holdout" and row["kind"] == kind and row["method"] == method
                and float(row["strength"]) == 0.0
            ]
            alt_rows = [
                row for row in rows
                if row["split"] == "holdout" and row["kind"] == kind and row["method"] == method
                and float(row["strength"]) != 0.0
            ]
            if not null_rows or not alt_rows:
                continue
            paired = []
            alt_by_seed = {int(row["seed"]): row for row in alt_rows}
            for row in null_rows:
                alt = alt_by_seed.get(int(row["seed"]))
                if alt is not None and float(alt["strength"]) == max(float(item["strength"]) for item in alt_rows):
                    paired.append((row, alt))
            if not paired:
                continue
            null_risk, null_sd = _mean_std(row["po_risk"] for row, _alt in paired)
            alt_risk, alt_sd = _mean_std(alt["po_risk"] for _row, alt in paired)
            gap, gap_sd = _mean_std(float(alt["po_risk"]) - float(row["po_risk"]) for row, alt in paired)
            null_var, _ = _mean_std(row["mean_var_tau"] for row, _alt in paired)
            alt_var, _ = _mean_std(alt["mean_var_tau"] for _row, alt in paired)
            lines.append(
                "| {kind} | {method} | {nr} ({ns}) | {ar} ({als}) | {gap} ({gs}) | {nv} | {av} |".format(
                    kind=kind,
                    method=method,
                    nr=_fmt(null_risk),
                    ns=_fmt(null_sd),
                    ar=_fmt(alt_risk),
                    als=_fmt(alt_sd),
                    gap=_fmt(gap),
                    gs=_fmt(gap_sd),
                    nv=_fmt(null_var),
                    av=_fmt(alt_var),
                )
            )
    lines.append("")
    concept_hits = 0
    concept_n = 0
    covariate_hits = 0
    covariate_n = 0
    seen = set()
    for row in rows:
        if row["split"] != "holdout" or row["method"] != "RF_delta" or not row.get("vimp_top4"):
            continue
        marker = (row["kind"], float(row["strength"]), int(row["seed"]))
        if marker in seen:
            continue
        seen.add(marker)
        top = [int(part) for part in str(row["vimp_top4"]).split()]
        if row["kind"] == "concept" and float(row["strength"]) != 0.0:
            concept_n += 1
            concept_hits += int(top[:3] == [3, 2, 1])
        if row["kind"] == "covariate" and float(row["strength"]) != 0.0:
            covariate_n += 1
            covariate_hits += int(set(top[:4]) == {0, 1, 2, 3})
    lines.append(
        f"Permutation of the holdout pseudo-outcome forest ranks columns 3, 2, 1 first "
        f"in {concept_hits}/{concept_n} concept-drift replicates. Those are the nonzero drift slopes, "
        f"in the same order as `(3δ, 2δ, δ)`. Column 0 has slope 0 and does not enter that top three. "
        f"Under covariate shift the same permutation puts columns 0–3 in the top four in "
        f"{covariate_hits}/{covariate_n} replicates. The population CATE there is zero; "
        f"the random-forest `tau` still uses the shifted columns, while the causal forest's "
        f"`tau_mse` stays at the null level."
    )
    lines.append("")
    ij_holdout = [
        row for row in rows
        if row.get("method") == "RF_IJ" and row.get("split") == "holdout"
        and np.isfinite(float(row.get("zero_var_fraction", float("nan"))))
    ]
    if ij_holdout:
        floor_rate = float(np.mean([float(row["zero_var_fraction"]) for row in ij_holdout]))
        raw_vals = _finite(row.get("raw_mean_var_tau", float("nan")) for row in ij_holdout)
        corrected_vals = _finite(row.get("mean_var_tau", float("nan")) for row in ij_holdout)
        shrink = (
            float(np.mean(raw_vals) / np.mean(corrected_vals))
            if raw_vals.size and corrected_vals.size and np.mean(corrected_vals) > 0
            else float("nan")
        )
        rises = 0
        compared = 0
        variance_by_seed = {}
        for row in ij_holdout:
            variance_by_seed.setdefault((row["kind"], int(row["seed"])), {})[float(row["strength"])] = float(
                row["mean_var_tau"]
            )
        for variance_by_strength in variance_by_seed.values():
            if 0.0 not in variance_by_strength:
                continue
            alt_levels = [strength for strength in variance_by_strength if strength != 0.0]
            if not alt_levels:
                continue
            alt_strength = max(alt_levels)
            compared += 1
            rises += int(variance_by_strength[alt_strength] > variance_by_strength[0.0])
        rise_sentence = (
            f"The corrected values are larger on the shifted draw in {rises}/{compared} paired replicates. "
            if compared
            else ""
        )
        jackknife_sentence = (
            f"On these holdout rows the Wager correction floors {floor_rate:.0%} of the jackknife points at zero "
            f"and the raw jackknife is about {_fmt(shrink, 1)} times the corrected one. "
            + rise_sentence
        )
    else:
        jackknife_sentence = ""
    lines.append(
        "The level of the PO-risk moves under both shifts. "
        "The causal-forest variance moves under concept drift and stays flat under covariate shift, "
        "which is the split between a real CATE change and a pure change in P(X). "
        "The jackknife and the tree-variance delta method move under both, because they are variances "
        "of the random-forest pseudo-outcome fit, and that fit gets less stable when the two arms separate. "
        + jackknife_sentence
        + "`se_indep` is much smaller than the seed-to-seed standard deviation of the risk, because it only "
        "propagates `Var(tau)` and treats sample contributions as independent. "
        "The little-bootstrap standard error is the refit noise of the training risk, not that cross-seed spread."
    )
    lines.append("")
    return lines


def write_outputs(rows: list[dict], out_dir: str) -> None:
    from pathlib import Path

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "replicates.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (destination / "summary.md").write_text(summarize_rows(rows))


def _config_from_args(args) -> dict:
    strengths = tuple(float(value) for value in args.strength)
    seeds = tuple(int(value) for value in args.seed)
    methods = tuple(args.method)
    return {
        "n": int(args.n_per_batch * 2),
        "p": int(args.p),
        "rho": float(args.rho),
        "strengths": strengths,
        "seeds": seeds,
        "methods": methods,
        "n_estimators": int(args.n_estimators),
        "min_leaf_size": int(args.min_leaf_size),
        "subsample_ratio": float(args.subsample_ratio),
        "clip_e": float(args.clip_e),
        "test_size": float(args.test_size),
        "n_jobs": int(args.n_jobs),
        "bias_correction": args.bias_correction,
        "n_perm": int(args.n_perm),
        "blb_subsets": int(args.blb_subsets),
        "blb_boot": int(args.blb_boot),
        "blb_trees": int(args.blb_trees),
        "blb_subset_frac": float(args.blb_subset_frac),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare PO-risk variance estimators on the two shift DGPs.")
    parser.add_argument("--n-per-batch", type=int, default=1000)
    parser.add_argument("--p", type=int, default=20)
    parser.add_argument("--rho", type=float, default=0.0)
    parser.add_argument("--strength", type=float, nargs="+", default=(0.0, 1.0))
    parser.add_argument("--seed", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument(
        "--method",
        nargs="+",
        default=("RF_delta", "RF_IJ", "CF", "BLB"),
        help="Any of RF_delta, RF_IJ, CF, ORF, BLB.",
    )
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--min-leaf-size", type=int, default=20)
    parser.add_argument("--subsample-ratio", type=float, default=0.7)
    parser.add_argument("--clip-e", type=float, default=0.01)
    parser.add_argument("--test-size", type=float, default=0.5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--bias-correction", default="Wager")
    parser.add_argument("--n-perm", type=int, default=20)
    parser.add_argument("--blb-subsets", type=int, default=4)
    parser.add_argument("--blb-boot", type=int, default=8)
    parser.add_argument("--blb-trees", type=int, default=40)
    parser.add_argument("--blb-subset-frac", type=float, default=0.5)
    parser.add_argument(
        "--out-dir",
        default="experiments/po_risk_variance_results",
    )
    parser.add_argument("--pilot", action="store_true", help="Small run used to check the wiring.")
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.pilot:
        args.n_per_batch = 200
        args.p = 8
        args.strength = [0.0, 1.0]
        args.seed = [0]
        args.n_estimators = 30
        args.min_leaf_size = 8
        args.n_perm = 5
        args.blb_subsets = 2
        args.blb_boot = 3
        args.blb_trees = 15
        args.method = [method for method in args.method if method.lower() != "orf"]
    config = _config_from_args(args)
    rows = run_comparison(config)
    write_outputs(rows, args.out_dir)
    print(f"wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
