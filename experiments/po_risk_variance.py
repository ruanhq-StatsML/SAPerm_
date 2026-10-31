"""PO-risk variance, in the form written for the two shift generators.

``Var((phi - tau)^2)`` is ``4 * (phi - tau)^2 * Var(tau)``.
``RF_delta`` uses the tree-prediction variance. ``RF_IJ`` subtracts
``Var_b(T_b) / B`` from the infinitesimal jackknife, which is the correction
written next to the Wager, Hastie, and Efron citation.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

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
    "model_m": "rf_regressor",
    "model_e": "logistic_classifier",
    "model_tau": "rf_regressor",
    "model_nu": "rf_regressor",
    "model_nu_bi": "logistic_classifier",
    "model_nu_multi": "multinomial_classifier",
}


def generate_concept_drift_dgp(n=2000, p=20, delta_beta=0.0, rho=0.0, seed=None):
    """Concept drift: same covariate distribution, CATE changes after n/2."""
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
    """Covariate shift: outcome mechanism fixed, P(X) differs on the first half."""
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


def _as_1d(values):
    return np.asarray(values).reshape(-1)


def make_registry(ntree=150, n_jobs=-1):
    """The two registry entries the PO-risk variance calls."""

    def rf_fit(X, Y, seed=0):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float).ravel()
        n, p = X.shape
        return RandomForestRegressor(
            n_estimators=int(ntree),
            max_features=max(1, int(round(p / 3.0))),
            min_samples_leaf=max(1, int(round(np.sqrt(n) / 2.0))),
            random_state=None if seed is None else int(seed),
            n_jobs=n_jobs,
        ).fit(X, Y)

    def rf_predict(fit, X_new):
        return np.asarray(fit.predict(np.asarray(X_new, dtype=float)), dtype=float).ravel()

    def logit_fit(X, Y, seed=0):
        return LogisticRegression(
            solver="lbfgs",
            max_iter=500,
            random_state=None if seed is None else int(seed),
        ).fit(np.asarray(X, dtype=float), np.asarray(Y).ravel())

    def logit_predict(fit, X_new):
        proba = np.asarray(fit.predict_proba(np.asarray(X_new, dtype=float)), dtype=float)
        classes = [int(c) for c in np.asarray(fit.classes_).tolist()]
        column = classes.index(1) if 1 in classes else int(np.argmax(classes))
        return proba[:, column]

    rf = {"name": "rf_regressor", "fit": rf_fit, "predict": rf_predict}
    logit = {"name": "logistic_classifier", "fit": logit_fit, "predict": logit_predict}
    return {
        "rf_regressor": rf,
        "logistic_classifier": logit,
        "multinomial_classifier": logit,
    }


MODEL_REGISTRY = make_registry(ntree=DEFAULT_CONFIGS["n_trees"], n_jobs=DEFAULT_CONFIGS["n_jobs"])


def infer_feature_type(X):
    p = X.shape[1]
    feature_types = ["continuous"] * p
    for j in range(p):
        n_level = np.unique(X[:, j]).size
        if n_level == 2:
            feature_types[j] = "binary"
        elif n_level <= 5:
            feature_types[j] = "categorical"
    return feature_types


def permuCATE_fit_nuisance(
    X, Y, W,
    model_registry=None,
    model_m="rf_regressor",
    model_e="logistic_classifier",
    model_tau="rf_regressor",
    model_nu="rf_regressor",
    model_nu_bi="logistic_classifier",
    model_nu_multi="multinomial_classifier",
    clip_e=1e-2,
    seed=2026,
):
    if model_registry is None:
        model_registry = MODEL_REGISTRY
    X = np.asarray(X, dtype=float).copy()
    Y = _as_1d(Y)
    W = _as_1d(W).astype(int)
    n, p = X.shape
    model_outcome0 = model_registry[model_m]
    model_outcome1 = model_registry[model_m]
    model_propensity_score = model_registry[model_e]
    model_tau_spec = model_registry[model_tau]
    X_ctr, X_trt = X[W == 0], X[W == 1]
    Y_ctr, Y_trt = Y[W == 0], Y[W == 1]
    mu0_fit = model_outcome0["fit"](X_ctr, Y_ctr, seed=seed)
    mu1_fit = model_outcome1["fit"](X_trt, Y_trt, seed=seed)
    mu0_est = model_outcome0["predict"](mu0_fit, X)
    mu1_est = model_outcome1["predict"](mu1_fit, X)
    model_e_fit = model_propensity_score["fit"](X, W, seed=seed + 2)
    pi_est = np.clip(model_propensity_score["predict"](model_e_fit, X), clip_e, 1 - clip_e)
    pseudo_outcome = mu1_est - mu0_est + (
        Y - np.where(W == 1, mu1_est, mu0_est)
    ) * (W - pi_est) / (pi_est * (1.0 - pi_est))
    model_tau_fit = model_tau_spec["fit"](X, pseudo_outcome, seed=seed + 3)
    tau_est = model_tau_spec["predict"](model_tau_fit, X)
    feature_types = infer_feature_type(X)
    nu_model = {}
    for j in range(p):
        if feature_types[j] == "continuous":
            model_nu_spec = model_registry[model_nu]
        else:
            X[:, j] = LabelEncoder().fit_transform(X[:, j])
            model_nu_spec = model_registry[
                model_nu_bi if feature_types[j] == "binary" else model_nu_multi
            ]
        model_nu_fit = model_nu_spec["fit"](
            np.delete(X, j, axis=1), X[:, j], seed=seed + j + 100
        )
        nu_model[j] = {"model_nu": model_nu_spec, "model_nu_fit": model_nu_fit}
    return {
        "model_outcome0": model_outcome0,
        "model_outcome1": model_outcome1,
        "model_propensity_score": model_propensity_score,
        "model_tau": model_tau_spec,
        "mu0_fit": mu0_fit,
        "mu1_fit": mu1_fit,
        "model_e_fit": model_e_fit,
        "model_tau_fit": model_tau_fit,
        "mu0_est": mu0_est,
        "mu1_est": mu1_est,
        "pi_est": pi_est,
        "tau_est": tau_est,
        "pseudo_outcome": pseudo_outcome,
        "nu_model": nu_model,
    }


def IJ_estimator(model_outputs, inbag_times, bias_correction="Wager"):
    """Infinitesimal jackknife. The Wager branch subtracts ``Var(T_b) / B``."""
    B = model_outputs.shape[1]
    T_centered = model_outputs - np.mean(model_outputs, axis=1, keepdims=True)
    N_centered = inbag_times - np.mean(inbag_times, axis=1, keepdims=True)
    cov_est = (T_centered @ N_centered.T) / B
    IJ_var = np.sum(cov_est ** 2, axis=1)
    if bias_correction == "Wager":
        naive_var = np.var(model_outputs, ddof=1, axis=1) / B
        IJ_var = np.maximum(IJ_var - naive_var, 0.0)
    return IJ_var


def permuCATE_PO(
    X, Y, W,
    model_registry=None,
    model_m="rf_regressor",
    model_e="logistic_classifier",
    model_tau="rf_regressor",
    model_nu="rf_regressor",
    model_nu_bi="logistic_classifier",
    model_nu_multi="multinomial_classifier",
    n_perm=100,
    seed=0,
    clip_e=0.01,
    normalize=False,
):
    if model_registry is None:
        model_registry = MODEL_REGISTRY
    X = np.asarray(X)
    Y = _as_1d(Y)
    W = _as_1d(W).astype(int)
    outputs = permuCATE_fit_nuisance(
        X, Y, W,
        model_registry=model_registry,
        model_m=model_m, model_e=model_e, model_tau=model_tau,
        model_nu=model_nu, model_nu_bi=model_nu_bi, model_nu_multi=model_nu_multi,
        clip_e=clip_e, seed=seed,
    )
    tau_est = outputs["model_tau"]["predict"](outputs["model_tau_fit"], X_new=X)
    mu0 = outputs["model_outcome0"]["predict"](outputs["mu0_fit"], X_new=X)
    mu1 = outputs["model_outcome1"]["predict"](outputs["mu1_fit"], X_new=X)
    pi = np.clip(
        outputs["model_propensity_score"]["predict"](outputs["model_e_fit"], X_new=X),
        clip_e, 1 - clip_e,
    )
    pi_clip = np.clip(pi, clip_e, 1 - clip_e)
    pseudo = (Y - np.where(W == 1, mu1, mu0)) * (W - pi_clip) / (pi_clip * (1 - pi_clip)) + mu1 - mu0
    po_risk = np.mean((tau_est - pseudo) ** 2)
    return tau_est, pseudo, po_risk


def _tree_matrix(forest, X):
    columns = [np.asarray(tree.predict(X), dtype=float).ravel() for tree in forest.estimators_]
    return np.column_stack(columns)


def _inbag_indicator(forest):
    """Presence in the bootstrap draw. A repeated row stays 1."""
    samples = forest.estimators_samples_
    n_trees = len(samples)
    n = int(getattr(forest, "n_samples_in_", samples[0].shape[0]))
    inbag_times = np.zeros((n, n_trees))
    for b, idx in enumerate(samples):
        inbag_times[np.asarray(idx, dtype=int), b] = 1
    return inbag_times


def variance_estimator_po_risk(
    X, Y, T,
    cate_model="PO",
    **kwargs,
):
    """Sample-wise delta-method variance ``4 * (phi - tau)^2 * Var(tau)``."""
    clip_e = kwargs.get("clip_e", 0.01)
    n_estimators = kwargs.get("n_estimators", 150)
    model_registry = kwargs.get("model_registry", MODEL_REGISTRY)
    model_m = kwargs.get("model_m", "rf_regressor")
    model_e = kwargs.get("model_e", "logistic_classifier")
    model_tau = kwargs.get("model_tau", "rf_regressor")
    model_nu = kwargs.get("model_nu", "rf_regressor")
    model_nu_bi = kwargs.get("model_nu_bi", "logistic_classifier")
    model_nu_multi = kwargs.get("model_nu_multi", "multinomial_classifier")
    bias_correction = kwargs.get("bias_correction", "Wager")
    if str(bias_correction).lower() == "wager":
        bias_correction = "Wager"
    seed = kwargs.get("seed", 2000)
    min_leaf_size = kwargs.get("min_leaf_size", 20)
    subsample_ratio = kwargs.get("subsample_ratio", 0.7)
    random_state = kwargs.get("random_state", seed)
    n_jobs = kwargs.get("n_jobs", -1)
    X = np.asarray(X, dtype=float)
    Y = _as_1d(Y)
    W = _as_1d(T).astype(int)
    nuisance = dict(
        model_registry=model_registry,
        model_m=model_m, model_e=model_e, model_tau=model_tau,
        model_nu=model_nu, model_nu_bi=model_nu_bi, model_nu_multi=model_nu_multi,
        clip_e=clip_e, seed=seed,
    )
    if cate_model in ("RF_delta", "RF_IJ", "PO"):
        outputs = permuCATE_fit_nuisance(X, Y, W, **nuisance)
        po_model = outputs["model_tau_fit"]
        tree_preds = _tree_matrix(po_model, X)
        if cate_model == "RF_delta":
            var_tau = np.var(tree_preds, axis=1, ddof=1)
        else:
            var_tau = IJ_estimator(tree_preds, _inbag_indicator(po_model), bias_correction=bias_correction)
    elif cate_model == "CF":
        from econml.dml import CausalForestDML

        subforest = next(size for size in (4, 3, 2, 1) if int(n_estimators) % size == 0)
        est = CausalForestDML(
            n_estimators=n_estimators,
            min_samples_leaf=min_leaf_size,
            subforest_size=subforest,
            discrete_treatment=True,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        est.fit(Y, W, X=X)
        var_tau = np.asarray(est.effect_inference(X).stderr, dtype=float).ravel() ** 2
    elif cate_model == "orf":
        from econml.orf import DMLOrthoForest

        est = DMLOrthoForest(
            n_trees=n_estimators,
            min_leaf_size=min_leaf_size,
            subsample_ratio=subsample_ratio,
            discrete_treatment=True,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=0,
        )
        est.fit(Y, W, X=X, inference="blb")
        var_tau = np.asarray(est.effect_inference(X, T0=0, T1=1).stderr, dtype=float).ravel() ** 2
    else:
        raise ValueError(cate_model)
    tau_est, pseudo, po_risk = permuCATE_PO(
        X, Y, W,
        model_registry=model_registry,
        model_m=model_m, model_e=model_e, model_tau=model_tau,
        model_nu=model_nu, model_nu_bi=model_nu_bi, model_nu_multi=model_nu_multi,
        clip_e=clip_e, seed=seed,
    )
    del po_risk
    r = (pseudo - tau_est) ** 2
    var_estimator = 4 * r * var_tau
    return var_estimator
