"""PO-risk variance, in the form written for the two shift generators.

``Var((phi - tau)^2)`` is ``4 * (phi - tau)^2 * Var(tau)``.
``RF_delta`` uses the tree-prediction variance. ``RF_IJ`` subtracts
``Var_b(T_b) / B`` from the infinitesimal jackknife, which is the correction
written next to the Wager, Hastie, and Efron citation.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from experiments.dgps import banded_toeplitz_corr
from model_registry_class import ModelRegistry

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


def po_risk(tau, Y, T, mu0, mu1, pi, clip=1e-3):
    pi_clip = np.clip(pi, clip, 1 - clip)
    pseudo = (Y - np.where(T == 1, mu1, mu0)) * (T - pi_clip) / (
        pi_clip * (1 - pi_clip)
    ) + (mu1 - mu0)
    return np.mean((pseudo - tau) ** 2)


def make_registry(ntree=150, n_jobs=1):
    """The same factory, with the tree count the tests pass in."""
    nthread = 1 if n_jobs is None or int(n_jobs) <= 1 else int(n_jobs)
    return ModelRegistry(
        ntree=int(ntree),
        ridge_alpha=0.25,
        nthread=nthread,
        maxit=200,
        max_depth=5,
        gamma=0.25,
        eta=0.15,
        mlp_hidden_size=4,
        mlp_decay=1e-5,
        mlp_max_iter=500,
        mlp_trace=False,
        mlp_max_coef_reg=10000,
        mlp_max_coef_clf=10000,
        warn_xgb_labels=True,
        positive_class=1,
    ).as_r_style_dict()


DEFAULT_MODEL_FACTORY = ModelRegistry(
    ntree=150,
    ridge_alpha=0.25,
    nthread=1,
    maxit=200,
    max_depth=5,
    gamma=0.25,
    eta=0.15,
    mlp_hidden_size=4,
    mlp_decay=1e-5,
    mlp_max_iter=500,
    mlp_trace=False,
    mlp_max_coef_reg=10000,
    mlp_max_coef_clf=10000,
    warn_xgb_labels=True,
    positive_class=1,
)

MODEL_REGISTRY = DEFAULT_MODEL_FACTORY.as_r_style_dict()


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
    risk = np.mean((tau_est - pseudo) ** 2)
    return tau_est, pseudo, risk


def permuCATE_vimp_withPOModel(
    X, Y, W,
    model_registry=None,
    model_m="rf_regressor",
    model_e="logistic_classifier",
    model_tau="rf_regressor",
    model_nu="rf_regressor",
    model_nu_bi="logistic_classifier",
    model_nu_multi="multinomial_classifier",
    n_perm=100,
    test_size=0.5,
    seed=0,
    clip_e=0.01,
    normalize=False,
):
    """Holdout residual permutation. Each coordinate's score is the rise in PO-risk."""
    if model_registry is None:
        model_registry = MODEL_REGISTRY
    X = np.asarray(X)
    Y = _as_1d(Y)
    W = _as_1d(W).astype(int)
    X_train, X_test, W_train, W_test, Y_train, Y_test = train_test_split(
        X, W, Y, test_size=test_size, stratify=W, random_state=seed
    )
    n, p = X.shape
    outputs = permuCATE_fit_nuisance(
        X_train, Y_train, W_train,
        model_registry=model_registry,
        model_m=model_m, model_e=model_e, model_tau=model_tau,
        model_nu=model_nu, model_nu_bi=model_nu_bi, model_nu_multi=model_nu_multi,
        clip_e=clip_e, seed=seed,
    )
    tau_est = outputs["model_tau"]["predict"](outputs["model_tau_fit"], X_new=X_test)
    mu0_test = outputs["model_outcome0"]["predict"](outputs["mu0_fit"], X_new=X_test)
    mu1_test = outputs["model_outcome1"]["predict"](outputs["mu1_fit"], X_new=X_test)
    pi_test = np.clip(
        outputs["model_propensity_score"]["predict"](outputs["model_e_fit"], X_new=X_test),
        clip_e, 1 - clip_e,
    )
    po_risk_original = po_risk(tau_est, Y_test, W_test, mu0_test, mu1_test, pi_test)
    vimp = np.zeros(p, dtype=float)
    vimp_sd = np.zeros(p, dtype=float)
    vimp_median = np.zeros(p, dtype=float)
    for j in range(p):
        rng = np.random.default_rng(seed + j)
        nu_spec = outputs["nu_model"][j]["model_nu"]
        nu_fit = outputs["nu_model"][j]["model_nu_fit"]
        nu_hat = nu_spec["predict"](nu_fit, np.delete(X_test, j, axis=1)).ravel()
        residual = (X_test[:, j] - nu_hat).ravel()
        drops = np.zeros(n_perm, dtype=float)
        for k in range(n_perm):
            X_perm = X_test.copy()
            X_perm[:, j] = nu_hat + rng.permutation(residual)
            tau_perm = outputs["model_tau"]["predict"](outputs["model_tau_fit"], X_perm)
            drops[k] = po_risk(tau_perm, Y_test, W_test, mu0_test, mu1_test, pi_test) - po_risk_original
        vimp[j] = drops.mean()
        vimp_sd[j] = drops.std()
        vimp_median[j] = float(np.median(drops))
    if normalize:
        return vimp / np.sum(vimp)
    return vimp, vimp_sd, vimp_median


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


def rank_by_po_risk_variance(var_estimator):
    """Rank samples by the PO-risk variance. The second half is the drifted batch."""
    var_estimator = np.asarray(var_estimator, dtype=float).ravel()
    half = var_estimator.shape[0] // 2
    order = np.argsort(-var_estimator)
    return {
        "mean_first": float(var_estimator[:half].mean()),
        "mean_second": float(var_estimator[half:].mean()),
        "top_half_in_second": float(np.mean(order[:half] >= half)),
    }
