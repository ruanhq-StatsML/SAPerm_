"""PermuCATE feature importance for a two-batch FSDS comparison.

Nuisance models are entries of ``model_registry_class``. The outcome
regression, the propensity score, the pseudo-outcome regression, and the
feature-wise conditional model are looked up by name. The score for a
coordinate is the rise in doubly robust pseudo-outcome risk after the
conditional residual of that coordinate is permuted.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ddpm_cf_attribution.model_registry_class import default_model_registry


OUTCOME_MODELS = (
    "rf_regressor",
    "xgb_regressor",
    "ridge_regressor",
    "lm_regressor",
    "mlp_regressor",
)


def fsds_registry(**kwargs):
    """Registry used by the biasfinder application."""
    settings = dict(
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
        warn_xgb_labels=True,
        positive_class=1,
    )
    settings.update(kwargs)
    return default_model_registry(**settings)


def _as_1d(y):
    return np.asarray(y, dtype=float).ravel()


def po_risk(tau, y, w, mu0, mu1, pi, clip=1e-3) -> float:
    pi_clip = np.clip(pi, clip, 1.0 - clip)
    pseudo = (y - np.where(w == 1, mu1, mu0)) * (w - pi_clip) / (pi_clip * (1.0 - pi_clip)) + (mu1 - mu0)
    return float(np.mean((pseudo - tau) ** 2))


def infer_feature_type(x):
    kinds = ["continuous"] * x.shape[1]
    for j in range(x.shape[1]):
        n_level = np.unique(x[:, j]).size
        if n_level == 2:
            kinds[j] = "binary"
        elif n_level <= 5:
            kinds[j] = "categorical"
    return kinds


def permuCATE_fit_nuisance(
    x, y, w,
    model_registry,
    model_m="rf_regressor",
    model_e="logistic_classifier",
    model_tau="rf_regressor",
    model_nu="rf_regressor",
    model_nu_bi="logistic_classifier",
    model_nu_multi="multinomial_classifier",
    clip_e=1e-2,
    seed=2026,
):
    x = np.asarray(x, dtype=float).copy()
    y = _as_1d(y)
    w = _as_1d(w).astype(int)
    outcome = model_registry[model_m]
    propensity = model_registry[model_e]
    tau_spec = model_registry[model_tau]
    mu0_fit = outcome["fit"](x[w == 0], y[w == 0], seed=seed)
    mu1_fit = outcome["fit"](x[w == 1], y[w == 1], seed=seed + 1)
    mu0 = outcome["predict"](mu0_fit, x)
    mu1 = outcome["predict"](mu1_fit, x)
    e_fit = propensity["fit"](x, w, seed=seed + 2)
    pi = np.clip(propensity["predict"](e_fit, x), clip_e, 1.0 - clip_e)
    pseudo = mu1 - mu0 + (y - np.where(w == 1, mu1, mu0)) * (w - pi) / (pi * (1.0 - pi))
    tau_fit = tau_spec["fit"](x, pseudo, seed=seed + 3)
    kinds = infer_feature_type(x)
    nu_model = {}
    for j in range(x.shape[1]):
        if kinds[j] == "continuous":
            spec = model_registry[model_nu]
        else:
            x[:, j] = LabelEncoder().fit_transform(x[:, j])
            spec = model_registry[model_nu_bi if kinds[j] == "binary" else model_nu_multi]
        nu_fit = spec["fit"](np.delete(x, j, axis=1), x[:, j], seed=seed + 100 + j)
        nu_model[j] = {"model_nu": spec, "model_nu_fit": nu_fit}
    return {
        "model_outcome0": outcome,
        "model_outcome1": outcome,
        "model_propensity_score": propensity,
        "model_tau": tau_spec,
        "mu0_fit": mu0_fit,
        "mu1_fit": mu1_fit,
        "model_e_fit": e_fit,
        "model_tau_fit": tau_fit,
        "nu_model": nu_model,
    }


def permuCATE_vimp(
    x, y, w,
    model_registry,
    model_m="rf_regressor",
    model_e="logistic_classifier",
    model_tau="rf_regressor",
    model_nu="rf_regressor",
    model_nu_bi="logistic_classifier",
    model_nu_multi="multinomial_classifier",
    n_perm=50,
    test_size=0.5,
    seed=0,
    clip_e=0.01,
):
    """FSDS score for one registry specification. Returns the mean drop and its sd."""
    x = np.asarray(x, dtype=float)
    y = _as_1d(y)
    w = _as_1d(w).astype(int)
    x_train, x_test, y_train, y_test, w_train, w_test = train_test_split(
        x, y, w, test_size=test_size, stratify=w, random_state=seed,
    )
    outputs = permuCATE_fit_nuisance(
        x_train, y_train, w_train, model_registry,
        model_m=model_m, model_e=model_e, model_tau=model_tau, model_nu=model_nu,
        model_nu_bi=model_nu_bi, model_nu_multi=model_nu_multi,
        clip_e=clip_e, seed=seed,
    )
    tau = np.asarray(outputs["model_tau"]["predict"](outputs["model_tau_fit"], x_test), dtype=float).ravel()
    mu0 = np.asarray(outputs["model_outcome0"]["predict"](outputs["mu0_fit"], x_test), dtype=float).ravel()
    mu1 = np.asarray(outputs["model_outcome1"]["predict"](outputs["mu1_fit"], x_test), dtype=float).ravel()
    pi = np.clip(
        np.asarray(outputs["model_propensity_score"]["predict"](outputs["model_e_fit"], x_test), dtype=float).ravel(),
        clip_e, 1.0 - clip_e,
    )
    base = po_risk(tau, y_test, w_test, mu0, mu1, pi, clip_e)
    p = x.shape[1]
    vimp = np.zeros(p)
    sd = np.zeros(p)
    median = np.zeros(p)
    for j in range(p):
        nu = outputs["nu_model"][j]
        nu_hat = np.asarray(
            nu["model_nu"]["predict"](nu["model_nu_fit"], np.delete(x_test, j, axis=1)),
            dtype=float,
        ).ravel()
        residual = x_test[:, j] - nu_hat
        rng = np.random.default_rng(seed + j)
        drops = np.zeros(n_perm)
        for k in range(n_perm):
            x_perm = x_test.copy()
            x_perm[:, j] = nu_hat + rng.permutation(residual)
            tau_perm = np.asarray(outputs["model_tau"]["predict"](outputs["model_tau_fit"], x_perm), dtype=float).ravel()
            drops[k] = po_risk(tau_perm, y_test, w_test, mu0, mu1, pi, clip_e) - base
        vimp[j] = float(drops.mean())
        sd[j] = float(drops.std())
        median[j] = float(np.median(drops))
    return {
        "name": model_m,
        "vimp": vimp,
        "sd": sd,
        "median": median,
        "po_risk": base,
        "tau": tau,
        "mu0": mu0,
        "mu1": mu1,
        "pi": pi,
        "y": y_test,
        "w": w_test,
    }
