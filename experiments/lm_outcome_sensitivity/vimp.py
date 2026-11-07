"""CFPerm variable-importance tests used for the power comparison.

The three notions follow the CFPerm Python sources:

* LOCO refits the R-learner without each covariate and records the increase
  in R-risk (``LOCO_vimp_r_risk.py``, with the R-risk from
  ``CFPerm_alternative_utils.py``).
* GRF is the causal-forest split importance (``grf_vimp_causalForest.py``).
  Batch-label inference is not part of that importance, so the forest is fit
  with ``inference=False``.
* PermuCATE permutes each covariate's residual and records the increase in
  doubly robust pseudo-outcome risk (``permuCATE_vimp.py``).

``cfperm_feature_pvals`` then permutes the batch label and refits, which is
the decision rule in ``CFPerm_vimp_combined.py``. Feature-wise upper quantiles
use ``level_feature``, and the cross-feature threshold uses
``level_across_feature``. Those two numbers are the quantile levels: the R
helper ``aggregate_power`` with ``level_feature=0.01`` and
``level_across_feature=0.05`` is the same rule as ``0.99`` and ``0.95``.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.lm_outcome_sensitivity.registry import _as_1d, _design, _seed


def _crossfit_nuisance(x, y, w, model_e, model_m, registry, seed, n_splits, clip_e):
    x = _design(x)
    y = _as_1d(y)
    w = _as_1d(w).astype(int)
    n = x.shape[0]
    n_splits = int(min(n_splits, np.min(np.bincount(w))))
    n_splits = max(n_splits, 2)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=_seed(seed))
    propensity = registry[model_e]
    outcome = registry[model_m]
    m_hat = np.zeros(n, dtype=float)
    e_hat = np.zeros(n, dtype=float)
    for fold, (train, test) in enumerate(cv.split(x, w)):
        fitted_e = propensity["fit"](x[train], w[train], seed=seed + 17 * fold)
        e_hat[test] = propensity["predict"](fitted_e, x[test])
        fitted_m = outcome["fit"](x[train], y[train], seed=seed + 17 * fold + 1)
        m_hat[test] = outcome["predict"](fitted_m, x[test])
    e_hat = np.clip(e_hat, clip_e, 1.0 - clip_e)
    return m_hat, e_hat


def _fit_rlearner_tau(x, y_tilde, w_tilde, seed=0, clip_wtilde=1e-3):
    x = _design(x)
    y_tilde = _as_1d(y_tilde)
    w_tilde = _as_1d(w_tilde)
    mask = np.abs(w_tilde) > clip_wtilde
    finite = mask & np.isfinite(y_tilde) & np.isfinite(w_tilde)
    if int(finite.sum()) < max(30, x.shape[1] + 5):

        class _Zero:
            def predict(self, x_new):
                return np.zeros(_design(x_new).shape[0], dtype=float)

        return _Zero()
    z = y_tilde[finite] / w_tilde[finite]
    weights = w_tilde[finite] ** 2
    ok = np.isfinite(z) & np.isfinite(weights) & (weights > 0)
    if int(ok.sum()) < max(30, x.shape[1] + 5):

        class _Zero:
            def predict(self, x_new):
                return np.zeros(_design(x_new).shape[0], dtype=float)

        return _Zero()
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )
    model.fit(x[finite][ok], z[ok], ridge__sample_weight=weights[ok])
    return model


def r_risk(y_tilde, w_tilde, tau_hat) -> float:
    """Empirical R-risk ``mean((Y - m(X) - tau(X) * (W - e(X)))^2)``."""
    y_tilde = _as_1d(y_tilde)
    w_tilde = _as_1d(w_tilde)
    tau_hat = _as_1d(tau_hat)
    return float(np.mean((y_tilde - w_tilde * tau_hat) ** 2))


def vimp_loco_r_risk(
    x,
    y,
    w,
    model_m,
    model_e,
    registry,
    *,
    seed: int = 0,
    n_splits: int = 5,
    clip_e: float = 0.01,
    clip_wtilde: float = 1e-3,
) -> np.ndarray:
    x = _design(x)
    y = _as_1d(y)
    w = _as_1d(w).astype(int)
    p = x.shape[1]
    m_hat, e_hat = _crossfit_nuisance(
        x, y, w, model_e, model_m, registry, seed=seed, n_splits=n_splits, clip_e=clip_e
    )
    y_tilde = y - m_hat
    w_tilde = w.astype(float) - e_hat
    tau_full = _fit_rlearner_tau(x, y_tilde, w_tilde, seed=seed, clip_wtilde=clip_wtilde)
    risk_full = r_risk(y_tilde, w_tilde, tau_full.predict(x))
    importance = np.zeros(p, dtype=float)
    for j in range(p):
        x_minus = np.delete(x, j, axis=1)
        tau_j = _fit_rlearner_tau(
            x_minus, y_tilde, w_tilde, seed=seed + 17 * (j + 1), clip_wtilde=clip_wtilde
        )
        importance[j] = r_risk(y_tilde, w_tilde, tau_j.predict(x_minus)) - risk_full
    return importance


def _grf_estimators(model_outcome: str, model_psm: str, seed: int, mlp_hidden: int, mlp_max_iter: int, mlp_alpha: float, ridge_alpha: float):
    if model_outcome == "lm_regressor":
        model_y = LinearRegression()
    elif model_outcome == "mlp_regressor":
        model_y = MLPRegressor(
            hidden_layer_sizes=(int(mlp_hidden),),
            alpha=float(mlp_alpha),
            max_iter=int(mlp_max_iter),
            early_stopping=False,
            learning_rate_init=1e-2,
            tol=1e-3,
            random_state=_seed(seed),
        )
    else:
        model_y = Ridge(alpha=1.0)
    c_value = 1.0 if model_psm == "logistic_classifier" else 1.0 / float(ridge_alpha)
    model_t = LogisticRegression(
        C=c_value,
        solver="lbfgs",
        max_iter=400,
        random_state=_seed(seed),
    )
    return model_y, model_t


def cf_variable_importance(
    x,
    y,
    w,
    model_psm,
    model_outcome,
    *,
    seed: int = 0,
    n_estimators: int = 40,
    min_samples_leaf: int = 5,
    max_depth: int = 4,
    max_features: str | int | float = "auto",
    normalize: bool = True,
    mlp_hidden: int = 4,
    mlp_max_iter: int = 80,
    mlp_alpha: float = 1e-5,
    ridge_alpha: float = 0.25,
) -> np.ndarray:
    from econml.dml import CausalForestDML

    x = _design(x)
    y = _as_1d(y)
    w = _as_1d(w).astype(int)
    model_y, model_t = _grf_estimators(
        model_outcome,
        model_psm,
        seed,
        mlp_hidden=mlp_hidden,
        mlp_max_iter=mlp_max_iter,
        mlp_alpha=mlp_alpha,
        ridge_alpha=ridge_alpha,
    )
    forest = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        n_estimators=int(n_estimators),
        min_samples_leaf=int(min_samples_leaf),
        max_depth=max_depth,
        max_features=max_features,
        discrete_treatment=True,
        cv=3,
        honest=True,
        inference=False,
        n_jobs=1,
        random_state=_seed(seed),
        verbose=0,
    )
    forest.fit(Y=y, T=w, X=x)
    importance = np.asarray(forest.feature_importances_, dtype=float).reshape(-1)
    if importance.size < x.shape[1]:
        importance = np.pad(importance, (0, x.shape[1] - importance.size))
    importance = importance[: x.shape[1]]
    if not np.any(np.isfinite(importance)):
        return np.zeros(x.shape[1], dtype=float)
    importance = np.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)
    total = float(np.sum(importance))
    if normalize and np.isfinite(total) and total > 0:
        return importance / total
    return importance


def _feature_types(x) -> list[str]:
    kinds = []
    for j in range(x.shape[1]):
        n_level = np.unique(x[:, j]).size
        if n_level <= 2:
            kinds.append("binary")
        elif n_level < 5:
            kinds.append("categorical")
        else:
            kinds.append("continuous")
    return kinds


def po_risk(tau, y, t, mu0, mu1, pi, clip=1e-3) -> float:
    pi_clip = np.clip(_as_1d(pi), clip, 1.0 - clip)
    treatment = _as_1d(t).astype(float)
    response = _as_1d(y)
    mu0 = _as_1d(mu0)
    mu1 = _as_1d(mu1)
    tau = _as_1d(tau)
    residual = response - np.where(treatment == 1, mu1, mu0)
    pseudo = residual * (treatment - pi_clip) / (pi_clip * (1.0 - pi_clip)) + (mu1 - mu0)
    return float(np.mean((pseudo - tau) ** 2))


def permuCATE_vimp(
    x,
    y,
    w,
    model_m,
    model_e,
    registry,
    *,
    model_tau=None,
    model_nu=None,
    n_perm: int = 20,
    seed: int = 0,
    clip_e: float = 0.01,
) -> np.ndarray:
    x = _design(x)
    y = _as_1d(y)
    w = _as_1d(w).astype(int)
    if model_tau is None:
        model_tau = model_m
    if model_nu is None:
        model_nu = model_m
    x_train, x_test, w_train, w_test, y_train, y_test = train_test_split(
        x, w, y, test_size=0.5, stratify=w, random_state=_seed(seed)
    )
    p = x.shape[1]
    outcome = registry[model_m]
    propensity = registry[model_e]
    tau_spec = registry[model_tau]
    nu_spec = registry[model_nu]
    control = w_train == 0
    treated = w_train == 1
    mu0_fit = outcome["fit"](x_train[control], y_train[control], seed=seed)
    mu1_fit = outcome["fit"](x_train[treated], y_train[treated], seed=seed + 1)
    mu0_train = outcome["predict"](mu0_fit, x_train)
    mu1_train = outcome["predict"](mu1_fit, x_train)
    e_fit = propensity["fit"](x_train, w_train, seed=seed + 2)
    pi_train = np.clip(propensity["predict"](e_fit, x_train), clip_e, 1.0 - clip_e)
    pseudo = (mu1_train - mu0_train) + (y_train - np.where(w_train == 1, mu1_train, mu0_train)) * (
        w_train - pi_train
    ) / (pi_train * (1.0 - pi_train))
    tau_fit = tau_spec["fit"](x_train, pseudo, seed=seed + 3)
    nu_fits = []
    for j, kind in enumerate(_feature_types(x_train)):
        if kind != "continuous":
            nu_fits.append(None)
            continue
        nu_fits.append(nu_spec["fit"](np.delete(x_train, j, axis=1), x_train[:, j], seed=seed + 100 + j))

    tau_test = tau_spec["predict"](tau_fit, x_test)
    mu0_test = outcome["predict"](mu0_fit, x_test)
    mu1_test = outcome["predict"](mu1_fit, x_test)
    pi_test = np.clip(propensity["predict"](e_fit, x_test), clip_e, 1.0 - clip_e)
    risk_full = po_risk(tau_test, y_test, w_test, mu0_test, mu1_test, pi_test)
    rng = np.random.default_rng(seed + 17)
    importance = np.zeros(p, dtype=float)
    for j in range(p):
        if nu_fits[j] is None:
            continue
        nu_hat = nu_spec["predict"](nu_fits[j], np.delete(x_test, j, axis=1))
        residual = x_test[:, j] - nu_hat
        increases = np.empty(n_perm, dtype=float)
        for k in range(n_perm):
            x_perm = x_test.copy()
            x_perm[:, j] = nu_hat + rng.permutation(residual)
            tau_perm = tau_spec["predict"](tau_fit, x_perm)
            increases[k] = po_risk(tau_perm, y_test, w_test, mu0_test, mu1_test, pi_test) - risk_full
        importance[j] = float(np.mean(increases))
    return importance


def _importance(x, y, w, vimp, model_m, model_e, registry, seed, spec) -> np.ndarray:
    if vimp == "loco":
        return vimp_loco_r_risk(
            x,
            y,
            w,
            model_m,
            model_e,
            registry,
            seed=seed,
            n_splits=spec["n_splits"],
        )
    if vimp == "grf":
        return cf_variable_importance(
            x,
            y,
            w,
            model_e,
            model_m,
            seed=seed,
            n_estimators=spec["grf_n_estimators"],
            min_samples_leaf=spec["grf_min_samples_leaf"],
            max_depth=spec["grf_max_depth"],
            max_features=spec["grf_max_features"],
            mlp_hidden=spec["mlp_hidden"],
            mlp_max_iter=spec["mlp_max_iter"],
            mlp_alpha=spec["mlp_alpha"],
            ridge_alpha=spec["ridge_alpha"],
        )
    if vimp == "permucate":
        return permuCATE_vimp(
            x,
            y,
            w,
            model_m,
            model_e,
            registry,
            n_perm=spec["n_perm_cate"],
            seed=seed,
        )
    raise ValueError(f"Unknown vimp notion: {vimp}")


def cfperm_feature_pvals(
    x,
    y,
    w,
    *,
    vimp: str,
    model_m: str,
    model_e: str,
    registry: dict,
    seed: int = 0,
    n_perm: int = 49,
    n_perm_cate: int = 20,
    n_splits: int = 5,
    top_k: int = 1,
    level_feature: float = 0.99,
    level_across_feature: float = 0.95,
    alpha: float = 0.05,
    sig_ind=None,
    grf_n_estimators: int = 40,
    grf_min_samples_leaf: int = 5,
    grf_max_depth: int = 4,
    grf_max_features: str | int | float = "auto",
    mlp_hidden: int = 4,
    mlp_max_iter: int = 80,
    mlp_alpha: float = 1e-5,
    ridge_alpha: float = 0.25,
) -> dict:
    """Permute the batch label and return the CFPerm rejection summary."""
    x = _design(x)
    y = _as_1d(y)
    w = _as_1d(w).astype(int)
    spec = {
        "n_perm_cate": int(n_perm_cate),
        "n_splits": int(n_splits),
        "grf_n_estimators": int(grf_n_estimators),
        "grf_min_samples_leaf": int(grf_min_samples_leaf),
        "grf_max_depth": int(grf_max_depth),
        "grf_max_features": grf_max_features,
        "mlp_hidden": int(mlp_hidden),
        "mlp_max_iter": int(mlp_max_iter),
        "mlp_alpha": float(mlp_alpha),
        "ridge_alpha": float(ridge_alpha),
    }
    observed = _importance(x, y, w, vimp, model_m, model_e, registry, seed, spec)
    p = observed.size
    perm = np.zeros((p, n_perm), dtype=float)
    rng = np.random.default_rng(seed + 7919)
    for b in range(n_perm):
        w_b = rng.permutation(w)
        perm[:, b] = _importance(
            x, y, w_b, vimp, model_m, model_e, registry, seed + 10_000 + b, spec
        )
    p_values = (1.0 + np.sum(perm >= observed[:, None], axis=1)) / (1.0 + n_perm)
    q_feature = np.quantile(perm, q=level_feature, axis=1)
    threshold = float(np.quantile(q_feature, q=level_across_feature))
    rejected_feature = [j for j in range(p) if observed[j] > threshold]
    rejected = int(len(rejected_feature) >= top_k)
    sig = np.asarray([] if sig_ind is None else sig_ind, dtype=int)
    signal_hit = int(any(j in set(sig.tolist()) for j in rejected_feature))
    signal_imp = float(np.mean(observed[sig])) if sig.size else float("nan")
    noise_mask = np.ones(p, dtype=bool)
    if sig.size:
        noise_mask[sig] = False
    noise_imp = float(np.mean(observed[noise_mask])) if np.any(noise_mask) else float("nan")
    min_p = float(np.min(p_values))
    return {
        "rejected": rejected,
        "reject_alpha": int(min_p < alpha),
        "reject_signal": signal_hit,
        "min_p": min_p,
        "signal_imp": signal_imp,
        "noise_imp": noise_imp,
        "threshold": threshold,
        "n_rejected_features": int(len(rejected_feature)),
    }
