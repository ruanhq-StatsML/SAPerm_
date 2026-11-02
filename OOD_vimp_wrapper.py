"""Out-of-distribution variable importance.

Three scores for a pair of batches, plus the PermuCATE pseudo-outcome risk
that the multi-treatment extension calls:

1. Pseudo-outcome risk of a doubly robust CATE fit, with conditional
   permutation (PermuCATE) and the same leave-one-covariate drop in risk.
2. Random-forest importance for a domain classifier (which batch a row
   came from).
3. RBF MMD, and its normalized sibling NAMMD, after dropping one covariate.

The nuisance regressions come from ``benchmark_config.MODEL_REGISTRY``.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from benchmark_config import MODEL_REGISTRY


def _as_1d(X):
    return np.asarray(X).reshape(-1)


def po_risk(tau, Y, T, mu0, mu1, pi, clip=1e-3):
    """Mean squared error of a CATE fit against the doubly robust pseudo-outcome."""
    pi_clip = np.clip(np.asarray(pi, dtype=float), clip, 1.0 - clip)
    treated = np.asarray(T).reshape(-1).astype(int) == 1
    mu_t = np.where(treated, mu1, mu0)
    pseudo = (np.asarray(Y, dtype=float).reshape(-1) - mu_t) * (
        np.asarray(T, dtype=float).reshape(-1) - pi_clip
    ) / (pi_clip * (1.0 - pi_clip)) + (
        np.asarray(mu1, dtype=float) - np.asarray(mu0, dtype=float)
    )
    return float(np.mean((pseudo - np.asarray(tau, dtype=float).reshape(-1)) ** 2))


def infer_feature_type(X):
    X = np.asarray(X)
    p = X.shape[1]
    feature_types = ["continuous"] * p
    for j in range(p):
        n_level = np.unique(X[:, j]).size
        if n_level <= 2:
            feature_types[j] = "binary"
        elif n_level <= 5:
            feature_types[j] = "categorical"
    return feature_types


def _encode_discrete(X, feature_types):
    """Copy X and integer-encode discrete columns. Continuous columns stay put."""
    X_work = np.array(X, dtype=float, copy=True)
    encoders = {}
    for j, kind in enumerate(feature_types):
        if kind == "continuous":
            continue
        encoder = LabelEncoder()
        X_work[:, j] = encoder.fit_transform(np.asarray(X[:, j]).reshape(-1))
        encoders[j] = encoder
    return X_work, encoders


def _apply_encoders(X, encoders):
    X_work = np.array(X, dtype=float, copy=True)
    for j, encoder in encoders.items():
        raw = np.asarray(X[:, j]).reshape(-1)
        known = set(encoder.classes_.tolist())
        fallback = encoder.classes_[0]
        safe = np.array([value if value in known else fallback for value in raw])
        X_work[:, j] = encoder.transform(safe)
    return X_work


def permuCATE_fit_nuisance(
    X, Y, W,
    model_registry=MODEL_REGISTRY,
    model_m="rf_regressor",
    model_e="logistic_classifier",
    model_tau="rf_regressor",
    model_nu="rf_regressor",
    model_nu_bi="logistic_classifier",
    model_nu_multi="multinomial_classifier",
    clip_e=1e-2,
    seed=2026,
):
    """Cross-sample nuisance fits for a binary treatment: mu0, mu1, propensity, tau, nu_j."""
    X = np.asarray(X, dtype=float)
    Y = _as_1d(Y).astype(float)
    W = _as_1d(W).astype(int)
    n, p = X.shape
    if n != len(Y) or n != len(W):
        raise ValueError("X, Y, and W must have the same number of rows")
    if len(np.unique(W)) < 2:
        raise ValueError("permuCATE_fit_nuisance needs both treatment arms")

    model_outcome0 = model_registry[model_m]
    model_outcome1 = model_registry[model_m]
    model_propensity_score = model_registry[model_e]
    model_tau_spec = model_registry[model_tau]
    X_ctr, X_trt = X[W == 0], X[W == 1]
    Y_ctr, Y_trt = Y[W == 0], Y[W == 1]
    mu0_fit = model_outcome0["fit"](X_ctr, Y_ctr, seed=seed)
    mu1_fit = model_outcome1["fit"](X_trt, Y_trt, seed=seed + 1)
    mu0_est = model_outcome0["predict"](mu0_fit, X)
    mu1_est = model_outcome1["predict"](mu1_fit, X)
    model_e_fit = model_propensity_score["fit"](X, W, seed=seed + 2)
    pi_est = np.clip(
        model_propensity_score["predict"](model_e_fit, X),
        clip_e,
        1.0 - clip_e,
    )
    pseudo_outcome = mu1_est - mu0_est + (
        Y - np.where(W == 1, mu1_est, mu0_est)
    ) * (W - pi_est) / (pi_est * (1.0 - pi_est))
    model_tau_fit = model_tau_spec["fit"](X, pseudo_outcome, seed=seed + 3)
    tau_est = model_tau_spec["predict"](model_tau_fit, X)

    feature_types = infer_feature_type(X)
    X_nu, encoders = _encode_discrete(X, feature_types)
    nu_model = {}
    for j in range(p):
        if feature_types[j] == "continuous":
            model_nu_spec = model_registry[model_nu]
        elif feature_types[j] == "binary":
            model_nu_spec = model_registry[model_nu_bi]
        else:
            model_nu_spec = model_registry[model_nu_multi]
        others = np.delete(X_nu, j, axis=1)
        model_nu_fit = model_nu_spec["fit"](others, X_nu[:, j], seed=seed + j + 100)
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
        "feature_types": feature_types,
        "encoders": encoders,
    }


def _heldout_nuisance(X, Y, W, test_size, seed, clip_e, registry_kwargs):
    X_train, X_test, W_train, W_test, Y_train, Y_test = train_test_split(
        X, W, Y, test_size=test_size, stratify=W, random_state=seed
    )
    outputs = permuCATE_fit_nuisance(
        X_train, Y_train, W_train, clip_e=clip_e, seed=seed, **registry_kwargs
    )
    tau_est = outputs["model_tau"]["predict"](outputs["model_tau_fit"], X_new=X_test)
    mu0_test = outputs["model_outcome0"]["predict"](outputs["mu0_fit"], X_new=X_test)
    mu1_test = outputs["model_outcome1"]["predict"](outputs["mu1_fit"], X_new=X_test)
    pi_test = np.clip(
        outputs["model_propensity_score"]["predict"](outputs["model_e_fit"], X_new=X_test),
        clip_e,
        1.0 - clip_e,
    )
    return outputs, X_test, Y_test, W_test, tau_est, mu0_test, mu1_test, pi_test


def permuCATE_calPO(
    X, Y, W,
    model_registry=MODEL_REGISTRY,
    model_m="rf_regressor",
    model_e="logistic_classifier",
    model_tau="rf_regressor",
    model_nu="rf_regressor",
    model_nu_bi="logistic_classifier",
    model_nu_multi="multinomial_classifier",
    test_size=0.5,
    seed=0,
    clip_e=0.01,
    **_ignored,
):
    """Held-out pseudo-outcome risk of the fitted CATE, with no permutation."""
    X = np.asarray(X, dtype=float)
    Y = _as_1d(Y)
    W = _as_1d(W).astype(int)
    registry_kwargs = dict(
        model_registry=model_registry,
        model_m=model_m,
        model_e=model_e,
        model_tau=model_tau,
        model_nu=model_nu,
        model_nu_bi=model_nu_bi,
        model_nu_multi=model_nu_multi,
    )
    _, _, Y_test, W_test, tau_est, mu0_test, mu1_test, pi_test = _heldout_nuisance(
        X, Y, W, test_size, seed, clip_e, registry_kwargs
    )
    return po_risk(tau_est, Y_test, W_test, mu0_test, mu1_test, pi_test, clip=clip_e)


def permuCATE_vimp_withPOModel(
    X, Y, W,
    model_registry=MODEL_REGISTRY,
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
    """Conditional-permutation importance of each feature for pseudo-outcome risk.

    For feature ``j``, residuals around ``nu_j(X_{-j})`` are shuffled and the
    increase in held-out ``po_risk`` is recorded. The return is the mean, the
    standard deviation, and the median of those increases.
    """
    X = np.asarray(X, dtype=float)
    Y = _as_1d(Y)
    W = _as_1d(W).astype(int)
    if np.unique(W).size != 2:
        raise ValueError("permuCATE_vimp expects a binary treatment; use permuCATE_vimp_multiple_trt")
    n, p = X.shape
    registry_kwargs = dict(
        model_registry=model_registry,
        model_m=model_m,
        model_e=model_e,
        model_tau=model_tau,
        model_nu=model_nu,
        model_nu_bi=model_nu_bi,
        model_nu_multi=model_nu_multi,
    )
    outputs, X_test, Y_test, W_test, tau_est, mu0_test, mu1_test, pi_test = _heldout_nuisance(
        X, Y, W, test_size, seed, clip_e, registry_kwargs
    )
    po_risk_original = po_risk(tau_est, Y_test, W_test, mu0_test, mu1_test, pi_test, clip=clip_e)
    X_test_nu = _apply_encoders(X_test, outputs["encoders"])
    vimp = np.zeros(p, dtype=float)
    vimp_sd = np.zeros(p, dtype=float)
    vimp_median = np.zeros(p, dtype=float)
    for j in range(p):
        rng = np.random.default_rng(seed + j + 1)
        nu_spec = outputs["nu_model"][j]["model_nu"]
        nu_fit = outputs["nu_model"][j]["model_nu_fit"]
        nu_hat = np.asarray(
            nu_spec["predict"](nu_fit, np.delete(X_test_nu, j, axis=1)),
            dtype=float,
        ).reshape(-1)
        residual = X_test_nu[:, j] - nu_hat
        drops = np.zeros(n_perm, dtype=float)
        for k in range(n_perm):
            X_perm = X_test_nu.copy()
            X_perm[:, j] = nu_hat + rng.permutation(residual)
            tau_perm = outputs["model_tau"]["predict"](outputs["model_tau_fit"], X_perm)
            drops[k] = po_risk(
                tau_perm, Y_test, W_test, mu0_test, mu1_test, pi_test, clip=clip_e
            ) - po_risk_original
        vimp[j] = float(drops.mean())
        vimp_sd[j] = float(drops.std())
        vimp_median[j] = float(np.median(drops))
    if normalize:
        total = float(np.sum(vimp))
        if total > 0:
            return vimp / total
        return vimp
    return vimp, vimp_sd, vimp_median


def permuCATE_vimp(X, Y, W, **kwargs):
    """Alias used by the multi-treatment driver and ``ood_vimp_wrapper``."""
    return permuCATE_vimp_withPOModel(X, Y, W, **kwargs)


def RF_domain_classifier(X_ref, X_new, seed=0):
    """Random-forest importance for predicting which batch a row came from."""
    X_ref = np.asarray(X_ref, dtype=float)
    X_new = np.asarray(X_new, dtype=float)
    X = np.vstack([X_ref, X_new])
    W = np.concatenate([np.zeros(len(X_ref)), np.ones(len(X_new))]).astype(int)
    p = X.shape[1]
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=max(2, int(np.round(np.sqrt(p)))),
        min_samples_leaf=max(1, int(np.round(np.sqrt(len(X)) // 2))),
        n_jobs=1,
        random_state=seed,
    )
    clf.fit(X, W)
    return clf.feature_importances_


def _rbf_gamma(X, Y, max_samples=400):
    n_x, n_y = X.shape[0], Y.shape[0]
    if n_x + n_y > max_samples:
        rng = np.random.default_rng(0)
        idx_x = rng.choice(n_x, size=min(n_x, max_samples // 2), replace=False)
        idx_y = rng.choice(n_y, size=min(n_y, max_samples // 2), replace=False)
        Z = np.vstack([X[idx_x], Y[idx_y]])
    else:
        Z = np.vstack([X, Y])
    sq = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=2)
    positive = sq[sq > 0]
    if positive.size == 0:
        return 1.0
    return 1.0 / float(np.median(positive))


def rbf_mmd2(X, Y, gamma=None):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if gamma is None:
        gamma = _rbf_gamma(X, Y)
    XX = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    YY = np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=2)
    XY = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)
    return float(
        np.mean(np.exp(-gamma * XX))
        + np.mean(np.exp(-gamma * YY))
        - 2.0 * np.mean(np.exp(-gamma * XY))
    )


def rbf_nammd2(X, Y, gamma=None):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if gamma is None:
        gamma = _rbf_gamma(X, Y)
    XX = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    YY = np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=2)
    XY = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)
    eXX = float(np.mean(np.exp(-gamma * XX)))
    eYY = float(np.mean(np.exp(-gamma * YY)))
    eXY = float(np.mean(np.exp(-gamma * XY)))
    nomi = eXX + eYY - 2.0 * eXY
    denom = 4.0 - eXX - eYY
    if abs(denom) < 1e-12:
        return 0.0
    return nomi / denom


class NAMMD:
    def __init__(self, compute_kernel="rbf"):
        if compute_kernel != "rbf":
            raise ValueError(f"unsupported kernel: {compute_kernel}")
        self._gamma = None

    def __call__(self, X, Y):
        if self._gamma is None:
            self._gamma = _rbf_gamma(np.asarray(X), np.asarray(Y))
        return rbf_nammd2(X, Y, gamma=self._gamma), None


class MMD:
    def __init__(self, compute_kernel="rbf"):
        if compute_kernel != "rbf":
            raise ValueError(f"unsupported kernel: {compute_kernel}")
        self._gamma = None

    def __call__(self, X, Y):
        if self._gamma is None:
            self._gamma = _rbf_gamma(np.asarray(X), np.asarray(Y))
        return rbf_mmd2(X, Y, gamma=self._gamma), None


def _mmd_loco(X_ref, X_new, kernel, max_n=1000, seed=0):
    rng = np.random.default_rng(seed)
    X_ref = np.asarray(X_ref, dtype=float)
    X_new = np.asarray(X_new, dtype=float)
    if len(X_ref) > max_n:
        X_ref = X_ref[rng.choice(len(X_ref), size=max_n, replace=False)]
    if len(X_new) > max_n:
        X_new = X_new[rng.choice(len(X_new), size=max_n, replace=False)]
    mmd_orig, _ = kernel(X_ref, X_new)
    p = X_ref.shape[1]
    mmd_drop = np.zeros(p, dtype=float)
    for j in range(p):
        mmd_drop[j], _ = kernel(
            np.delete(X_ref, j, axis=1),
            np.delete(X_new, j, axis=1),
        )
    return mmd_orig - mmd_drop


def NAMMD_LOCO(X_ref, X_new, max_n=1000, seed=0):
    return _mmd_loco(X_ref, X_new, NAMMD(compute_kernel="rbf"), max_n=max_n, seed=seed)


def MMD_LOCO(X_ref, X_new, max_n=1000, seed=0):
    """Full-sample RBF MMD minus MMD after dropping feature j."""
    return _mmd_loco(X_ref, X_new, MMD(compute_kernel="rbf"), max_n=max_n, seed=seed)


def brier_L2(Y_hat, Y_gt):
    """Per-row multiclass Brier score from hard labels."""
    Y_hat = np.asarray(Y_hat).astype(int).reshape(-1)
    Y_gt = np.asarray(Y_gt).astype(int).reshape(-1)
    n_classes = int(max(Y_hat.max(initial=0), Y_gt.max(initial=0)) + 1)
    eye = np.eye(n_classes)
    return np.sum((eye[Y_gt] - eye[Y_hat]) ** 2, axis=1)


def pred_vimp_wrapper(
    df,
    model_registry=MODEL_REGISTRY,
    model_m="rf_regressor",
    seed=1024,
    n_perm=100,
):
    """Permutation importance for predictive MSE. The last column is the outcome."""
    df = np.asarray(df, dtype=float)
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=seed)
    df1_X = df_train[:, :-1]
    df2_X = df_test[:, :-1]
    df1_Y = df_train[:, -1]
    df2_Y = df_test[:, -1]
    p = df1_X.shape[1]
    model = model_registry[model_m]
    model_fit = model["fit"](df1_X, df1_Y, seed=seed)
    pred_test = model["predict"](model_fit, df2_X)
    mse_orig = float(np.mean((df2_Y - pred_test) ** 2))
    mse_perm = np.zeros((p, n_perm))
    for j in range(p):
        for k in range(n_perm):
            df2_X_new = df2_X.copy()
            rng = np.random.default_rng(seed + j * n_perm + k)
            df2_X_new[:, j] = rng.permutation(df2_X_new[:, j])
            df2_pred = model["predict"](model_fit, df2_X_new)
            mse_perm[j, k] = np.mean((df2_Y - df2_pred) ** 2) - mse_orig
    return {
        "vimp_std": mse_perm.std(axis=1),
        "vimp_mean": mse_perm.mean(axis=1),
        "mse_orig": mse_orig,
    }


def pred_vimp_wrapper_loco(
    df,
    model_registry=MODEL_REGISTRY,
    model_m="rf_regressor",
    seed=1024,
):
    """Leave-one-covariate-out increase in predictive MSE."""
    df = np.asarray(df, dtype=float)
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=seed)
    df1_X = df_train[:, :-1]
    df2_X = df_test[:, :-1]
    df1_Y = df_train[:, -1]
    df2_Y = df_test[:, -1]
    p = df1_X.shape[1]
    model = model_registry[model_m]
    model_fit = model["fit"](df1_X, df1_Y, seed=seed)
    pred_test = model["predict"](model_fit, df2_X)
    mse_orig = float(np.mean((df2_Y - pred_test) ** 2))
    mse_loco = np.zeros(p)
    for j in range(p):
        model_fit_new = model["fit"](np.delete(df1_X, j, axis=1), df1_Y, seed=seed + j + 1)
        df2_pred = model["predict"](model_fit_new, np.delete(df2_X, j, axis=1))
        mse_loco[j] = np.mean((df2_Y - df2_pred) ** 2) - mse_orig
    return mse_loco


def ood_vimp_wrapper(df1, df2, n_perm=25, **kwargs):
    """PO-risk PermuCATE, domain-classifier importance, and MMD-LOCO.

    Each frame's last column is the outcome. The treatment inside PermuCATE
    is the batch indicator, so the PO-risk score attributes a change in the
    conditional outcome between the two batches.
    """
    df1 = np.asarray(df1, dtype=float)
    df2 = np.asarray(df2, dtype=float)
    n1 = df1.shape[0]
    n2 = df2.shape[0]
    df1_X = df1[:, :-1]
    df2_X = df2[:, :-1]
    X = np.vstack([df1_X, df2_X])
    Y = np.concatenate([df1[:, -1], df2[:, -1]])
    T = np.concatenate([np.zeros(n1), np.ones(n2)])
    po_vimp, po_vimp_sd, po_vimp_median = permuCATE_vimp(X, Y, T, n_perm=n_perm, **kwargs)
    rf_vimp = RF_domain_classifier(df1_X, df2_X)
    mmd_vimp = MMD_LOCO(df1_X, df2_X, max_n=max(n1, n2))
    return {
        "po_vimp": po_vimp,
        "po_vimp_sd": po_vimp_sd,
        "po_vimp_median": po_vimp_median,
        "rf_vimp": rf_vimp,
        "MMD_vimp": mmd_vimp,
    }


def cauchy_combine_pval(p_values):
    """Cauchy combination p-value. Inputs are clipped away from 0 and 1."""
    p = np.clip(np.asarray(p_values, dtype=float), 1e-12, 1.0 - 1e-12)
    stat = np.mean(np.tan(np.pi * (0.5 - p)))
    return float(0.5 - np.arctan(stat) / np.pi)


__all__ = [
    "po_risk",
    "infer_feature_type",
    "permuCATE_fit_nuisance",
    "permuCATE_calPO",
    "permuCATE_vimp_withPOModel",
    "permuCATE_vimp",
    "RF_domain_classifier",
    "rbf_mmd2",
    "rbf_nammd2",
    "MMD",
    "NAMMD",
    "MMD_LOCO",
    "NAMMD_LOCO",
    "brier_L2",
    "pred_vimp_wrapper",
    "pred_vimp_wrapper_loco",
    "ood_vimp_wrapper",
    "cauchy_combine_pval",
]
