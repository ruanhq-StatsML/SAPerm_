"""Out-of-distribution variable importance.

Three dimension-wise scores, pooled into one weight per coordinate:

1. Pseudo-outcome risk of a PermuCATE fit, with conditional permutation.
2. Random-forest importance for the domain indicator.
3. RBF MMD, leave-one-covariate-out.

``po_risk_inference`` also returns the per-row squared pseudo-outcome residual.
That vector is the sample weight the active sampler can pass into a query rule.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from experiments.deep_active_sampling.model_registry import default_model_registry

MODEL_REGISTRY = default_model_registry()


def _as_1d(y) -> np.ndarray:
    return np.asarray(y, dtype=float).reshape(-1)


def _adapter(registry, name):
    spec = registry[name]
    if isinstance(spec, dict):
        return spec["fit"], spec["predict"]
    return spec.fit, spec.predict


def pseudo_outcome(Y, T, mu0, mu1, pi, clip: float = 1e-3) -> np.ndarray:
    """Doubly robust pseudo-outcome ``phi(Z)``."""
    pi_clip = np.clip(np.asarray(pi, dtype=float).reshape(-1), clip, 1.0 - clip)
    treated = np.asarray(T).reshape(-1).astype(int) == 1
    mu0 = np.asarray(mu0, dtype=float).reshape(-1)
    mu1 = np.asarray(mu1, dtype=float).reshape(-1)
    outcome = np.asarray(Y, dtype=float).reshape(-1)
    mu_t = np.where(treated, mu1, mu0)
    treatment = np.asarray(T, dtype=float).reshape(-1)
    return (outcome - mu_t) * (treatment - pi_clip) / (pi_clip * (1.0 - pi_clip)) + (mu1 - mu0)


def po_risk(tau, Y, T, mu0, mu1, pi, clip: float = 1e-3) -> float:
    """Mean squared gap between the pseudo-outcome and ``tau(X)``."""
    pseudo = pseudo_outcome(Y, T, mu0, mu1, pi, clip=clip)
    gap = pseudo - np.asarray(tau, dtype=float).reshape(-1)
    return float(np.mean(gap ** 2))


def po_risk_index(tau, Y, T, mu0, mu1, pi, clip: float = 1e-3) -> np.ndarray:
    """Per-row squared pseudo-outcome residual."""
    pseudo = pseudo_outcome(Y, T, mu0, mu1, pi, clip=clip)
    gap = pseudo - np.asarray(tau, dtype=float).reshape(-1)
    return gap ** 2


def infer_feature_type(X) -> list[str]:
    X = np.asarray(X)
    feature_types = ["continuous"] * X.shape[1]
    for j in range(X.shape[1]):
        n_level = np.unique(X[:, j]).size
        if n_level == 2:
            feature_types[j] = "binary"
        elif 3 <= n_level <= 5:
            feature_types[j] = "categorical"
    return feature_types


def _encode_discrete_columns(X, feature_types):
    """Encode discrete columns on a copy. Continuous columns stay as they are."""
    encoded = np.array(X, dtype=float, copy=True)
    for j, kind in enumerate(feature_types):
        if kind == "continuous":
            continue
        encoded[:, j] = LabelEncoder().fit_transform(X[:, j]).astype(float)
    return encoded


def permuCATE_fit_nuisance(
    X,
    Y,
    W,
    model_registry=None,
    model_m: str = "ridge_regressor",
    model_e: str = "logistic_classifier",
    model_tau: str = "ridge_regressor",
    model_nu: str = "ridge_regressor",
    model_nu_bi: str = "logistic_classifier",
    model_nu_multi: str = "multinomial_classifier",
    clip_e: float = 1e-2,
    seed: int = 2026,
):
    """Fit outcome, propensity, pseudo-outcome regression, and conditional models.

    The conditional model for coordinate ``j`` is ``X_j | X_{-j}``. Discrete
    columns are label-encoded on a copy before that regression.
    """
    if model_registry is None:
        model_registry = MODEL_REGISTRY
    X_raw = np.asarray(X, dtype=float)
    Y = _as_1d(Y)
    W = _as_1d(W).astype(int)
    n, p = X_raw.shape
    if n != len(Y) or n != len(W):
        raise ValueError("X, Y, and W must have the same number of rows")
    feature_types = infer_feature_type(X_raw)
    X_model = _encode_discrete_columns(X_raw, feature_types)

    fit_m, pred_m = _adapter(model_registry, model_m)
    fit_e, pred_e = _adapter(model_registry, model_e)
    fit_tau, pred_tau = _adapter(model_registry, model_tau)

    X_ctr, X_trt = X_model[W == 0], X_model[W == 1]
    Y_ctr, Y_trt = Y[W == 0], Y[W == 1]
    if len(Y_ctr) < 2 or len(Y_trt) < 2:
        raise ValueError("each treatment arm needs at least two rows to fit the nuisance")

    mu0_fit = fit_m(X_ctr, Y_ctr, seed=seed)
    mu1_fit = fit_m(X_trt, Y_trt, seed=seed + 1)
    mu0_est = pred_m(mu0_fit, X_model)
    mu1_est = pred_m(mu1_fit, X_model)
    model_e_fit = fit_e(X_model, W, seed=seed + 2)
    pi_est = np.clip(pred_e(model_e_fit, X_model), clip_e, 1.0 - clip_e)
    pseudo = pseudo_outcome(Y, W, mu0_est, mu1_est, pi_est, clip=clip_e)
    model_tau_fit = fit_tau(X_model, pseudo, seed=seed + 3)
    tau_est = pred_tau(model_tau_fit, X_model)

    nu_model = {}
    for j in range(p):
        if feature_types[j] == "continuous":
            nu_name = model_nu
        elif feature_types[j] == "binary":
            nu_name = model_nu_bi
        else:
            nu_name = model_nu_multi
        fit_nu, pred_nu = _adapter(model_registry, nu_name)
        design = np.delete(X_model, j, axis=1)
        nu_fit = fit_nu(design, X_model[:, j], seed=seed + j + 100)
        nu_model[j] = {
            "model_nu_name": nu_name,
            "model_nu": {"fit": fit_nu, "predict": pred_nu},
            "model_nu_fit": nu_fit,
        }

    return {
        "model_outcome0": {"fit": fit_m, "predict": pred_m},
        "model_outcome1": {"fit": fit_m, "predict": pred_m},
        "model_propensity_score": {"fit": fit_e, "predict": pred_e},
        "model_tau": {"fit": fit_tau, "predict": pred_tau},
        "mu0_fit": mu0_fit,
        "mu1_fit": mu1_fit,
        "model_e_fit": model_e_fit,
        "model_tau_fit": model_tau_fit,
        "mu0_est": mu0_est,
        "mu1_est": mu1_est,
        "pi_est": pi_est,
        "tau_est": tau_est,
        "pseudo_outcome": pseudo,
        "nu_model": nu_model,
        "feature_types": feature_types,
        "clip_e": float(clip_e),
        "X_model": X_model,
    }


def po_risk_inference(X_new, Y_new, T_new, outputs) -> dict:
    """Score a new batch with a fitted nuisance.

    Returns the mean pseudo-outcome risk and the per-row index used as a
    sample weight. ``T_new`` is the domain indicator for the new rows.
    """
    clip_e = float(outputs.get("clip_e", 1e-2))
    X_new = np.asarray(X_new, dtype=float)
    feature_types = outputs.get("feature_types")
    if feature_types is not None:
        X_new = _encode_discrete_columns(X_new, feature_types)
    tau_est = outputs["model_tau"]["predict"](outputs["model_tau_fit"], X_new)
    mu0_test = outputs["model_outcome0"]["predict"](outputs["mu0_fit"], X_new)
    mu1_test = outputs["model_outcome1"]["predict"](outputs["mu1_fit"], X_new)
    pi_test = np.clip(
        outputs["model_propensity_score"]["predict"](outputs["model_e_fit"], X_new),
        clip_e,
        1.0 - clip_e,
    )
    index = po_risk_index(tau_est, Y_new, T_new, mu0_test, mu1_test, pi_test, clip=clip_e)
    pseudo = pseudo_outcome(Y_new, T_new, mu0_test, mu1_test, pi_test, clip=clip_e)
    return {
        "po_risk": float(np.mean(index)),
        "po_risk_index": index,
        "tau": np.asarray(tau_est, dtype=float).reshape(-1),
        "mu0": np.asarray(mu0_test, dtype=float).reshape(-1),
        "mu1": np.asarray(mu1_test, dtype=float).reshape(-1),
        "pi": np.asarray(pi_test, dtype=float).reshape(-1),
        "pseudo": pseudo,
    }


def permuCATE_calPO(
    X,
    Y,
    W,
    model_registry=None,
    model_m: str = "ridge_regressor",
    model_e: str = "logistic_classifier",
    model_tau: str = "ridge_regressor",
    model_nu: str = "ridge_regressor",
    model_nu_bi: str = "logistic_classifier",
    model_nu_multi: str = "multinomial_classifier",
    test_size: float = 0.5,
    seed: int = 0,
    clip_e: float = 0.01,
):
    """Held-out pseudo-outcome risk of a nuisance fit on the complementary split."""
    if model_registry is None:
        model_registry = MODEL_REGISTRY
    X = np.asarray(X, dtype=float)
    Y = _as_1d(Y)
    W = _as_1d(W).astype(int)
    X_train, X_test, W_train, W_test, Y_train, Y_test = train_test_split(
        X, W, Y, test_size=test_size, stratify=W, random_state=seed
    )
    outputs = permuCATE_fit_nuisance(
        X_train,
        Y_train,
        W_train,
        model_registry=model_registry,
        model_m=model_m,
        model_e=model_e,
        model_tau=model_tau,
        model_nu=model_nu,
        model_nu_bi=model_nu_bi,
        model_nu_multi=model_nu_multi,
        clip_e=clip_e,
        seed=seed,
    )
    scored = po_risk_inference(X_test, Y_test, W_test, outputs)
    return float(scored["po_risk"])


def permuCATE_vimp_withPOModel(
    X,
    Y,
    W,
    model_registry=None,
    model_m: str = "ridge_regressor",
    model_e: str = "logistic_classifier",
    model_tau: str = "ridge_regressor",
    model_nu: str = "ridge_regressor",
    model_nu_bi: str = "logistic_classifier",
    model_nu_multi: str = "multinomial_classifier",
    n_perm: int = 20,
    test_size: float = 0.5,
    seed: int = 0,
    clip_e: float = 0.01,
    normalize: bool = False,
):
    """Conditional-permutation drop in held-out pseudo-outcome risk.

    For coordinate ``j``, the test column is replaced by the conditional fit
    plus a permutation of the residual. The importance is the average increase
    in pseudo-outcome risk. Outcome and propensity fits stay fixed.
    """
    if model_registry is None:
        model_registry = MODEL_REGISTRY
    X = np.asarray(X, dtype=float)
    Y = _as_1d(Y)
    W = _as_1d(W).astype(int)
    X_train, X_test, W_train, W_test, Y_train, Y_test = train_test_split(
        X, W, Y, test_size=test_size, stratify=W, random_state=seed
    )
    p = X.shape[1]
    outputs = permuCATE_fit_nuisance(
        X_train,
        Y_train,
        W_train,
        model_registry=model_registry,
        model_m=model_m,
        model_e=model_e,
        model_tau=model_tau,
        model_nu=model_nu,
        model_nu_bi=model_nu_bi,
        model_nu_multi=model_nu_multi,
        clip_e=clip_e,
        seed=seed,
    )
    scored = po_risk_inference(X_test, Y_test, W_test, outputs)
    po_risk_original = float(scored["po_risk"])
    # Inference encodes discrete columns. Permute that same matrix.
    feature_types = outputs["feature_types"]
    X_test_model = _encode_discrete_columns(X_test, feature_types)
    mu0_test = scored["mu0"]
    mu1_test = scored["mu1"]
    pi_test = scored["pi"]

    vimp = np.zeros(p, dtype=float)
    vimp_sd = np.zeros(p, dtype=float)
    vimp_median = np.zeros(p, dtype=float)
    for j in range(p):
        rng = np.random.default_rng(seed + j + 17)
        nu_spec = outputs["nu_model"][j]["model_nu"]
        nu_fit = outputs["nu_model"][j]["model_nu_fit"]
        nu_hat = np.asarray(
            nu_spec["predict"](nu_fit, np.delete(X_test_model, j, axis=1)),
            dtype=float,
        ).reshape(-1)
        residual = X_test_model[:, j] - nu_hat
        drops = np.zeros(n_perm, dtype=float)
        for k in range(n_perm):
            X_perm = np.array(X_test_model, copy=True)
            X_perm[:, j] = nu_hat + rng.permutation(residual)
            tau_perm = outputs["model_tau"]["predict"](outputs["model_tau_fit"], X_perm)
            risk_perm = po_risk(tau_perm, Y_test, W_test, mu0_test, mu1_test, pi_test, clip=clip_e)
            drops[k] = risk_perm - po_risk_original
        vimp[j] = float(drops.mean())
        vimp_sd[j] = float(drops.std())
        vimp_median[j] = float(np.median(drops))
    if normalize:
        total = float(np.sum(vimp))
        if total != 0.0:
            return vimp / total
        return vimp
    return vimp, vimp_sd, vimp_median


def permuCATE_vimp(*args, **kwargs):
    """Alias used by the pooled wrapper."""
    return permuCATE_vimp_withPOModel(*args, **kwargs)


def RF_domain_classifier(X_ref, X_new, seed: int = 0, n_estimators: int = 80, max_depth=None):
    """Random-forest importance for predicting which batch a row came from."""
    X_ref = np.asarray(X_ref, dtype=float)
    X_new = np.asarray(X_new, dtype=float)
    X = np.vstack([X_ref, X_new])
    domain = np.concatenate([np.zeros(len(X_ref)), np.ones(len(X_new))]).astype(int)
    p = X.shape[1]
    if max_depth is None:
        max_depth = max(2, int(np.round(np.sqrt(p))))
    clf = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_leaf=max(1, int(np.round(np.sqrt(len(X)) / 2.0))),
        n_jobs=1,
        random_state=int(seed),
    )
    clf.fit(X, domain)
    return np.asarray(clf.feature_importances_, dtype=float)


def _rbf_gamma(X, Y, max_samples: int = 400) -> float:
    n_x, n_y = X.shape[0], Y.shape[0]
    rng = np.random.default_rng(0)
    if n_x + n_y > max_samples:
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


def _rbf_parts(X, Y, gamma: float):
    XX = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    YY = np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=2)
    XY = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)
    eXX = float(np.mean(np.exp(-gamma * XX)))
    eYY = float(np.mean(np.exp(-gamma * YY)))
    eXY = float(np.mean(np.exp(-gamma * XY)))
    return eXX, eYY, eXY


def rbf_mmd2(X, Y, gamma=None) -> float:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if gamma is None:
        gamma = _rbf_gamma(X, Y)
    eXX, eYY, eXY = _rbf_parts(X, Y, gamma)
    return float(eXX + eYY - 2.0 * eXY)


def rbf_nammd2(X, Y, gamma=None) -> float:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if gamma is None:
        gamma = _rbf_gamma(X, Y)
    eXX, eYY, eXY = _rbf_parts(X, Y, gamma)
    nomi = eXX + eYY - 2.0 * eXY
    denom = 4.0 - eXX - eYY
    if abs(denom) < 1e-12:
        return 0.0
    return float(nomi / denom)


class NAMMD:
    def __init__(self, compute_kernel: str = "rbf"):
        if compute_kernel != "rbf":
            raise ValueError(f"unsupported kernel: {compute_kernel}")
        self._gamma = None

    def __call__(self, X, Y):
        if self._gamma is None:
            self._gamma = _rbf_gamma(np.asarray(X), np.asarray(Y))
        return rbf_nammd2(X, Y, gamma=self._gamma), None


class MMD:
    def __init__(self, compute_kernel: str = "rbf"):
        if compute_kernel != "rbf":
            raise ValueError(f"unsupported kernel: {compute_kernel}")
        self._gamma = None

    def __call__(self, X, Y):
        if self._gamma is None:
            self._gamma = _rbf_gamma(np.asarray(X), np.asarray(Y))
        return rbf_mmd2(X, Y, gamma=self._gamma), None


def _subsample_pair(X_ref, X_new, max_n: int, seed: int):
    rng = np.random.default_rng(seed)
    X_ref = np.asarray(X_ref, dtype=float)
    X_new = np.asarray(X_new, dtype=float)
    if len(X_ref) > max_n:
        X_ref = X_ref[rng.choice(len(X_ref), size=max_n, replace=False)]
    if len(X_new) > max_n:
        X_new = X_new[rng.choice(len(X_new), size=max_n, replace=False)]
    return X_ref, X_new


def _loco_drop(X_ref, X_new, kernel, max_n: int, seed: int) -> np.ndarray:
    X_ref, X_new = _subsample_pair(X_ref, X_new, max_n, seed)
    original, _ = kernel(X_ref, X_new)
    p = X_ref.shape[1]
    dropped = np.zeros(p, dtype=float)
    for j in range(p):
        dropped[j], _ = kernel(np.delete(X_ref, j, axis=1), np.delete(X_new, j, axis=1))
    return np.asarray(original, dtype=float) - dropped


def MMD_LOCO(X_ref, X_new, max_n: int = 1000, seed: int = 0) -> np.ndarray:
    """Full-sample RBF MMD minus MMD after dropping feature ``j``."""
    return _loco_drop(X_ref, X_new, MMD(compute_kernel="rbf"), max_n, seed)


def NAMMD_LOCO(X_ref, X_new, max_n: int = 1000, seed: int = 0) -> np.ndarray:
    return _loco_drop(X_ref, X_new, NAMMD(compute_kernel="rbf"), max_n, seed)


def brier_L2(Y_hat, Y_gt) -> np.ndarray:
    """Squared L2 distance between one-hot labels and one-hot predictions."""
    pred = np.asarray(Y_hat).astype(int).reshape(-1)
    truth = np.asarray(Y_gt).astype(int).reshape(-1)
    if pred.shape[0] != truth.shape[0]:
        raise ValueError("Y_hat and Y_gt must have the same length")
    n_levels = int(max(pred.max(initial=0), truth.max(initial=0))) + 1
    n = pred.shape[0]
    one_hot_truth = np.zeros((n, n_levels), dtype=float)
    one_hot_pred = np.zeros((n, n_levels), dtype=float)
    one_hot_truth[np.arange(n), truth] = 1.0
    one_hot_pred[np.arange(n), pred] = 1.0
    return ((one_hot_truth - one_hot_pred) ** 2).sum(axis=1)


def pred_vimp_wrapper(
    df,
    model_registry=None,
    model_m: str = "ridge_regressor",
    seed: int = 1024,
    n_perm: int = 30,
):
    """Permutation importance of a predictive model on a held-out slice.

    The last column of ``df`` is the outcome. Importance is the increase in
    test mean squared error when a test column is permuted. The fit is not
    redone inside the permutation loop.
    """
    if model_registry is None:
        model_registry = MODEL_REGISTRY
    df = np.asarray(df, dtype=float)
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=seed)
    X_train = df_train[:, :-1]
    X_test = df_test[:, :-1]
    y_train = df_train[:, -1]
    y_test = df_test[:, -1]
    p = X_train.shape[1]
    fit_fn, pred_fn = _adapter(model_registry, model_m)
    model_fit = fit_fn(X_train, y_train, seed=seed)
    pred_test = pred_fn(model_fit, X_test)
    mse_orig = float(np.mean((y_test - pred_test) ** 2))
    mse_perm = np.zeros((p, n_perm), dtype=float)
    for j in range(p):
        for k in range(n_perm):
            X_perm = np.array(X_test, copy=True)
            rng = np.random.default_rng(seed + j * n_perm + k)
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            pred_perm = pred_fn(model_fit, X_perm)
            mse_perm[j, k] = float(np.mean((y_test - pred_perm) ** 2) - mse_orig)
    return {
        "vimp_std": mse_perm.std(axis=1),
        "vimp_mean": mse_perm.mean(axis=1),
        "mse": mse_orig,
    }


def pred_vimp_wrapper_loco(
    df,
    model_registry=None,
    model_m: str = "ridge_regressor",
    seed: int = 1024,
):
    """Refit without column ``j`` and record the increase in test MSE."""
    if model_registry is None:
        model_registry = MODEL_REGISTRY
    df = np.asarray(df, dtype=float)
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=seed)
    X_train = df_train[:, :-1]
    X_test = df_test[:, :-1]
    y_train = df_train[:, -1]
    y_test = df_test[:, -1]
    p = X_train.shape[1]
    fit_fn, pred_fn = _adapter(model_registry, model_m)
    model_fit = fit_fn(X_train, y_train, seed=seed)
    mse_orig = float(np.mean((y_test - pred_fn(model_fit, X_test)) ** 2))
    mse_loco = np.zeros(p, dtype=float)
    for j in range(p):
        fit_j = fit_fn(np.delete(X_train, j, axis=1), y_train, seed=seed + j + 1)
        pred_j = pred_fn(fit_j, np.delete(X_test, j, axis=1))
        mse_loco[j] = float(np.mean((y_test - pred_j) ** 2) - mse_orig)
    return mse_loco


def pool_feature_weights(*vimps) -> np.ndarray:
    """Average the positive, sum-to-one version of each importance vector.

    A method whose scores are all non-positive contributes a flat weight.
    The result sums to one and is the dimension weight used for sample scores.
    """
    parts = []
    width = None
    for raw in vimps:
        v = np.maximum(np.asarray(raw, dtype=float).reshape(-1), 0.0)
        if width is None:
            width = v.size
        elif v.size != width:
            raise ValueError("importance vectors must have the same length")
        total = float(v.sum())
        if total <= 0.0:
            parts.append(np.full(width, 1.0 / width))
        else:
            parts.append(v / total)
    if not parts:
        raise ValueError("pool_feature_weights needs at least one vector")
    pooled = np.mean(np.stack(parts, axis=0), axis=0)
    pooled_sum = float(pooled.sum())
    if pooled_sum <= 0.0:
        return np.full(width, 1.0 / width)
    return pooled / pooled_sum


def ood_vimp_wrapper(df1, df2, n_perm: int = 20, seed: int = 0, model_registry=None):
    """Pool PermuCATE, domain-classifier, and MMD-LOCO importance.

    Each frame's last column is the outcome. Domain ``0`` is ``df1`` and
    domain ``1`` is ``df2``. The covariate blocks are what the domain
    classifier and MMD see.
    """
    if model_registry is None:
        model_registry = MODEL_REGISTRY
    df1 = np.asarray(df1, dtype=float)
    df2 = np.asarray(df2, dtype=float)
    if df1.shape[1] != df2.shape[1]:
        raise ValueError("df1 and df2 must have the same number of columns")
    X1 = df1[:, :-1]
    X2 = df2[:, :-1]
    X = np.vstack([X1, X2])
    Y = np.concatenate([df1[:, -1], df2[:, -1]])
    T = np.concatenate([np.zeros(len(df1)), np.ones(len(df2))]).astype(int)
    po_vimp, po_vimp_sd, po_vimp_median = permuCATE_vimp(
        X, Y, T, n_perm=n_perm, seed=seed, model_registry=model_registry
    )
    rf_vimp = RF_domain_classifier(X1, X2, seed=seed)
    mmd_vimp = MMD_LOCO(X1, X2, max_n=max(len(X1), len(X2), 1), seed=seed)
    return {
        "po_vimp": po_vimp,
        "po_vimp_sd": po_vimp_sd,
        "po_vimp_median": po_vimp_median,
        "rf_vimp": rf_vimp,
        "MMD_vimp": mmd_vimp,
        "pooled": pool_feature_weights(po_vimp, rf_vimp, mmd_vimp),
    }


def cauchy_combine_pval(p_values) -> float:
    """Cauchy combination of p-values (Liu and Xie, 2020)."""
    p = np.clip(np.asarray(p_values, dtype=float).reshape(-1), 1e-12, 1.0 - 1e-12)
    if p.size == 0:
        raise ValueError("p_values is empty")
    statistic = float(np.mean(np.tan(np.pi * (0.5 - p))))
    return float(0.5 - np.arctan(statistic) / np.pi)
