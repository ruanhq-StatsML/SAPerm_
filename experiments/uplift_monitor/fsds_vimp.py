"""FSDS scores used by the uplift monitor.

The arithmetic is the code in the FSDS tree:

- ``MMD`` and ``compute_loco_mmd_batches`` from ``VIMP_mmd_benchmark.py``
- ``compute_hsic_loco_batches`` from ``VIMP_hsic_benchmark.py``
- the R-learner loss ``mean((Y - mu - tau * (W - e))^2)`` and
  ``_fit_tau_rlearner_weighted`` from ``LOCO_vimp_r_risk.py``
- ``po_risk`` from ``permuCATE_vimp.py``
- ``mmd_path`` and ``summarize_path_nonuniqueness`` from ``fsds_nonuniqueness.py``

Those modules are not importable as checked in: two of them ``chdir`` onto a
local Mac path and import a model registry that is not in this tree. The
functions below are the same calculations, with the nuisance regressions set
to logistic and ridge so a batch monitor can call them.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _rank_positions(vimp):
    order = np.argsort(-np.asarray(vimp, dtype=float))
    rank_pos = np.empty_like(order)
    rank_pos[order] = np.arange(len(vimp))
    return rank_pos


def _subsample_batch(X, max_n, rng):
    X = np.asarray(X, dtype=float)
    if X.shape[0] <= max_n:
        return X
    idx = rng.choice(X.shape[0], max_n, replace=False)
    return X[idx]


def _median_bandwidth(Z):
    if Z.shape[0] < 2:
        return 1.0
    diff = Z[:, None, :] - Z[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    positive = dist[dist > 0]
    med = np.median(positive) if positive.size else 1.0
    return float(med) if med > 0 else 1.0


def _rbf_mmd2_unbiased(X, Y, gamma):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n = X.shape[0]
    m = Y.shape[0]
    if n < 2 or m < 2:
        return 0.0

    def _sq_dists(A, B):
        A2 = np.sum(A * A, axis=1, keepdims=True)
        B2 = np.sum(B * B, axis=1, keepdims=True)
        return A2 + B2.T - 2.0 * (A @ B.T)

    Kxx = np.exp(-gamma * _sq_dists(X, X))
    Kyy = np.exp(-gamma * _sq_dists(Y, Y))
    Kxy = np.exp(-gamma * _sq_dists(X, Y))
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    term_xx = Kxx.sum() / (n * (n - 1))
    term_yy = Kyy.sum() / (m * (m - 1))
    term_xy = Kxy.mean()
    return float(max(term_xx + term_yy - 2.0 * term_xy, 0.0))


class MMD:
    """Maximum mean discrepancy between an existing batch and a new batch."""

    def __init__(self, compute_kernel: str = "rbf", gamma: float | None = None):
        if compute_kernel != "rbf":
            raise ValueError(f"unsupported kernel: {compute_kernel}")
        self.compute_kernel = compute_kernel
        self.gamma = gamma

    def __call__(self, X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        gamma = self.gamma
        if gamma is None:
            pooled = np.vstack([X, Y])
            med = _median_bandwidth(pooled)
            gamma = 1.0 / (2.0 * med * med + 1e-12)
        stat = _rbf_mmd2_unbiased(X, Y, gamma)
        return stat, float("nan")


def compute_loco_mmd_batches(df1_X, df2_X, max_n=1000, seed=0):
    rng = np.random.default_rng(seed)
    X_exist = _subsample_batch(np.asarray(df1_X, dtype=float), max_n, rng)
    X_new = _subsample_batch(np.asarray(df2_X, dtype=float), max_n, rng)
    mmd_test = MMD(compute_kernel="rbf")
    mmd_orig, _ = mmd_test(X_exist, X_new)
    p = X_exist.shape[1]
    mmd_dic = np.zeros(p, dtype=float)
    for j in range(p):
        mmd_perm, _ = mmd_test(
            np.delete(X_exist, j, axis=1),
            np.delete(X_new, j, axis=1),
        )
        mmd_dic[j] = mmd_perm
    mmd_vimp = mmd_orig - mmd_dic
    return float(mmd_orig), mmd_vimp, _rank_positions(mmd_vimp)


def _rbf_kernel(A, B, sigma):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    A2 = np.sum(A * A, axis=1, keepdims=True)
    B2 = np.sum(B * B, axis=1, keepdims=True)
    dist = A2 + B2.T - 2.0 * (A @ B.T)
    gamma = 1.0 / (2.0 * float(sigma) ** 2 + 1e-12)
    return np.exp(-gamma * np.maximum(dist, 0.0))


def _center_kernel(K):
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def _hsic(X, W, sigma_x, sigma_w):
    X = np.asarray(X, dtype=float)
    W = np.asarray(W, dtype=float).reshape(-1, 1)
    n = X.shape[0]
    if n < 3:
        return 0.0
    K = _center_kernel(_rbf_kernel(X, X, sigma_x))
    L = _center_kernel(_rbf_kernel(W, W, sigma_w))
    return float(np.trace(K @ L) / ((n - 1) ** 2))


def compute_hsic_loco_batches(df1_X, df2_X, max_n=1000, seed=0, sigma=1.0):
    rng = np.random.default_rng(seed)
    X1 = _subsample_batch(np.asarray(df1_X, dtype=float), max_n, rng)
    X2 = _subsample_batch(np.asarray(df2_X, dtype=float), max_n, rng)
    X = np.vstack([X1, X2])
    W = np.concatenate([np.zeros(len(X1)), np.ones(len(X2))])
    p = X.shape[1]
    base = _hsic(X, W, sigma_x=sigma, sigma_w=sigma)
    vimp = np.zeros(p, dtype=float)
    for j in range(p):
        X_minus = np.delete(X, j, axis=1)
        vimp[j] = base - _hsic(X_minus, W, sigma_x=sigma, sigma_w=sigma)
    return float(base), vimp, _rank_positions(vimp)


def _fit_tau_rlearner_weighted(X, Y_tilde, W_tilde, seed: int = 0, clip_wtilde: float = 1e-3):
    X = np.asarray(X)
    Y_tilde = np.asarray(Y_tilde).reshape(-1)
    W_tilde = np.asarray(W_tilde).reshape(-1)
    mask = np.abs(W_tilde) > clip_wtilde
    if int(mask.sum()) < 8:
        return None
    z = Y_tilde[mask] / W_tilde[mask]
    weights = W_tilde[mask] ** 2
    tau_model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=1.0, random_state=seed)),
        ]
    )
    tau_model.fit(X[mask], z, ridge__sample_weight=weights)
    return tau_model


def _cross_fit_nuisance(X, Y, W, n_folds: int, seed: int):
    """Cross-fit outcome and propensity, the same split ``vimp_loco_r_risk`` uses."""
    n = len(Y)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    folds = np.array_split(order, n_folds)
    mu = np.full(n, float(np.mean(Y)))
    propensity = np.full(n, float(np.mean(W)))
    for k, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        y_tr = Y[train_idx].astype(int)
        w_tr = W[train_idx].astype(int)
        if len(np.unique(y_tr)) >= 2:
            outcome = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200, random_state=seed + k)
            outcome.fit(X[train_idx], y_tr)
            mu[test_idx] = outcome.predict_proba(X[test_idx])[:, 1]
        if len(np.unique(w_tr)) >= 2:
            propensity_model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200, random_state=seed + 3 * k)
            propensity_model.fit(X[train_idx], w_tr)
            propensity[test_idx] = propensity_model.predict_proba(X[test_idx])[:, 1]
    return mu, np.clip(propensity, 0.01, 0.99)


def r_learner_risk(X, Y, W, tau=None, seed: int = 0, n_folds: int = 3) -> float:
    """R-learner residual risk from ``vimp_loco_r_risk``.

    Nuisances are cross-fit. If ``tau`` is omitted, tau is the weighted ridge
    R-learner in ``_fit_tau_rlearner_weighted``. If ``tau`` is supplied, it is
    a deployed uplift plugged into the same loss.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    W = np.asarray(W, dtype=int).reshape(-1)
    if len(Y) < n_folds * 8:
        return float("nan")
    mu, propensity = _cross_fit_nuisance(X, Y, W, n_folds, seed)
    residual_y = Y - mu
    residual_w = W - propensity
    if tau is None:
        fitted = _fit_tau_rlearner_weighted(X, residual_y, residual_w, seed=seed)
        if fitted is None:
            return float("nan")
        tau_hat = fitted.predict(X)
    else:
        tau_hat = np.asarray(tau, dtype=float).reshape(-1)
    return float(np.mean((residual_y - tau_hat * residual_w) ** 2))


def po_risk(tau, Y, T, mu0, mu1, pi, clip: float = 1e-3) -> float:
    """PermuCATE pseudo-outcome risk, ``E[(phi(Z) - tau(X))^2]``."""
    pi_clip = np.clip(np.asarray(pi, dtype=float), clip, 1.0 - clip)
    treated = np.asarray(T).astype(int) == 1
    mu_t = np.where(treated, mu1, mu0)
    pseudo = (np.asarray(Y, dtype=float) - mu_t) * (np.asarray(T, dtype=float) - pi_clip) / (
        pi_clip * (1.0 - pi_clip)
    ) + (np.asarray(mu1, dtype=float) - np.asarray(mu0, dtype=float))
    return float(np.mean((pseudo - np.asarray(tau, dtype=float)) ** 2))


def _grow_path_blocks(df1, df2, feature_path, step):
    cols = feature_path[: step + 1]
    return df1[:, cols], df2[:, cols]


def mmd_path(df1, df2, B=20, rng=None):
    """Random feature-addition paths and the full-batch MMD."""
    rng = np.random.default_rng() if rng is None else rng
    df1 = np.asarray(df1, dtype=float)
    df2 = np.asarray(df2, dtype=float)
    p = df1.shape[1]
    mmd_test = MMD(compute_kernel="rbf")
    mmd_orig, _ = mmd_test(df1, df2)
    paths = []
    for _ in range(B):
        feature_path = rng.permutation(p)
        curve = []
        for step in range(p):
            x1, x2 = _grow_path_blocks(df1, df2, feature_path, step)
            value, _ = mmd_test(x1, x2)
            curve.append(float(value))
        paths.append(
            {
                "feature_path": feature_path.astype(int),
                "metric_path": np.asarray(curve, dtype=float),
            }
        )
    return paths, float(mmd_orig)


def summarize_path_nonuniqueness(paths, metric_orig, atol=1e-8):
    """How much random addition orders disagree before they all reach the full MMD."""
    curves = np.stack([p["metric_path"] for p in paths], axis=0)
    b, p = curves.shape
    final_vals = curves[:, -1]
    pairwise = [
        np.max(np.abs(curves[i] - curves[j]))
        for i in range(b)
        for j in range(i + 1, b)
    ]
    return {
        "n_paths": int(b),
        "n_steps": int(p),
        "metric_orig": float(metric_orig),
        "final_range": float(final_vals.max() - final_vals.min()),
        "final_all_equal_orig": bool(np.allclose(final_vals, metric_orig, atol=atol)),
        "step_std_mean": float(curves.std(axis=0).mean()),
        "step_std_max": float(curves.std(axis=0).max()),
        "pairwise_curve_diff_mean": float(np.mean(pairwise)) if pairwise else 0.0,
    }
