"""CFPerm / RFPerm / DRPerm wrappers for multimodal concept-drift power boards.

Aligned with ``ruanhq-StatsML/CFPerm`` (``DRPerm``, CFPerm-style global test) and
``ruanhq-StatsML/RFPerm`` (``Python/RFPerm_randomForest.py`` + ``R/perm_oobtest.R``).

These are **global** hypothesis tests for distribution / concept shift — **not**
feature attribution. Outcome / tau models default to ``rf_regressor``.
Default ``n_perm=150`` for CFPerm / DRPerm / RFPerm.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from benchmark_config import MODEL_REGISTRY
from experiments.multimodal_attribution.vendor.rfperm_core import (  # noqa: E402
    RFPerm as _RFPermOfficial,
    mse_loss,
    perm_test_oob_vs_pred,
)

DEFAULT_N_PERM = 150


def _perm_n_jobs() -> int:
    """Inner permutation parallelism for CF/DR (env ``CFDR_PERM_JOBS``, default 2)."""
    try:
        return max(1, int(os.environ.get("CFDR_PERM_JOBS", "2")))
    except ValueError:
        return 2


def _as_1d(y) -> np.ndarray:
    return np.asarray(y, dtype=float).reshape(-1)


def _adapter(registry, name: str):
    spec = registry[name]
    if isinstance(spec, dict):
        return spec["fit"], spec["predict"]
    return spec.fit, spec.predict


def perm_test_means(
    mse_a: np.ndarray,
    mse_b: np.ndarray,
    *,
    n_perm: int = DEFAULT_N_PERM,
    seed: int = 0,
) -> float:
    """One-sided: mean(mse_a) > mean(mse_b)? (a=pred/test, b=oob/ref)."""
    return perm_test_oob_vs_pred(mse_b, mse_a, n_perm=n_perm, seed=seed)


def RFPerm(
    df_exist,
    df_new,
    loss=None,
    *,
    n_perm: int = DEFAULT_N_PERM,
    seed: int = 0,
    n_estimators: int = 150,
) -> float:
    """RF OOB vs new-domain prediction-error test (``ruanhq-StatsML/RFPerm``).

    Thin wrapper over ``vendor.rfperm_core.RFPerm`` (official Python + R semantics).
    """
    return float(
        _RFPermOfficial(
            df_exist,
            df_new,
            loss=loss or mse_loss,
            n_perm=int(n_perm),
            seed=int(seed),
            n_estimators=int(n_estimators),
        )
    )


def DRPerm(
    X,
    Y,
    W,
    *,
    n_perm: int = DEFAULT_N_PERM,
    n_splits: int = 5,
    clip_e: float = 0.01,
    seed: int = 0,
    model_registry=MODEL_REGISTRY,
    model_m: str = "rf_regressor",
    model_e: str = "logistic_classifier",
    model_tau: str = "rf_regressor",
) -> float:
    """Doubly-robust PO-risk permutation test (``ruanhq-StatsML/CFPerm`` ``DRPerm``).

    Cross-fit ``μ(X)``, ``e(X)``; statistic = mean ``τ(X)²`` on DR pseudo-outcome
    ``(Y-μ)(W-e)``. Under the null, permute batch labels ``W``, keep ``Y-μ`` fixed,
    refit propensity + ``τ`` (permute-then-refit). Default ``n_perm=150``.
    """
    X = np.asarray(X, dtype=float)
    Y = _as_1d(Y)
    W = _as_1d(W).astype(int)
    n = X.shape[0]
    fit_m, pred_m = _adapter(model_registry, model_m)
    fit_e, pred_e = _adapter(model_registry, model_e)
    fit_tau, pred_tau = _adapter(model_registry, model_tau)

    mu = np.zeros(n, dtype=float)
    e = np.zeros(n, dtype=float)
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(folds.split(X)):
        fm = fit_m(X[tr], Y[tr], seed=seed + k)
        mu[te] = np.asarray(pred_m(fm, X[te]), dtype=float).reshape(-1)
        fe = fit_e(X[tr], W[tr], seed=seed + 100 + k)
        e[te] = np.asarray(pred_e(fe, X[te]), dtype=float).reshape(-1)
    e = np.clip(e, clip_e, 1.0 - clip_e)
    residual_y = Y - mu

    def _po_risk(W_use: np.ndarray, base_seed: int) -> float:
        fe = fit_e(X, W_use, seed=base_seed + 300)
        e_hat = np.clip(np.asarray(pred_e(fe, X), dtype=float).reshape(-1), clip_e, 1.0 - clip_e)
        pseudo = residual_y * (W_use.astype(float) - e_hat)
        ft = fit_tau(X, pseudo, seed=base_seed + 400)
        tau = np.asarray(pred_tau(ft, X), dtype=float).reshape(-1)
        return float(np.mean(tau ** 2))

    obs = _po_risk(W, seed)
    rng = np.random.default_rng(seed + 301)
    # Pre-draw permutations so parallel workers are deterministic given seed.
    W_perms = [rng.permutation(W) for _ in range(int(n_perm))]
    jobs = _perm_n_jobs()

    def _one(b: int) -> bool:
        return _po_risk(W_perms[b], seed + 500 + b) >= obs - 1e-12

    if jobs == 1 or int(n_perm) <= 1:
        exceed = sum(1 for b in range(int(n_perm)) if _one(b))
    else:
        from joblib import Parallel, delayed

        flags = Parallel(n_jobs=jobs, prefer="processes")(
            delayed(_one)(b) for b in range(int(n_perm))
        )
        exceed = int(sum(bool(f) for f in flags))
    return float((exceed + 1.0) / (n_perm + 1.0))


def CFPerm(
    X,
    Y,
    W,
    *,
    n_perm: int = DEFAULT_N_PERM,
    seed: int = 0,
    model_registry=MODEL_REGISTRY,
    model_m: str = "rf_regressor",
    top_k: int = 8,
) -> float:
    """CFPerm-style **global** test (no feature-attribution report).

    Leave-one-coordinate-out MSE rise under an RF outcome model, plus a
    batch-conditional residual contrast; max-statistic permutation of ``W``.
    Default ``n_perm=150``. ``top_k`` informative coords (by |corr|) for speed.
    """
    X = np.asarray(X, dtype=float)
    Y = _as_1d(Y)
    W = _as_1d(W).astype(int)
    _n, p = X.shape
    fit_m, pred_m = _adapter(model_registry, model_m)
    corr = np.array(
        [
            abs(float(np.corrcoef(X[:, j], Y)[0, 1])) if np.std(X[:, j]) > 1e-12 else 0.0
            for j in range(p)
        ]
    )
    k = min(int(top_k), p)
    idx = np.argsort(-corr)[:k]

    def _imp(W_use: np.ndarray, base_seed: int) -> np.ndarray:
        X_tr, X_te, Y_tr, Y_te, _W_tr, _W_te = train_test_split(
            X, Y, W_use, test_size=0.4, random_state=base_seed
        )
        fm = fit_m(X_tr, Y_tr, seed=base_seed)
        pred = np.asarray(pred_m(fm, X_te), dtype=float).reshape(-1)
        base_mse = float(np.mean((Y_te - pred) ** 2))
        imps = np.zeros(k, dtype=float)
        rng = np.random.default_rng(base_seed + 9)
        for t, j in enumerate(idx):
            Xp = np.array(X_te, copy=True)
            Xp[:, j] = rng.permutation(Xp[:, j])
            pred_j = np.asarray(pred_m(fm, Xp), dtype=float).reshape(-1)
            imps[t] = float(np.mean((Y_te - pred_j) ** 2) - base_mse)
        r = Y - np.asarray(pred_m(fm, X), dtype=float).reshape(-1)
        for t, j in enumerate(idx):
            m1 = float(np.mean(r[W_use == 1] * X[W_use == 1, j])) if np.any(W_use == 1) else 0.0
            m0 = float(np.mean(r[W_use == 0] * X[W_use == 0, j])) if np.any(W_use == 0) else 0.0
            imps[t] = abs(imps[t]) + abs(m1 - m0)
        return imps

    obs = float(np.max(_imp(W, seed)))
    rng = np.random.default_rng(seed + 77)
    W_perms = [rng.permutation(W) for _ in range(int(n_perm))]
    jobs = _perm_n_jobs()

    def _one(b: int) -> bool:
        return float(np.max(_imp(W_perms[b], seed + 100 + b))) >= obs - 1e-12

    if jobs == 1 or int(n_perm) <= 1:
        exceed = sum(1 for b in range(int(n_perm)) if _one(b))
    else:
        from joblib import Parallel, delayed

        flags = Parallel(n_jobs=jobs, prefer="processes")(
            delayed(_one)(b) for b in range(int(n_perm))
        )
        exceed = int(sum(bool(f) for f in flags))
    return float((exceed + 1.0) / (n_perm + 1.0))


def apply_concept_drift(
    X_new: np.ndarray,
    y_new: np.ndarray,
    *,
    amount: float = 0.35,
    noise: float = 0.70,
    seed: int = 0,
) -> np.ndarray:
    """Manual concept drift: keep ``X``, change ``Y|X`` on the new domain.

    Uses a **mid-PC nonlinear** signal (not the leading PC) plus outcome noise so
    the DGP is not trivial for RFPerm (amount=1.5 on PC0 hit the p=1/(n_perm+1)
    floor every rep). Still **no independent re-zscore** — new ``Y`` stays on the
    exist scale so OOB-vs-pred MSE stays aligned for RFPerm.
    """
    rng = np.random.default_rng(int(seed))
    X_new = np.asarray(X_new, dtype=float)
    y = np.asarray(y_new, dtype=float).reshape(-1).copy()
    p = X_new.shape[1]
    # Prefer mid-band coords when PCA concat is wide; fall back gracefully.
    i0 = min(5, max(0, p - 1))
    i1 = min(11, max(0, p - 1))
    i2 = min(17, max(0, p - 1))
    x0, x1, x2 = X_new[:, i0], X_new[:, i1], X_new[:, i2]
    signal = np.tanh(x0) + 0.5 * x1 * x2 + 0.25 * np.sin(x0 + x1)
    y = y + float(amount) * signal + float(noise) * rng.normal(size=len(y))
    return y
