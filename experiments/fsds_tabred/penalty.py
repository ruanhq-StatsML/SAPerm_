"""Fast block-wise FSDS penalties for industrial feature counts.

Full ``permuCATE_vimp`` fits a conditional model per coordinate. On TabReD
(p ~ 100–400) that is too slow for a streaming loop. These helpers fit the
outcome / propensity / tau models once per batch, then:

* ``sum_VIMP`` — marginal column permutation drop in held-out PO risk, summed
  inside each block
* ``leave_one_group`` — refit after dropping a whole block (no per-feature nu)
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def _as_1d(y) -> np.ndarray:
    return np.asarray(y, dtype=float).reshape(-1)


def _adapter(registry, name):
    spec = registry[name]
    return spec["fit"], spec["predict"]


def pseudo_outcome(Y, T, mu0, mu1, pi, clip: float = 1e-3) -> np.ndarray:
    pi_clip = np.clip(np.asarray(pi, dtype=float).reshape(-1), clip, 1.0 - clip)
    treated = np.asarray(T).reshape(-1).astype(int) == 1
    mu0 = np.asarray(mu0, dtype=float).reshape(-1)
    mu1 = np.asarray(mu1, dtype=float).reshape(-1)
    outcome = np.asarray(Y, dtype=float).reshape(-1)
    mu_t = np.where(treated, mu1, mu0)
    treatment = np.asarray(T, dtype=float).reshape(-1)
    return (outcome - mu_t) * (treatment - pi_clip) / (pi_clip * (1.0 - pi_clip)) + (mu1 - mu0)


def po_risk(tau, Y, T, mu0, mu1, pi, clip: float = 1e-3) -> float:
    pseudo = pseudo_outcome(Y, T, mu0, mu1, pi, clip=clip)
    gap = pseudo - np.asarray(tau, dtype=float).reshape(-1)
    return float(np.mean(gap ** 2))


def _as_share(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    total = float(np.sum(np.maximum(raw, 0.0)))
    if total <= 0.0:
        return np.ones(len(raw), dtype=float) / max(len(raw), 1)
    return np.maximum(raw, 0.0) / total


def _stack(X_pre, Y_pre, X_cur, Y_cur):
    n0 = len(Y_pre)
    n1 = len(Y_cur)
    T = np.concatenate([np.zeros(n0), np.ones(n1)]).astype(int)
    X = np.vstack([np.asarray(X_pre, dtype=float), np.asarray(X_cur, dtype=float)])
    Y = np.concatenate([np.asarray(Y_pre).ravel(), np.asarray(Y_cur).ravel()])
    return X, Y, T


def _fit_po_bundle(X, Y, W, model_registry, model_m, model_e, model_tau, clip_e, seed):
    X = np.asarray(X, dtype=float)
    Y = _as_1d(Y)
    W = _as_1d(W).astype(int)
    fit_m, pred_m = _adapter(model_registry, model_m)
    fit_e, pred_e = _adapter(model_registry, model_e)
    fit_tau, pred_tau = _adapter(model_registry, model_tau)
    X0, X1 = X[W == 0], X[W == 1]
    Y0, Y1 = Y[W == 0], Y[W == 1]
    if len(Y0) < 2 or len(Y1) < 2:
        raise ValueError("each domain needs at least two rows")
    mu0_fit = fit_m(X0, Y0, seed=seed)
    mu1_fit = fit_m(X1, Y1, seed=seed + 1)
    mu0 = pred_m(mu0_fit, X)
    mu1 = pred_m(mu1_fit, X)
    e_fit = fit_e(X, W, seed=seed + 2)
    pi = np.clip(pred_e(e_fit, X), clip_e, 1.0 - clip_e)
    po = pseudo_outcome(Y, W, mu0, mu1, pi, clip=clip_e)
    tau_fit = fit_tau(X, po, seed=seed + 3)
    return {
        "pred_m": pred_m,
        "pred_e": pred_e,
        "pred_tau": pred_tau,
        "mu0_fit": mu0_fit,
        "mu1_fit": mu1_fit,
        "e_fit": e_fit,
        "tau_fit": tau_fit,
        "clip_e": float(clip_e),
    }


def _score_po(bundle, X, Y, W):
    mu0 = bundle["pred_m"](bundle["mu0_fit"], X)
    mu1 = bundle["pred_m"](bundle["mu1_fit"], X)
    pi = np.clip(bundle["pred_e"](bundle["e_fit"], X), bundle["clip_e"], 1.0 - bundle["clip_e"])
    tau = bundle["pred_tau"](bundle["tau_fit"], X)
    return po_risk(tau, Y, W, mu0, mu1, pi, clip=bundle["clip_e"])


def block_penalty_fast(
    X_pre,
    Y_pre,
    X_cur,
    Y_cur,
    block_indices,
    *,
    model_registry,
    kind: str = "sum_VIMP",
    model_m: str = "ridge_regressor",
    model_e: str = "logistic_classifier",
    model_tau: str = "ridge_regressor",
    n_perm: int = 12,
    test_size: float = 0.5,
    seed: int = 2026,
    clip_e: float = 0.01,
    max_features_vimp: int | None = 120,
):
    """Return a length-K probability vector of block shift shares."""
    if kind in ("vimp_sum",):
        kind = "sum_VIMP"
    elif kind in ("logo",):
        kind = "leave_one_group"
    elif kind not in ("sum_VIMP", "leave_one_group"):
        raise ValueError(kind)

    X, Y, T = _stack(X_pre, Y_pre, X_cur, Y_cur)
    Xtr, Xte, Wtr, Wte, Ytr, Yte = train_test_split(
        X, T, Y, test_size=test_size, stratify=T, random_state=seed
    )
    k = len(block_indices)
    raw = np.zeros(k, dtype=float)

    if kind == "leave_one_group":
        bundle = _fit_po_bundle(Xtr, Ytr, Wtr, model_registry, model_m, model_e, model_tau, clip_e, seed)
        base = _score_po(bundle, Xte, Yte, Wte)
        for i, cols in enumerate(block_indices):
            drop = list(np.asarray(cols).ravel())
            keep = [j for j in range(X.shape[1]) if j not in set(drop)]
            if not keep:
                raw[i] = 0.0
                continue
            b2 = _fit_po_bundle(
                Xtr[:, keep], Ytr, Wtr, model_registry, model_m, model_e, model_tau, clip_e, seed + i + 1
            )
            raw[i] = base - _score_po(b2, Xte[:, keep], Yte, Wte)
        return _as_share(raw)

    bundle = _fit_po_bundle(Xtr, Ytr, Wtr, model_registry, model_m, model_e, model_tau, clip_e, seed)
    base = _score_po(bundle, Xte, Yte, Wte)
    p = X.shape[1]
    feat_vimp = np.zeros(p, dtype=float)
    if max_features_vimp is not None and p > max_features_vimp:
        chosen = []
        per = max(1, max_features_vimp // max(k, 1))
        rng_pick = np.random.default_rng(seed + 17)
        for cols in block_indices:
            cols = list(np.asarray(cols).ravel())
            if len(cols) <= per:
                chosen.extend(cols)
            else:
                chosen.extend(rng_pick.choice(cols, size=per, replace=False).tolist())
        scored = sorted(set(int(j) for j in chosen))
    else:
        scored = list(range(p))

    for j in scored:
        rng = np.random.default_rng(seed + j)
        drops = np.empty(n_perm, dtype=float)
        for r in range(n_perm):
            Xp = Xte.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops[r] = _score_po(bundle, Xp, Yte, Wte) - base
        feat_vimp[j] = float(np.mean(drops))

    for i, cols in enumerate(block_indices):
        raw[i] = float(np.sum(feat_vimp[list(np.asarray(cols).ravel())]))
    return _as_share(raw)
