"""High-order interaction / shift quantities.

One module for:
  - PO-risk (+ DRPerm p-value)
  - R-risk (+ CFPerm LOCO)
  - RBF-MMD² (per modality / share π)
  - predictive power-loss (RFPerm)
  - ρ_PO = PO(rich) / PO(base)

No dataset logic here — callers pass arrays + optional modality indices.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from experiments.meta_vs_embed_power.stats_utils import (
    as_1d,
    batch_xy,
    brier_losses,
    make_folds,
    rf_classifier,
    rf_regressor,
    standardize_pair,
)

# ---------------------------------------------------------------------------
# R-risk / power-loss scalars
# ---------------------------------------------------------------------------


def r_risk(X: np.ndarray, Y: np.ndarray, W: np.ndarray, *, seed: int = 0, n_estimators: int = 60) -> float:
    """E[(μ₁(X)−μ₀(X))²] — apparent batch effect on Y given X."""
    Y = as_1d(Y)
    W = as_1d(W).astype(int)
    Xw = np.column_stack([X, W])
    m = rf_regressor(seed, n_estimators=n_estimators)
    m.fit(Xw, Y)
    x0 = np.column_stack([X, np.zeros(len(X))])
    x1 = np.column_stack([X, np.ones(len(X))])
    return float(np.mean((m.predict(x1) - m.predict(x0)) ** 2))


def power_loss(
    X_exist: np.ndarray,
    y_exist: np.ndarray,
    X_new: np.ndarray,
    y_new: np.ndarray,
    *,
    seed: int = 0,
    n_estimators: int = 80,
) -> float:
    """Mean Brier(new) − mean Brier_OOB(exist); RFPerm statistic."""
    y_exist, y_new = as_1d(y_exist), as_1d(y_new)
    clf = rf_classifier(seed, n_estimators=n_estimators, oob_score=True)
    clf.fit(X_exist, y_exist.astype(int))
    oob = clf.oob_decision_function_
    classes = list(clf.classes_)
    if oob.ndim == 2:
        y_hat_oob = oob[:, classes.index(1)] if 1 in classes else 1.0 - oob[:, 0]
    else:
        y_hat_oob = oob
    proba = clf.predict_proba(X_new)
    y_hat_new = proba[:, classes.index(1)] if 1 in classes else 1.0 - proba[:, 0]
    return float(np.mean(brier_losses(y_new, y_hat_new)) - np.mean(brier_losses(y_exist, y_hat_oob)))


# ---------------------------------------------------------------------------
# PO-risk
# ---------------------------------------------------------------------------


def po_risk(
    X: np.ndarray,
    Y: np.ndarray,
    W: np.ndarray,
    *,
    n_splits: int = 2,
    seed: int = 0,
    clip_e: float = 0.05,
    n_estimators: int = 60,
) -> float:
    """E[τ(X)²] with τ≈(Y−μ)(W−e) — DRPerm statistic."""
    out = drperm_pvalue(
        X, Y, W, n_perm=0, n_splits=n_splits, seed=seed, clip_e=clip_e, n_estimators=n_estimators
    )
    return float(out["po_risk"])


def rho_po(po_rich: float, po_base: float, *, eps: float = 1e-12) -> float:
    """ρ_PO = PO(rich) / PO(base)."""
    return float(po_rich / max(po_base, eps))


# ---------------------------------------------------------------------------
# MMD
# ---------------------------------------------------------------------------


def rbf_mmd2(X: np.ndarray, Y: np.ndarray, *, max_median: int = 400, seed: int = 0) -> float:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X, Y must be 2-d")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"dim mismatch {X.shape} vs {Y.shape}")
    if len(X) == 0 or len(Y) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    n = len(X) + len(Y)
    if n > max_median:
        ix = rng.choice(len(X), min(len(X), max_median // 2), replace=False)
        iy = rng.choice(len(Y), min(len(Y), max_median // 2), replace=False)
        Z = np.vstack([X[ix], Y[iy]])
    else:
        Z = np.vstack([X, Y])
    sq = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=-1)
    pos = sq[sq > 0]
    gamma = 1.0 / float(np.median(pos)) if pos.size else 1.0

    def kmean(A, B):
        d = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
        return float(np.mean(np.exp(-gamma * d)))

    return float(kmean(X, X) + kmean(Y, Y) - 2.0 * kmean(X, Y))


def mmd_by_modality(
    modalities: Mapping[str, np.ndarray],
    idx0: np.ndarray,
    idx1: np.ndarray,
    *,
    standardize: bool = True,
    max_median: int = 400,
    seed: int = 0,
) -> dict[str, float]:
    idx0, idx1 = np.asarray(idx0, dtype=int), np.asarray(idx1, dtype=int)
    out: dict[str, float] = {}
    for i, (name, X) in enumerate(modalities.items()):
        X = np.asarray(X, dtype=float)
        X0, X1 = X[idx0], X[idx1]
        if standardize:
            X0, X1 = standardize_pair(X0, X1)
        out[name] = rbf_mmd2(X0, X1, max_median=max_median, seed=seed + i)
    return out


def mmd_share(mmd: Mapping[str, float]) -> dict[str, float]:
    """π_m = MMD_m / Σ MMD."""
    total = float(sum(max(0.0, v) for v in mmd.values())) + 1e-12
    return {k: float(max(0.0, v) / total) for k, v in mmd.items()}


def mmd_from_indices(
    X: np.ndarray,
    modality_indices: Mapping[str, list[int] | np.ndarray],
    idx0: np.ndarray,
    idx1: np.ndarray,
    **kwargs,
) -> dict[str, float]:
    X = np.asarray(X, dtype=float)
    mods = {name: X[:, np.asarray(cols, dtype=int)] for name, cols in modality_indices.items()}
    return mmd_by_modality(mods, idx0, idx1, **kwargs)


# ---------------------------------------------------------------------------
# Permutation tests (power)
# ---------------------------------------------------------------------------


def rfperm_pvalue(
    X_exist, y_exist, X_new, y_new, *, n_perm: int = 120, seed: int = 0, n_estimators: int = 80
) -> dict:
    obs = power_loss(X_exist, y_exist, X_new, y_new, seed=seed, n_estimators=n_estimators)
    y_exist, y_new = as_1d(y_exist), as_1d(y_new)
    clf = rf_classifier(seed, n_estimators=n_estimators, oob_score=True)
    clf.fit(X_exist, y_exist.astype(int))
    oob = clf.oob_decision_function_
    classes = list(clf.classes_)
    if oob.ndim == 2:
        y_hat_oob = oob[:, classes.index(1)] if 1 in classes else 1.0 - oob[:, 0]
    else:
        y_hat_oob = oob
    proba = clf.predict_proba(X_new)
    y_hat_new = proba[:, classes.index(1)] if 1 in classes else 1.0 - proba[:, 0]
    loss_exist = brier_losses(y_exist, y_hat_oob)
    loss_new = brier_losses(y_new, y_hat_new)
    rng = np.random.default_rng(seed + 11)
    pooled = np.concatenate([loss_exist, loss_new])
    n_e = len(loss_exist)
    exceed = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        dif = float(np.mean(perm[n_e:]) - np.mean(perm[:n_e]))
        if dif >= obs - 1e-15:
            exceed += 1
    p = (1 + exceed) / (1 + n_perm)
    return {
        "method": "RFPerm",
        "statistic": float(obs),
        "power_loss": float(obs),
        "p_value": float(p),
        "reject": bool(p < 0.05),
    }


def drperm_pvalue(
    X, Y, W, *, n_perm: int = 120, n_splits: int = 2, seed: int = 0, clip_e: float = 0.05, n_estimators: int = 60
) -> dict:
    X = np.asarray(X, dtype=float)
    Y, W = as_1d(Y), as_1d(W).astype(int)
    n = X.shape[0]
    folds = make_folds(n, n_splits, seed)
    mu_hat = np.zeros(n)
    e_hat = np.zeros(n)
    for k, te in enumerate(folds):
        tr = np.setdiff1d(np.arange(n), te)
        m = rf_regressor(seed + k, n_estimators=n_estimators)
        m.fit(X[tr], Y[tr])
        mu_hat[te] = m.predict(X[te])
        if len(np.unique(W[tr])) < 2:
            e_hat[te] = float(np.mean(W[tr]))
        else:
            e = rf_classifier(seed + 100 + k, n_estimators=max(20, n_estimators // 2))
            e.fit(X[tr], W[tr])
            classes = list(e.classes_)
            proba = e.predict_proba(X[te])
            e_hat[te] = proba[:, classes.index(1)] if 1 in classes else 1.0 - proba[:, 0]
    e_hat = np.clip(e_hat, clip_e, 1 - clip_e)
    residual_y = Y - mu_hat
    n_est_e = max(20, n_estimators // 2)

    def _crossfit_e(Wb, seed_e):
        out = np.zeros(n)
        for k, te in enumerate(folds):
            tr = np.setdiff1d(np.arange(n), te)
            if len(np.unique(Wb[tr])) < 2:
                out[te] = float(np.mean(Wb[tr]))
                continue
            e = rf_classifier(seed_e + k, n_estimators=n_est_e)
            e.fit(X[tr], Wb[tr])
            classes = list(e.classes_)
            proba = e.predict_proba(X[te])
            out[te] = proba[:, classes.index(1)] if 1 in classes else 1.0 - proba[:, 0]
        return np.clip(out, clip_e, 1.0 - clip_e)

    def _po(Wb, e_hat_b, seed_tau):
        po_b = residual_y * (Wb - e_hat_b)
        tau_b = rf_regressor(seed_tau, n_estimators=n_estimators)
        tau_b.fit(X, po_b)
        return float(np.mean(tau_b.predict(X) ** 2))

    observed = _po(W, e_hat, seed + 200)
    if n_perm <= 0:
        return {
            "method": "DRPerm",
            "statistic": observed,
            "p_value": float("nan"),
            "reject": False,
            "po_risk": observed,
        }
    rng = np.random.default_rng(seed + 7)
    null = np.zeros(n_perm)
    for b in range(n_perm):
        Wb = rng.permutation(W)
        null[b] = _po(Wb, _crossfit_e(Wb, seed + 300 + 17 * b), seed + 400 + b)
    p = (1.0 + np.sum(null >= observed - 1e-15)) / (1 + n_perm)
    return {
        "method": "DRPerm",
        "statistic": observed,
        "p_value": float(p),
        "reject": p < 0.05,
        "po_risk": observed,
        "mu_hat_mean": float(np.mean(mu_hat)),
        "e_hat_mean": float(np.mean(e_hat)),
    }


def cfperm_loco_pvalue(
    X, Y, W, *, test_indices, n_perm: int = 120, seed: int = 0, n_estimators: int = 50
) -> dict:
    """CFPerm LOCO on a block: importance = R(−block) − R(full)."""
    X = np.asarray(X, dtype=float)
    Y, W = as_1d(Y), as_1d(W).astype(int)
    idx = np.asarray(test_indices, dtype=int)
    keep = np.setdiff1d(np.arange(X.shape[1]), idx)
    r_full = r_risk(X, Y, W, seed=seed, n_estimators=n_estimators)
    r_minus = (
        r_full if keep.size == 0 else r_risk(X[:, keep], Y, W, seed=seed + 1, n_estimators=n_estimators)
    )
    imp = max(0.0, r_minus - r_full)
    rng = np.random.default_rng(seed + 17)
    null = np.zeros(n_perm)
    for b in range(n_perm):
        Wb = rng.permutation(W)
        rf = r_risk(X, Y, Wb, seed=seed + 100 + b, n_estimators=n_estimators)
        rm = (
            rf
            if keep.size == 0
            else r_risk(X[:, keep], Y, Wb, seed=seed + 200 + b, n_estimators=n_estimators)
        )
        null[b] = max(0.0, rm - rf)
    p = (1.0 + np.sum(null >= imp - 1e-15)) / (1 + n_perm)
    return {
        "method": "CFPerm_loco_block",
        "statistic": float(imp),
        "r_risk_full": float(r_full),
        "r_risk_minus": float(r_minus),
        "p_value": float(p),
        "reject": p < 0.05,
        "n_test_features": int(len(idx)),
    }


def mmd_pvalue(
    A: np.ndarray,
    B: np.ndarray,
    *,
    n_perm: int = 120,
    seed: int = 0,
    standardize: bool = True,
) -> dict:
    """Permutation MMD² two-sample test (RF-free; covariate / joint)."""
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    if standardize:
        A, B = standardize_pair(A, B)
    obs = rbf_mmd2(A, B, seed=seed)
    pooled = np.vstack([A, B])
    n0 = len(A)
    rng = np.random.default_rng(seed + 19)
    exceed = 0
    for i in range(n_perm):
        perm = rng.permutation(len(pooled))
        m = rbf_mmd2(pooled[perm[:n0]], pooled[perm[n0:]], seed=seed + 100 + i)
        if m >= obs - 1e-15:
            exceed += 1
    p = (1 + exceed) / (1 + n_perm)
    return {"method": "MMD", "statistic": float(obs), "p_value": float(p), "reject": p < 0.05}


def c2st_pvalue(
    A: np.ndarray,
    B: np.ndarray,
    *,
    n_perm: int = 120,
    seed: int = 0,
    n_estimators: int = 80,
) -> dict:
    """Classifier two-sample test with RF outcome/classifier (Lopez-Paz & Oquab)."""
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    X = np.vstack([A, B])
    W = np.concatenate([np.zeros(len(A), dtype=int), np.ones(len(B), dtype=int)])

    def _acc(Wb: np.ndarray, s: int) -> float:
        clf = rf_classifier(s, n_estimators=n_estimators)
        # 50/50 holdout accuracy
        rng = np.random.default_rng(s)
        idx = rng.permutation(len(X))
        cut = len(X) // 2
        tr, te = idx[:cut], idx[cut:]
        if len(np.unique(Wb[tr])) < 2:
            return 0.5
        clf.fit(X[tr], Wb[tr])
        return float(np.mean(clf.predict(X[te]) == Wb[te]))

    obs = _acc(W, seed)
    rng = np.random.default_rng(seed + 23)
    exceed = 0
    for i in range(n_perm):
        Wb = rng.permutation(W)
        if _acc(Wb, seed + 50 + i) >= obs - 1e-15:
            exceed += 1
    p = (1 + exceed) / (1 + n_perm)
    return {"method": "C2ST", "statistic": float(obs), "p_value": float(p), "reject": p < 0.05}


def rrperm_pvalue(
    X: np.ndarray,
    Y: np.ndarray,
    W: np.ndarray,
    *,
    n_perm: int = 120,
    seed: int = 0,
    n_estimators: int = 60,
) -> dict:
    """R-risk permutation test (RF outcome): permute W under null."""
    obs = r_risk(X, Y, W, seed=seed, n_estimators=n_estimators)
    rng = np.random.default_rng(seed + 29)
    null = np.zeros(n_perm)
    for b in range(n_perm):
        null[b] = r_risk(X, Y, rng.permutation(W), seed=seed + 100 + b, n_estimators=n_estimators)
    p = (1.0 + np.sum(null >= obs - 1e-15)) / (1 + n_perm)
    return {
        "method": "RRPerm",
        "statistic": float(obs),
        "r_risk": float(obs),
        "p_value": float(p),
        "reject": p < 0.05,
    }


def window_bundle(
    X0: np.ndarray,
    y0: np.ndarray,
    X1: np.ndarray,
    y1: np.ndarray,
    *,
    modality_indices: Mapping[str, list[int] | np.ndarray] | None = None,
    base_slice: slice | None = None,
    rich_slice: slice | None = None,
    n_perm: int = 40,
    seed: int = 0,
    n_estimators: int = 40,
    all_methods: bool = False,
) -> dict:
    """One window → RF / DR(PO) / optional ρ_PO / MMD share / CF per modality.

    If ``all_methods``, also RRPerm, C2ST(X), C2ST(X,Y), MMD(X), MMD(X,Y).
    """
    X, Y, W = batch_xy(X0, y0, X1, y1)
    rf = rfperm_pvalue(X0, y0, X1, y1, n_perm=n_perm, seed=seed, n_estimators=n_estimators)
    dr = drperm_pvalue(X, Y, W, n_perm=n_perm, seed=seed + 3, n_estimators=n_estimators)
    out: dict = {
        "RFPerm": rf,
        "DRPerm": dr,
        "po_risk": float(dr["po_risk"]),
        "power_loss": float(rf["power_loss"]),
        "r_risk": float(r_risk(X, Y, W, seed=seed + 5, n_estimators=n_estimators)),
    }
    if all_methods:
        out["RRPerm"] = rrperm_pvalue(X, Y, W, n_perm=n_perm, seed=seed + 11, n_estimators=n_estimators)
        # joint vs covariate-only two-sample
        XY0 = np.column_stack([X0, as_1d(y0)])
        XY1 = np.column_stack([X1, as_1d(y1)])
        out["MMD_X"] = mmd_pvalue(X0, X1, n_perm=n_perm, seed=seed + 13)
        out["MMD_XY"] = mmd_pvalue(XY0, XY1, n_perm=n_perm, seed=seed + 15)
        out["C2ST_X"] = c2st_pvalue(X0, X1, n_perm=n_perm, seed=seed + 17, n_estimators=n_estimators)
        out["C2ST_XY"] = c2st_pvalue(XY0, XY1, n_perm=n_perm, seed=seed + 19, n_estimators=n_estimators)
    if base_slice is not None and rich_slice is not None:
        Xb0, Xb1 = X0[:, base_slice], X1[:, base_slice]
        Xr0, Xr1 = X0[:, rich_slice], X1[:, rich_slice]
        po_b = drperm_pvalue(*batch_xy(Xb0, y0, Xb1, y1), n_perm=0, seed=seed + 7, n_estimators=n_estimators)[
            "po_risk"
        ]
        po_r = drperm_pvalue(*batch_xy(Xr0, y0, Xr1, y1), n_perm=0, seed=seed + 9, n_estimators=n_estimators)[
            "po_risk"
        ]
        out["po_base"] = float(po_b)
        out["po_rich"] = float(po_r)
        out["rho_po"] = rho_po(po_r, po_b)
    if modality_indices:
        idx0 = np.arange(len(y0))
        idx1 = np.arange(len(y0), len(y0) + len(y1))
        mmd = mmd_from_indices(X, modality_indices, idx0, idx1, seed=seed + 13)
        out["mmd"] = mmd
        out["mmd_share"] = mmd_share(mmd)
        cf = {}
        for name, cols in modality_indices.items():
            cf[name] = cfperm_loco_pvalue(
                X, Y, W,
                test_indices=np.asarray(cols, dtype=int),
                n_perm=n_perm,
                seed=seed + 17 + (hash(name) % 50),
                n_estimators=n_estimators,
            )
        out["CFPerm"] = cf
    return out


# ---------------------------------------------------------------------------
# Packaged localization (CFPerm block → coordinate subset)
# ---------------------------------------------------------------------------

def localize_subset(
    X0, X1, *,
    block_indices,
    names,
    Y0=None, Y1=None,
    n_perm: int = 100,
    seed: int = 0,
    n_estimators: int = 60,
    top_k: int = 5,
):
    """Use packaged subset_localize: block LOCO then driving feature subset."""
    from subset_localize import carrier_rank_from_windows, select_subset

    X, Y, W = batch_xy(
        X0,
        Y0 if Y0 is not None else np.zeros(len(X0)),
        X1,
        Y1 if Y1 is not None else np.zeros(len(X1)),
    )
    idx = np.asarray(block_indices, dtype=int)
    block = cfperm_loco_pvalue(
        X, Y, W, test_indices=idx, n_perm=n_perm, seed=seed, n_estimators=n_estimators,
    )
    carriers = carrier_rank_from_windows(X0, X1, idx, list(names))
    subset = select_subset(
        [{**r, "p_value": 1.0, "reject": False} for r in carriers],
        top_k=top_k, cum_share=0.85,
    )
    return {"block": block, "carriers": carriers, "subset": subset}
