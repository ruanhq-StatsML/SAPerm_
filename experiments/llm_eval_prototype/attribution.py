"""Block LOCO attribution and permutation detection for PO(S)=mean(tau^2)."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from experiments.llm_eval_prototype.dgp import BLOCKS, brier
from experiments.llm_eval_prototype.po_risk import (
    RFConfig,
    fit_nuisance,
    po_mean_tau_sq,
)


def block_shift_scores(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    d_fit: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    d_eval: np.ndarray,
    blocks: Optional[Dict[str, Sequence[int]]] = None,
    seed: int = 0,
    cfg: Optional[RFConfig] = None,
) -> dict:
    blocks = blocks or BLOCKS
    cfg = cfg or RFConfig()
    all_cols = list(range(X_fit.shape[1]))

    full = fit_nuisance(X_fit, y_fit, d_fit, cols=all_cols, seed=seed, cfg=cfg)
    full_po = po_mean_tau_sq(full, X_eval, y_eval, d_eval)
    po_all = full_po["po_mean_tau_sq"]

    scores: Dict[str, float] = {}
    details = {}
    for name, idxs in blocks.items():
        keep = [j for j in all_cols if j not in set(idxs)]
        if not keep:
            scores[name] = float("nan")
            continue
        red = fit_nuisance(X_fit, y_fit, d_fit, cols=keep, seed=seed + 11, cfg=cfg)
        red_po = po_mean_tau_sq(red, X_eval, y_eval, d_eval)
        scores[name] = float(po_all - red_po["po_mean_tau_sq"])
        details[name] = {
            "po_without": red_po["po_mean_tau_sq"],
            "outside_support": red_po["outside_support"],
        }

    ranked = sorted(scores.items(), key=lambda kv: (-(kv[1] if np.isfinite(kv[1]) else -1e18), kv[0]))
    top = ranked[0][0] if ranked else None
    return {
        "po_all": po_all,
        "shift_scores": scores,
        "top_block": top,
        "details": details,
        "outside_support_full": full_po["outside_support"],
        "n_overlap": full_po["n_overlap"],
        "po_squared_prediction_error": full_po.get("po_squared_prediction_error"),
    }


def max_shift_statistic(scores: Dict[str, float]) -> float:
    vals = [v for v in scores.values() if np.isfinite(v)]
    return float(max(vals)) if vals else float("nan")


def permute_max_p(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    d_fit: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    d_eval: np.ndarray,
    n_perm: int = 49,
    seed: int = 0,
    cfg: Optional[RFConfig] = None,
    blocks: Optional[Dict[str, Sequence[int]]] = None,
) -> dict:
    """Permute domain labels on the fit set; recompute max shift_score."""
    obs = block_shift_scores(
        X_fit, y_fit, d_fit, X_eval, y_eval, d_eval, blocks=blocks, seed=seed, cfg=cfg
    )
    observed = max_shift_statistic(obs["shift_scores"])
    rng = np.random.default_rng(seed + 101)
    d0 = np.asarray(d_fit, dtype=float).copy()
    ge = 0
    for b in range(n_perm):
        d_perm = rng.permutation(d0)
        # Keep eval domain labels fixed; only refit under permuted fit domains.
        try:
            perm = block_shift_scores(
                X_fit,
                y_fit,
                d_perm,
                X_eval,
                y_eval,
                d_eval,
                blocks=blocks,
                seed=seed + 1000 + b,
                cfg=cfg,
            )
            stat = max_shift_statistic(perm["shift_scores"])
        except Exception:
            stat = float("nan")
        if np.isfinite(stat) and np.isfinite(observed) and stat >= observed - 1e-15:
            ge += 1
        elif not np.isfinite(observed):
            ge += 1
    p = (1.0 + ge) / (1.0 + n_perm)
    return {"observed_max_shift": observed, "p_value": float(p), "attribution": obs}


def median_heuristic_gamma(X: np.ndarray, max_n: int = 200, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.choice(n, size=min(max_n, n), replace=False) if n > max_n else np.arange(n)
    Z = X[idx]
    d2 = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=-1)
    tri = d2[np.triu_indices(Z.shape[0], k=1)]
    med = float(np.median(tri)) if tri.size else 1.0
    if med <= 1e-12:
        med = 1.0
    return 1.0 / med


def block_mmd_p(
    X0: np.ndarray,
    X1: np.ndarray,
    n_perm: int = 49,
    seed: int = 0,
) -> dict:
    """RBF MMD^2 with permutation p-value on a feature submatrix."""
    rng = np.random.default_rng(seed)
    n0, n1 = X0.shape[0], X1.shape[0]
    # Cap for smoke speed.
    m = min(120, n0, n1)
    i0 = rng.choice(n0, size=m, replace=False)
    i1 = rng.choice(n1, size=m, replace=False)
    A = X0[i0]
    B = X1[i1]
    gamma = median_heuristic_gamma(np.vstack([A, B]), seed=seed)

    def mmd2(U, V):
        Kuu = rbf_kernel(U, U, gamma=gamma)
        Kvv = rbf_kernel(V, V, gamma=gamma)
        Kuv = rbf_kernel(U, V, gamma=gamma)
        np.fill_diagonal(Kuu, 0.0)
        np.fill_diagonal(Kvv, 0.0)
        return float(
            Kuu.sum() / (m * (m - 1))
            + Kvv.sum() / (m * (m - 1))
            - 2.0 * Kuv.mean()
        )

    observed = mmd2(A, B)
    pool = np.vstack([A, B])
    ge = 0
    for b in range(n_perm):
        perm = rng.permutation(2 * m)
        U = pool[perm[:m]]
        V = pool[perm[m:]]
        if mmd2(U, V) >= observed - 1e-15:
            ge += 1
    p = (1.0 + ge) / (1.0 + n_perm)
    return {"mmd2": observed, "p_value": float(p)}


def predictive_risk_gap(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    d_fit: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    d_eval: np.ndarray,
    seed: int = 0,
    cfg: Optional[RFConfig] = None,
) -> dict:
    """Source vs target Brier of an RF fit on source-only rows (reference judge)."""
    cfg = cfg or RFConfig()
    d_fit = np.asarray(d_fit)
    mask_s = d_fit < 0.5
    from experiments.llm_eval_prototype.po_risk import _rf_clf, _predict_pos

    model = _rf_clf(seed, cfg)
    ys = y_fit[mask_s].astype(int)
    if len(np.unique(ys)) < 2:
        p_const = float(np.mean(ys)) if ys.size else 0.5
        p_eval = np.full(X_eval.shape[0], p_const)
    else:
        model.fit(X_fit[mask_s], ys)
        p_eval = _predict_pos(model, X_eval)
    d_eval = np.asarray(d_eval)
    src = d_eval < 0.5
    tgt = ~src
    return {
        "brier_source": brier(y_eval[src], p_eval[src]) if src.any() else float("nan"),
        "brier_target": brier(y_eval[tgt], p_eval[tgt]) if tgt.any() else float("nan"),
        "brier_gap": (
            brier(y_eval[tgt], p_eval[tgt]) - brier(y_eval[src], p_eval[src])
            if src.any() and tgt.any()
            else float("nan")
        ),
    }
