"""Doubly robust PO-risk with random-forest nuisances.

Selection / attribution score is PO(S) = mean(tau_S(X_S)^2) on overlap-
restricted eval rows. This is intentionally NOT mean((psi - tau)^2), which
is a residual fit diagnostic used elsewhere in the repo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


@dataclass
class RFConfig:
    n_estimators: int = 60
    max_depth: int = 6
    min_samples_leaf: int = 8
    clip: float = 0.05


def _rf_clf(seed: int, cfg: RFConfig) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        max_features="sqrt",
        n_jobs=1,
        random_state=seed,
    )


def _rf_reg(seed: int, cfg: RFConfig) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        max_features="sqrt",
        n_jobs=1,
        random_state=seed,
    )


def _predict_pos(model, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 not in classes and 1.0 not in classes:
        # No positive class seen in training.
        return np.zeros(X.shape[0], dtype=float)
    idx = classes.index(1) if 1 in classes else classes.index(1.0)
    return proba[:, idx]


def pseudo_outcome(
    y: np.ndarray,
    d: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    e: np.ndarray,
    clip: float = 0.05,
) -> np.ndarray:
    e = np.clip(np.asarray(e, dtype=float).reshape(-1), clip, 1.0 - clip)
    d = np.asarray(d, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    mu0 = np.asarray(mu0, dtype=float).reshape(-1)
    mu1 = np.asarray(mu1, dtype=float).reshape(-1)
    mu_d = np.where(d >= 0.5, mu1, mu0)
    return (y - mu_d) * (d - e) / (e * (1.0 - e)) + (mu1 - mu0)


@dataclass
class FitBundle:
    e_model: object
    mu0_model: object
    mu1_model: object
    tau_model: object
    cfg: RFConfig
    cols: np.ndarray


def fit_nuisance(
    X: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    cols: Optional[Sequence[int]] = None,
    seed: int = 0,
    cfg: Optional[RFConfig] = None,
) -> FitBundle:
    cfg = cfg or RFConfig()
    cols = np.asarray(list(range(X.shape[1])) if cols is None else cols, dtype=int)
    Xs = np.asarray(X)[:, cols]
    y = np.asarray(y, dtype=float).reshape(-1)
    d = np.asarray(d, dtype=float).reshape(-1)
    d_int = (d >= 0.5).astype(int)

    e_model = _rf_clf(seed, cfg)
    e_model.fit(Xs, d_int)

    mu0_model = _rf_clf(seed + 1, cfg)
    mu1_model = _rf_clf(seed + 2, cfg)
    mask0 = d_int == 0
    mask1 = d_int == 1
    if len(np.unique(y[mask0])) < 2:
        mu0_model = ("const", float(np.mean(y[mask0])) if mask0.any() else 0.5)
    else:
        mu0_model.fit(Xs[mask0], y[mask0].astype(int))
    if len(np.unique(y[mask1])) < 2:
        mu1_model = ("const", float(np.mean(y[mask1])) if mask1.any() else 0.5)
    else:
        mu1_model.fit(Xs[mask1], y[mask1].astype(int))

    def _mu(model, Xq):
        if isinstance(model, tuple):
            return np.full(Xq.shape[0], model[1])
        return _predict_pos(model, Xq)

    e_hat = _predict_pos(e_model, Xs)
    mu0 = _mu(mu0_model, Xs)
    mu1 = _mu(mu1_model, Xs)
    psi = pseudo_outcome(y, d, mu0, mu1, e_hat, clip=cfg.clip)

    tau_model = _rf_reg(seed + 3, cfg)
    tau_model.fit(Xs, psi)

    return FitBundle(
        e_model=e_model,
        mu0_model=mu0_model,
        mu1_model=mu1_model,
        tau_model=tau_model,
        cfg=cfg,
        cols=cols,
    )


def _mu_predict(model, X: np.ndarray) -> np.ndarray:
    if isinstance(model, tuple):
        return np.full(X.shape[0], float(model[1]))
    return _predict_pos(model, X)


def predict_tau(bundle: FitBundle, X: np.ndarray) -> np.ndarray:
    Xs = np.asarray(X)[:, bundle.cols]
    return np.asarray(bundle.tau_model.predict(Xs), dtype=float).reshape(-1)


def predict_nuisance(bundle: FitBundle, X: np.ndarray):
    Xs = np.asarray(X)[:, bundle.cols]
    e = _predict_pos(bundle.e_model, Xs)
    mu0 = _mu_predict(bundle.mu0_model, Xs)
    mu1 = _mu_predict(bundle.mu1_model, Xs)
    return e, mu0, mu1


def overlap_mask(e: np.ndarray, clip: float) -> np.ndarray:
    e = np.asarray(e, dtype=float).reshape(-1)
    return (e >= clip) & (e <= 1.0 - clip)


def po_mean_tau_sq(
    bundle: FitBundle,
    X_eval: np.ndarray,
    y_eval: np.ndarray | None = None,
    d_eval: np.ndarray | None = None,
) -> dict:
    """PO(S) = mean(tau(X)^2) on overlap-restricted eval rows."""
    e, mu0, mu1 = predict_nuisance(bundle, X_eval)
    mask = overlap_mask(e, bundle.cfg.clip)
    tau = predict_tau(bundle, X_eval)
    if not np.any(mask):
        return {
            "po_mean_tau_sq": float("nan"),
            "n_eval": int(X_eval.shape[0]),
            "n_overlap": 0,
            "outside_support": int(X_eval.shape[0]),
            "mean_abs_tau": float("nan"),
        }
    tau_m = tau[mask]
    out = {
        "po_mean_tau_sq": float(np.mean(tau_m ** 2)),
        "n_eval": int(X_eval.shape[0]),
        "n_overlap": int(mask.sum()),
        "outside_support": int((~mask).sum()),
        "mean_abs_tau": float(np.mean(np.abs(tau_m))),
    }
    # Optional residual diagnostic (NOT used for selection).
    if y_eval is not None and d_eval is not None:
        psi = pseudo_outcome(y_eval, d_eval, mu0, mu1, e, clip=bundle.cfg.clip)
        out["po_squared_prediction_error"] = float(np.mean((psi[mask] - tau_m) ** 2))
    return out
