"""Online doubly robust pseudo-outcome score.

The pseudo-outcome is the PermuCATE OOD loss. With batch label ``T``
(0 = reference window, 1 = queried window),

```
phi = (Y - mu_T) * (T - pi) / (pi * (1 - pi)) + (mu_1 - mu_0)
```

``po_risk`` is ``mean((phi - tau(X))^2)``, the same residual used inside
PermuCATE. The online OOD score is ``mean(tau(X)^2)``: the squared batch
effect fit to that pseudo-outcome. A pure conditional shift moves ``tau``
and leaves a covariate-only active sample looking at ``X`` alone.

OOD interpretability: Shapley values of the fitted ``tau`` model attribute
which covariates drive the estimated batch effect (same idea as predictive
SHAP, applied to the pseudo-outcome learner). The three-way
``ood_vimp_wrapper`` also returns RF domain importance and MMD-LOCO.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from experiments.uplift_monitor.fsds_vimp import compute_loco_mmd_batches, po_risk


def _as_2d(x) -> np.ndarray:
    array = np.asarray(x, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return array


def _ridge(x, y, seed: int) -> Ridge:
    model = Ridge(alpha=1.0, random_state=int(seed) % (2**31 - 1))
    model.fit(_as_2d(x), np.asarray(y, dtype=float).ravel())
    return model


def _tau_regressor(kind: str, seed: int, n_estimators: int = 80):
    kind = str(kind).lower()
    if kind == "ridge":
        return Ridge(alpha=1.0, random_state=int(seed) % (2**31 - 1))
    if kind in {"rf", "forest", "random_forest"}:
        return RandomForestRegressor(
            n_estimators=int(n_estimators),
            max_depth=6,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=1,
            random_state=int(seed) % (2**31 - 1),
        )
    raise ValueError(f"unsupported tau model: {kind}")


def fit_dr_tau(
    x,
    y,
    t,
    *,
    seed: int = 0,
    clip: float = 0.01,
    tau_kind: str = "ridge",
    n_estimators: int = 80,
) -> dict | None:
    """Fit nuisances + ``tau(X) ≈ phi``. Returns ``None`` when the window is too thin."""
    x = _as_2d(x)
    y = np.asarray(y, dtype=float).ravel()
    t = np.asarray(t, dtype=int).ravel()
    n = x.shape[0]
    if n < 12 or np.unique(t).size < 2:
        return None
    control = t == 0
    treated = t == 1
    if int(control.sum()) < 6 or int(treated.sum()) < 6:
        return None
    mu0 = _ridge(x[control], y[control], seed).predict(x)
    mu1 = _ridge(x[treated], y[treated], seed + 1).predict(x)
    propensity = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=200, random_state=int(seed + 2) % (2**31 - 1)
    )
    propensity.fit(x, t)
    pi = np.clip(propensity.predict_proba(x)[:, 1], clip, 1.0 - clip)
    mu_t = np.where(treated, mu1, mu0)
    phi = (y - mu_t) * (t.astype(float) - pi) / (pi * (1.0 - pi)) + (mu1 - mu0)
    tau_model = _tau_regressor(tau_kind, seed + 3, n_estimators=n_estimators)
    tau_model.fit(x, phi)
    tau = np.asarray(tau_model.predict(x), dtype=float).ravel()
    return {
        "tau_model": tau_model,
        "tau_kind": str(tau_kind).lower(),
        "tau": tau,
        "phi": phi,
        "mu0": mu0,
        "mu1": mu1,
        "pi": pi,
        "x": x,
        "y": y,
        "t": t,
        "tau2": float(np.mean(tau**2)),
        "po_risk": float(po_risk(tau, y, t, mu0, mu1, pi, clip=clip)),
        "phi2": float(np.mean(phi**2)),
    }


def dr_scores(x, y, t, seed: int = 0, clip: float = 0.01) -> dict[str, float]:
    """Fit mu0, mu1, propensity, and tau, then return the OOD score and PO-risk."""
    fitted = fit_dr_tau(x, y, t, seed=seed, clip=clip, tau_kind="ridge")
    if fitted is None:
        return {"tau2": float("nan"), "po_risk": float("nan"), "phi2": float("nan")}
    return {
        "tau2": float(fitted["tau2"]),
        "po_risk": float(fitted["po_risk"]),
        "phi2": float(fitted["phi2"]),
    }


def _linear_shap_abs_mean(coef: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Exact interventional Shapley for a linear model: φ_j = β_j (x_j − E x_j)."""
    coef = np.asarray(coef, dtype=float).ravel()
    center = x - x.mean(axis=0, keepdims=True)
    return np.mean(np.abs(center * coef.reshape(1, -1)), axis=0)


def tau_shapley_importance(
    tau_model,
    x,
    *,
    background: np.ndarray | None = None,
    max_samples: int = 200,
) -> np.ndarray:
    """Global OOD attribution: mean |SHAP| of the pseudo-outcome model ``tau``.

    Ridge uses exact linear Shapley (``shap.LinearExplainer`` when available).
    Random forests use ``shap.TreeExplainer``.
    """
    x = _as_2d(x)
    bg = _as_2d(background) if background is not None else x
    if len(bg) > max_samples:
        rng = np.random.default_rng(0)
        bg = bg[rng.choice(len(bg), size=max_samples, replace=False)]
    x_eval = x
    if len(x_eval) > max_samples:
        rng = np.random.default_rng(1)
        x_eval = x_eval[rng.choice(len(x_eval), size=max_samples, replace=False)]

    if isinstance(tau_model, Ridge):
        try:
            import shap

            explainer = shap.LinearExplainer(tau_model, bg)
            values = np.asarray(explainer.shap_values(x_eval), dtype=float)
            if values.ndim == 1:
                values = values.reshape(-1, 1)
            return np.mean(np.abs(values), axis=0)
        except Exception:
            return _linear_shap_abs_mean(tau_model.coef_, x_eval)

    if isinstance(tau_model, RandomForestRegressor):
        try:
            import shap

            explainer = shap.TreeExplainer(tau_model, feature_perturbation="tree_path_dependent")
            values = np.asarray(explainer.shap_values(x_eval), dtype=float)
            if isinstance(values, list):
                values = values[0]
            values = np.asarray(values, dtype=float)
            if values.ndim == 1:
                values = values.reshape(-1, 1)
            return np.mean(np.abs(values), axis=0)
        except Exception:
            # Fallback: permutation importance of tau predictions vs phi proxy (yhat).
            rng = np.random.default_rng(0)
            yhat = np.asarray(tau_model.predict(x_eval), dtype=float).ravel()
            base = float(np.mean(yhat**2))
            p = x_eval.shape[1]
            out = np.zeros(p, dtype=float)
            for j in range(p):
                xp = np.array(x_eval, copy=True)
                xp[:, j] = x_eval[rng.permutation(len(x_eval)), j]
                out[j] = float(np.mean(np.asarray(tau_model.predict(xp), dtype=float) ** 2) - base)
            return np.maximum(out, 0.0)

    raise TypeError(f"unsupported tau model type: {type(tau_model)!r}")


def RF_domain_classifier(X_ref, X_new, *, seed: int = 0, n_estimators: int = 120) -> np.ndarray:
    """Random-forest importance for predicting which batch a row came from."""
    X_ref = _as_2d(X_ref)
    X_new = _as_2d(X_new)
    X = np.vstack([X_ref, X_new])
    W = np.concatenate([np.zeros(len(X_ref), dtype=int), np.ones(len(X_new), dtype=int)])
    p = X.shape[1]
    clf = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=max(2, int(np.round(np.sqrt(p)))),
        min_samples_leaf=max(1, int(np.round(np.sqrt(len(X)) // 2)) or 1),
        n_jobs=1,
        random_state=int(seed) % (2**31 - 1),
    )
    clf.fit(X, W)
    return np.asarray(clf.feature_importances_, dtype=float)


def MMD_LOCO(X_ref, X_new, *, max_n: int = 1000, seed: int = 0) -> np.ndarray:
    """Full-sample RBF MMD minus MMD after dropping feature ``j`` (from fsds_vimp)."""
    _orig, drop, _ranks = compute_loco_mmd_batches(X_ref, X_new, max_n=max_n, seed=seed)
    return np.asarray(drop, dtype=float)


def ood_vimp_wrapper(
    df1,
    df2,
    *,
    n_perm: int = 25,
    seed: int = 0,
    tau_kind: str = "rf",
    clip: float = 0.01,
    max_n_mmd: int = 1000,
) -> dict[str, np.ndarray | float]:
    """Three OOD attributions on a ref/new window pair.

    1. Shapley of the pseudo-outcome model ``tau`` (plus residual-permutation VIMP)
    2. RF domain classifier importances
    3. MMD-LOCO

    ``df1`` / ``df2`` may be ``(n, p+1)`` matrices with ``Y`` in the last column,
    or callers may pass ``X`` only via ``ood_vimp(X_ref, y_ref, X_new, y_new)``.
    """
    df1 = np.asarray(df1, dtype=float)
    df2 = np.asarray(df2, dtype=float)
    if df1.ndim != 2 or df2.ndim != 2:
        raise ValueError("df1 and df2 must be 2-d")
    if df1.shape[1] != df2.shape[1]:
        raise ValueError(f"column mismatch {df1.shape} vs {df2.shape}")
    X_ref, y_ref = df1[:, :-1], df1[:, -1]
    X_new, y_new = df2[:, :-1], df2[:, -1]
    return ood_vimp(
        X_ref,
        y_ref,
        X_new,
        y_new,
        n_perm=n_perm,
        seed=seed,
        tau_kind=tau_kind,
        clip=clip,
        max_n_mmd=max_n_mmd,
    )


def ood_vimp(
    X_ref,
    y_ref,
    X_new,
    y_new,
    *,
    n_perm: int = 25,
    seed: int = 0,
    tau_kind: str = "rf",
    clip: float = 0.01,
    max_n_mmd: int = 1000,
) -> dict[str, np.ndarray | float]:
    """Shapley-on-``tau`` + RF domain + MMD-LOCO for one ref/new pair."""
    X_ref = _as_2d(X_ref)
    X_new = _as_2d(X_new)
    y_ref = np.asarray(y_ref, dtype=float).ravel()
    y_new = np.asarray(y_new, dtype=float).ravel()
    X = np.vstack([X_ref, X_new])
    Y = np.concatenate([y_ref, y_new])
    T = np.concatenate([np.zeros(len(y_ref), dtype=int), np.ones(len(y_new), dtype=int)])
    fitted = fit_dr_tau(X, Y, T, seed=seed, clip=clip, tau_kind=tau_kind)
    p = X.shape[1]
    if fitted is None:
        nan = np.full(p, np.nan)
        return {
            "po_shapley": nan,
            "po_vimp": nan,
            "po_vimp_sd": nan,
            "po_vimp_median": nan,
            "rf_vimp": RF_domain_classifier(X_ref, X_new, seed=seed),
            "MMD_vimp": MMD_LOCO(X_ref, X_new, max_n=max_n_mmd, seed=seed),
            "tau2": float("nan"),
            "po_risk": float("nan"),
        }

    po_shapley = tau_shapley_importance(fitted["tau_model"], fitted["x"])
    # Residual-permutation VIMP on held-together sample (same spirit as permuCATE_vimp).
    rng = np.random.default_rng(seed + 11)
    tau_model = fitted["tau_model"]
    base_risk = float(fitted["po_risk"])
    po_vimp = np.zeros(p, dtype=float)
    po_vimp_sd = np.zeros(p, dtype=float)
    po_vimp_median = np.zeros(p, dtype=float)
    for j in range(p):
        drops = np.zeros(n_perm, dtype=float)
        for k in range(n_perm):
            Xp = np.array(fitted["x"], copy=True)
            Xp[:, j] = fitted["x"][rng.permutation(len(fitted["x"])), j]
            tau_perm = np.asarray(tau_model.predict(Xp), dtype=float).ravel()
            drops[k] = (
                po_risk(
                    tau_perm,
                    fitted["y"],
                    fitted["t"],
                    fitted["mu0"],
                    fitted["mu1"],
                    fitted["pi"],
                    clip=clip,
                )
                - base_risk
            )
        po_vimp[j] = float(drops.mean())
        po_vimp_sd[j] = float(drops.std())
        po_vimp_median[j] = float(np.median(drops))

    return {
        "po_shapley": np.asarray(po_shapley, dtype=float),
        "po_vimp": po_vimp,
        "po_vimp_sd": po_vimp_sd,
        "po_vimp_median": po_vimp_median,
        "rf_vimp": RF_domain_classifier(X_ref, X_new, seed=seed),
        "MMD_vimp": MMD_LOCO(X_ref, X_new, max_n=max_n_mmd, seed=seed),
        "tau2": float(fitted["tau2"]),
        "po_risk": float(fitted["po_risk"]),
    }


def resample_indices(w, rng: np.random.Generator, method: str = "bootstrap") -> np.ndarray:
    """Draw a within-batch bootstrap or a half-sample, preserving both labels."""
    w = np.asarray(w, dtype=int).ravel()
    idx0 = np.flatnonzero(w == 0)
    idx1 = np.flatnonzero(w == 1)
    if method == "subsample":
        b0 = rng.choice(idx0, size=max(2, int(round(0.5 * len(idx0)))), replace=False)
        b1 = rng.choice(idx1, size=max(2, int(round(0.5 * len(idx1)))), replace=False)
    else:
        b0 = rng.choice(idx0, size=len(idx0), replace=True)
        b1 = rng.choice(idx1, size=len(idx1), replace=True)
    return np.concatenate([b0, b1])


def prediction_permutation_importance(x, y, rng: np.random.Generator, seed: int, n_repeats: int = 1) -> np.ndarray:
    """Fit once, permute each covariate, and record the rise in squared error."""
    x = _as_2d(x)
    y = np.asarray(y, dtype=float).ravel()
    model = _ridge(x, y, seed)
    yhat = model.predict(x)
    base = float(np.mean((y - yhat) ** 2))
    n, p = x.shape
    importance = np.empty(p, dtype=float)
    for j in range(p):
        rises = np.empty(n_repeats, dtype=float)
        for r in range(n_repeats):
            xp = np.array(x, copy=True)
            xp[:, j] = x[rng.permutation(n), j]
            yhat_j = model.predict(xp)
            rises[r] = float(np.mean((y - yhat_j) ** 2) - base)
        importance[j] = float(rises.mean())
    return importance


def online_pfi(x, y, window: int, step: int, seed: int, n_repeats: int, signal_idx: np.ndarray) -> list[dict]:
    """Sliding-window permutation importance of a ridge outcome model."""
    x = _as_2d(x)
    y = np.asarray(y, dtype=float).ravel()
    n, p = x.shape
    rng = np.random.default_rng(seed)
    signal = np.asarray(signal_idx, dtype=int)
    rows = []
    for m in range(window, n + 1, step):
        sl = slice(m - window, m)
        importance = prediction_permutation_importance(x[sl], y[sl], rng, seed + m, n_repeats=n_repeats)
        top = int(np.argmax(importance))
        noise = np.ones(p, dtype=bool)
        noise[signal] = False
        rows.append(
            {
                "m": int(m),
                "top_feature": top,
                "top_is_signal": int(top in set(signal.tolist())),
                "signal_pfi": float(np.mean(importance[signal])),
                "noise_pfi": float(np.mean(importance[noise])) if np.any(noise) else float("nan"),
            }
        )
    return rows


def online_po_shapley(
    x,
    y,
    *,
    window: int,
    step: int,
    seed: int,
    signal_idx: np.ndarray,
    tau_kind: str = "rf",
    n_perm: int = 15,
) -> list[dict]:
    """Sliding-window Shapley of ``tau``: first half of the window is ref, second is new."""
    x = _as_2d(x)
    y = np.asarray(y, dtype=float).ravel()
    n, p = x.shape
    signal = np.asarray(signal_idx, dtype=int)
    mid = max(6, window // 2)
    rows = []
    for m in range(window, n + 1, step):
        sl = slice(m - window, m)
        xw, yw = x[sl], y[sl]
        out = ood_vimp(
            xw[:mid],
            yw[:mid],
            xw[mid:],
            yw[mid:],
            n_perm=n_perm,
            seed=seed + m,
            tau_kind=tau_kind,
        )
        shap = np.asarray(out["po_shapley"], dtype=float)
        if not np.all(np.isfinite(shap)):
            continue
        top = int(np.argmax(shap))
        noise = np.ones(p, dtype=bool)
        noise[signal] = False
        rows.append(
            {
                "m": int(m),
                "top_feature": top,
                "top_is_signal": int(top in set(signal.tolist())),
                "signal_shapley": float(np.mean(shap[signal])),
                "noise_shapley": float(np.mean(shap[noise])) if np.any(noise) else float("nan"),
                "tau2": float(out["tau2"]),
                "po_risk": float(out["po_risk"]),
            }
        )
    return rows
