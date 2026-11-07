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
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from experiments.uplift_monitor.fsds_vimp import po_risk


def _as_2d(x) -> np.ndarray:
    array = np.asarray(x, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return array


def _ridge(x, y, seed: int) -> Ridge:
    model = Ridge(alpha=1.0, random_state=int(seed) % (2**31 - 1))
    model.fit(_as_2d(x), np.asarray(y, dtype=float).ravel())
    return model


def dr_scores(x, y, t, seed: int = 0, clip: float = 0.01) -> dict[str, float]:
    """Fit mu0, mu1, propensity, and tau, then return the OOD score and PO-risk."""
    x = _as_2d(x)
    y = np.asarray(y, dtype=float).ravel()
    t = np.asarray(t, dtype=int).ravel()
    n = x.shape[0]
    nan = {"tau2": float("nan"), "po_risk": float("nan"), "phi2": float("nan")}
    if n < 12 or np.unique(t).size < 2:
        return nan
    control = t == 0
    treated = t == 1
    if int(control.sum()) < 6 or int(treated.sum()) < 6:
        return nan
    mu0 = _ridge(x[control], y[control], seed).predict(x)
    mu1 = _ridge(x[treated], y[treated], seed + 1).predict(x)
    propensity = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=200, random_state=int(seed + 2) % (2**31 - 1)
    )
    propensity.fit(x, t)
    pi = np.clip(propensity.predict_proba(x)[:, 1], clip, 1.0 - clip)
    mu_t = np.where(treated, mu1, mu0)
    phi = (y - mu_t) * (t.astype(float) - pi) / (pi * (1.0 - pi)) + (mu1 - mu0)
    tau_model = _ridge(x, phi, seed + 3)
    tau = tau_model.predict(x)
    return {
        "tau2": float(np.mean(tau**2)),
        "po_risk": float(po_risk(tau, y, t, mu0, mu1, pi, clip=clip)),
        "phi2": float(np.mean(phi**2)),
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
