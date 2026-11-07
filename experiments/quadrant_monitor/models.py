"""Outcome, propensity, PermuCATE, and the domain forest.

PermuCATE follows ``permuCATE_vimp`` in the FSDS python tree. The outcome
models μ0, μ1 and the conditional models ν_j = E[X_j | X_{-j}] are
``RandomForestRegressor``. The pseudo-outcome regression is a random forest
as well. On the randomized slice the propensity is the known 1/2, so the
pseudo-outcome does not need another fitted propensity.

The left arm is a random-forest domain classifier. Its permutation importance
is the covariate-shift VIMP (RF-domain).
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from experiments.quadrant_monitor.dgp import FEATURES, logging_prob

N_ESTIMATORS = 40
MAX_DEPTH = 6
MIN_LEAF = 8


def make_rf_regressor(seed: int, leaf: int = MIN_LEAF) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=leaf,
        max_features="sqrt",
        n_jobs=1,
        random_state=seed,
    )


def make_rf_classifier(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=1,
        random_state=seed,
    )


def po_risk(
    tau: np.ndarray,
    y: np.ndarray,
    treat: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    pi: float | np.ndarray = 0.5,
    clip: float = 1e-3,
) -> float:
    """E[(φ(Z) - τ(X))^2], with the doubly robust pseudo-outcome φ."""
    pi_clip = np.clip(np.asarray(pi, dtype=float), clip, 1.0 - clip)
    treated = np.asarray(treat).astype(int) == 1
    mu_t = np.where(treated, mu1, mu0)
    pseudo = (np.asarray(y, dtype=float) - mu_t) * (np.asarray(treat, dtype=float) - pi_clip) / (
        pi_clip * (1.0 - pi_clip)
    ) + (np.asarray(mu1, dtype=float) - np.asarray(mu0, dtype=float))
    return float(np.mean((pseudo - np.asarray(tau, dtype=float)) ** 2))


def pseudo_outcome(
    y: np.ndarray,
    treat: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    pi: float | np.ndarray = 0.5,
    clip: float = 1e-3,
) -> np.ndarray:
    pi_clip = np.clip(np.asarray(pi, dtype=float), clip, 1.0 - clip)
    treated = np.asarray(treat).astype(int) == 1
    mu_t = np.where(treated, mu1, mu0)
    return (np.asarray(y, dtype=float) - mu_t) * (np.asarray(treat, dtype=float) - pi_clip) / (
        pi_clip * (1.0 - pi_clip)
    ) + (np.asarray(mu1, dtype=float) - np.asarray(mu0, dtype=float))


class OutcomePair:
    """Two random-forest outcome regressions, Q(0, x) and Q(1, x)."""

    def __init__(self, seed: int):
        self.seed = seed
        self.mu0: RandomForestRegressor | None = None
        self.mu1: RandomForestRegressor | None = None
        self.fallback0 = 0.0
        self.fallback1 = 0.0
        self.ready = False

    def fit(self, x: np.ndarray, treat: np.ndarray, y: np.ndarray) -> bool:
        treat = np.asarray(treat).astype(int)
        y = np.asarray(y, dtype=float)
        self.fallback0 = float(y[treat == 0].mean()) if np.any(treat == 0) else float(y.mean())
        self.fallback1 = float(y[treat == 1].mean()) if np.any(treat == 1) else float(y.mean())
        ok = True
        for arm, name in ((0, "mu0"), (1, "mu1")):
            mask = treat == arm
            if int(mask.sum()) < MIN_LEAF * 2:
                setattr(self, name, None)
                ok = False
                continue
            model = make_rf_regressor(self.seed + arm)
            model.fit(x[mask], y[mask])
            setattr(self, name, model)
        self.ready = self.mu0 is not None and self.mu1 is not None
        return self.ready and ok

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(x)
        mu0 = self.mu0.predict(x) if self.mu0 is not None else np.full(n, self.fallback0)
        mu1 = self.mu1.predict(x) if self.mu1 is not None else np.full(n, self.fallback1)
        return mu0, mu1, mu1 - mu0


class PropensityModel:
    """Logistic g fit on the historical logging policy, not on the sandbox."""

    def __init__(self, seed: int):
        self.seed = seed
        self.model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=300, random_state=seed)
        self.ready = False

    def fit(self, x: np.ndarray, rng: np.random.Generator) -> None:
        prob = logging_prob(x)
        logged = rng.binomial(1, prob).astype(int)
        if len(np.unique(logged)) < 2:
            self.ready = False
            return
        self.model.fit(x, logged)
        self.ready = True

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.ready:
            return logging_prob(x)
        return self.model.predict_proba(x)[:, 1]


def deployed_score(plugin: np.ndarray, g: np.ndarray, eps: float) -> np.ndarray:
    """Plugin CATE times the IPW factor 1/2 / g_ε.

    When g is near 1/2 the factor is near 1. When g falls below ``eps`` the
    factor is capped at 1/(2 eps). Raising ``eps`` from 0.05 to 0.15 lowers
    that cap and pulls extreme scores down.
    """
    g_clip = np.clip(np.asarray(g, dtype=float), eps, 1.0 - eps)
    return np.asarray(plugin, dtype=float) * (0.5 / g_clip)


def squared_error(
    tau: np.ndarray, y: np.ndarray, treat: np.ndarray, mu0: np.ndarray, mu1: np.ndarray
) -> np.ndarray:
    phi = pseudo_outcome(y, treat, mu0, mu1, pi=0.5)
    return (phi - np.asarray(tau, dtype=float)) ** 2


def _positive_share(v: np.ndarray) -> np.ndarray:
    a = np.maximum(np.asarray(v, dtype=float), 0.0)
    total = float(a.sum())
    if total <= 1e-12:
        return np.zeros_like(a)
    return a / total


def rf_domain_vimp(
    x_ref: np.ndarray,
    x_new: np.ndarray,
    seed: int,
    n_perm: int = 6,
    max_n: int = 700,
) -> dict[str, float | np.ndarray]:
    """Permutation importance of a forest that classifies reference vs new."""
    rng = np.random.default_rng(seed)
    n = min(max_n, len(x_ref), len(x_new))
    if n < 40:
        p = x_ref.shape[1]
        return {"accuracy": float("nan"), "vimp": np.zeros(p), "share": np.zeros(p)}
    ref = x_ref[rng.choice(len(x_ref), n, replace=False)]
    new = x_new[rng.choice(len(x_new), n, replace=False)]
    x = np.vstack([ref, new])
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    order = rng.permutation(len(y))
    n_te = max(30, len(y) // 3)
    te, tr = order[:n_te], order[n_te:]
    clf = make_rf_classifier(seed)
    clf.fit(x[tr], y[tr])
    base = float(clf.score(x[te], y[te]))
    p = x.shape[1]
    vimp = np.zeros(p, dtype=float)
    x_te = x[te]
    y_te = y[te]
    for j in range(p):
        drops = np.empty(n_perm, dtype=float)
        for k in range(n_perm):
            xp = x_te.copy()
            xp[:, j] = rng.permutation(xp[:, j])
            drops[k] = base - float(clf.score(xp, y_te))
        vimp[j] = float(drops.mean())
    return {"accuracy": base, "vimp": vimp, "share": _positive_share(vimp)}


def _fit_or_mean(x: np.ndarray, target: np.ndarray, seed: int) -> tuple[RandomForestRegressor | None, float]:
    target = np.asarray(target, dtype=float).reshape(-1)
    fallback = float(np.mean(target)) if len(target) else 0.0
    if len(target) < MIN_LEAF * 2:
        return None, fallback
    model = make_rf_regressor(seed, leaf=max(4, min(MIN_LEAF, len(target) // 5)))
    model.fit(x, target)
    return model, fallback


def permucate_vimp(
    x: np.ndarray,
    y: np.ndarray,
    treat: np.ndarray,
    seed: int,
    n_perm: int = 8,
) -> dict[str, float | np.ndarray]:
    """Conditional-permutation VIMP of the DR pseudo-outcome.

    Train on half the randomized slice, score PO-risk on the other half.
    Permuting feature j adds the shuffled residual of the conditional forest
    ν_j back onto ν_j(X_{-j}), then re-predicts τ. Both ν_j and the outcome
    forests are ``RandomForestRegressor``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    treat = np.asarray(treat, dtype=int).reshape(-1)
    p = x.shape[1]
    zeros = np.zeros(p, dtype=float)
    if len(y) < 80 or len(np.unique(treat)) < 2:
        return {"po_risk": float("nan"), "vimp": zeros, "share": zeros}
    rng = np.random.default_rng(seed)
    # Stratify on the coupon so each half keeps both arms.
    arms = []
    for arm in (0, 1):
        idx = np.flatnonzero(treat == arm)
        rng.shuffle(idx)
        arms.append(idx)
    if min(len(arms[0]), len(arms[1])) < 20:
        return {"po_risk": float("nan"), "vimp": zeros, "share": zeros}
    train_parts, test_parts = [], []
    for idx in arms:
        cut = len(idx) // 2
        train_parts.append(idx[:cut])
        test_parts.append(idx[cut:])
    tr = np.concatenate(train_parts)
    te = np.concatenate(test_parts)
    x_tr, y_tr, t_tr = x[tr], y[tr], treat[tr]
    x_te, y_te, t_te = x[te], y[te], treat[te]

    mu_models = []
    mu_fallback = []
    for arm in (0, 1):
        mask = t_tr == arm
        model, fallback = _fit_or_mean(x_tr[mask], y_tr[mask], seed + 11 * arm + 3)
        mu_models.append(model)
        mu_fallback.append(fallback)

    def _mu(models_fallback, xx):
        out = []
        for model, fallback in models_fallback:
            if model is None:
                out.append(np.full(len(xx), fallback))
            else:
                out.append(model.predict(xx))
        return out[0], out[1]

    packed = list(zip(mu_models, mu_fallback))
    mu0_tr, mu1_tr = _mu(packed, x_tr)
    phi_tr = pseudo_outcome(y_tr, t_tr, mu0_tr, mu1_tr, pi=0.5)
    tau_model, tau_fallback = _fit_or_mean(x_tr, phi_tr, seed + 19)
    nu_models = []
    nu_fallback = []
    for j in range(p):
        cols = [c for c in range(p) if c != j]
        model, fallback = _fit_or_mean(x_tr[:, cols], x_tr[:, j], seed + 100 + j)
        nu_models.append(model)
        nu_fallback.append(fallback)

    mu0_te, mu1_te = _mu(packed, x_te)
    if tau_model is None:
        tau_te = np.full(len(x_te), tau_fallback)
    else:
        tau_te = tau_model.predict(x_te)
    base = po_risk(tau_te, y_te, t_te, mu0_te, mu1_te, pi=0.5)
    vimp = np.zeros(p, dtype=float)
    for j in range(p):
        cols = [c for c in range(p) if c != j]
        if nu_models[j] is None:
            nu_hat = np.full(len(x_te), nu_fallback[j])
        else:
            nu_hat = nu_models[j].predict(x_te[:, cols])
        resid = x_te[:, j] - nu_hat
        lifts = np.empty(n_perm, dtype=float)
        for k in range(n_perm):
            xp = x_te.copy()
            xp[:, j] = nu_hat + rng.permutation(resid)
            tau_p = np.full(len(xp), tau_fallback) if tau_model is None else tau_model.predict(xp)
            lifts[k] = po_risk(tau_p, y_te, t_te, mu0_te, mu1_te, pi=0.5) - base
        vimp[j] = float(lifts.mean())
    return {"po_risk": base, "vimp": vimp, "share": _positive_share(vimp)}


def top_feature(vimp: np.ndarray) -> int:
    return int(np.argmax(np.asarray(vimp, dtype=float)))


def feature_name(index: int) -> str:
    return FEATURES[int(index)]
