"""Feature scores from the FSDS benchmark, pointed at a pure-noise design.

The calculations follow the public FSDS tree:

- PermuCATE and ``po_risk`` from ``python/permuCATE_vimp.py``
- LOCO R-risk from ``python/LOCO_vimp_r_risk.py``
- causal-forest importance from ``python/grf_vimp_causalForest.py``
- LOCO-MMD from ``python/VIMP_mmd_benchmark.py``
- LOCO-HSIC from ``python/VIMP_hsic_benchmark.py``
- RF domain importance and delta-SHAP from ``python/benchmark_methods.py``

PermuCATE's outcome regressions and the conditional model ``X_j | X_{-j}``
are random forests, as is the pseudo-outcome regression. The propensity
model is a random forest classifier, matching ``benchmark_whole_feature_selection``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ScoreConfig:
    n_estimators: int = 80
    max_depth: int = 5
    min_samples_leaf: int = 1
    n_perm: int = 20
    n_folds: int = 5
    shap_estimators: int = 80
    grf_estimators: int = 50
    max_kernel_n: int = 400
    hsic_sigma: float = 1.0


def selection_auc(scores: np.ndarray, support: np.ndarray) -> float:
    """Selection AUC used by ``metrics`` in ``benchmark_methods.py``.

    A constant score has no ranking. That case is the coin-flip value 0.5,
    which ``roc_auc_score`` would otherwise reject.
    """
    sc = np.nan_to_num(np.asarray(scores, dtype=float).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    membership = np.zeros(sc.size, dtype=int)
    membership[np.asarray(support, dtype=int)] = 1
    if membership.sum() == 0 or membership.sum() == membership.size:
        return float("nan")
    if np.allclose(sc, sc[0]):
        return 0.5
    return float(roc_auc_score(membership, sc))


def _rf_reg(seed: int, cfg: ScoreConfig) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        n_jobs=1,
        random_state=int(seed),
    )


def _rf_clf(seed: int, cfg: ScoreConfig) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        n_jobs=1,
        random_state=int(seed),
    )


def po_risk(tau, y, t, mu0, mu1, pi, clip: float = 1e-3) -> float:
    """Doubly robust pseudo-outcome risk, ``E[(phi(Z) - tau(X))^2]``."""
    pi_clip = np.clip(np.asarray(pi, dtype=float), clip, 1.0 - clip)
    treated = np.asarray(t).astype(int) == 1
    mu_t = np.where(treated, mu1, mu0)
    pseudo = (np.asarray(y, dtype=float) - mu_t) * (np.asarray(t, dtype=float) - pi_clip) / (
        pi_clip * (1.0 - pi_clip)
    ) + (np.asarray(mu1, dtype=float) - np.asarray(mu0, dtype=float))
    return float(np.mean((pseudo - np.asarray(tau, dtype=float)) ** 2))


def _predict_propensity(model: RandomForestClassifier, x: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(x)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(x), dtype=float)
    return proba[:, classes.index(1)]


def permucate_vimp(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    cfg: ScoreConfig,
    seed: int,
) -> np.ndarray:
    """Conditional-permutation CATE importance.

    Outcome models ``mu_0, mu_1``, the pseudo-outcome model, and every
    conditional model ``X_j | X_{-j}`` are random forests.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    w = np.asarray(w, dtype=int).ravel()
    x_train, x_test, y_train, y_test, w_train, w_test = train_test_split(
        x, y, w, test_size=0.5, stratify=w, random_state=seed
    )
    p = x.shape[1]
    rng = np.random.default_rng(seed)

    mu0 = _rf_reg(seed + 11, cfg)
    mu1 = _rf_reg(seed + 12, cfg)
    mu0.fit(x_train[w_train == 0], y_train[w_train == 0])
    mu1.fit(x_train[w_train == 1], y_train[w_train == 1])
    mu0_tr = mu0.predict(x_train)
    mu1_tr = mu1.predict(x_train)

    propensity = _rf_clf(seed + 13, cfg)
    propensity.fit(x_train, w_train)
    pi_tr = np.clip(_predict_propensity(propensity, x_train), 0.01, 0.99)
    pseudo = (mu1_tr - mu0_tr) + (y_train - np.where(w_train == 1, mu1_tr, mu0_tr)) * (
        w_train - pi_tr
    ) / (pi_tr * (1.0 - pi_tr))
    tau = _rf_reg(seed + 14, cfg)
    tau.fit(x_train, pseudo)

    mu0_te = mu0.predict(x_test)
    mu1_te = mu1.predict(x_test)
    pi_te = np.clip(_predict_propensity(propensity, x_test), 0.01, 0.99)
    tau_te = tau.predict(x_test)
    base = po_risk(tau_te, y_test, w_test, mu0_te, mu1_te, pi_te)

    vimp = np.zeros(p, dtype=float)
    for j in range(p):
        cols = np.delete(np.arange(p), j)
        nu = _rf_reg(seed + 100 + j, cfg)
        nu.fit(x_train[:, cols], x_train[:, j])
        nu_hat = nu.predict(x_test[:, cols])
        residual = x_test[:, j] - nu_hat
        lifts = np.empty(cfg.n_perm, dtype=float)
        for k in range(cfg.n_perm):
            x_perm = x_test.copy()
            x_perm[:, j] = nu_hat + rng.permutation(residual)
            lifts[k] = po_risk(tau.predict(x_perm), y_test, w_test, mu0_te, mu1_te, pi_te) - base
        vimp[j] = float(lifts.mean())
    return vimp


def _fit_tau_rlearner(x, y_tilde, w_tilde, seed: int, clip: float = 1e-3):
    mask = np.abs(w_tilde) > clip
    if int(mask.sum()) < 8:
        return None
    z = y_tilde[mask] / w_tilde[mask]
    weights = w_tilde[mask] ** 2
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )
    model.fit(x[mask], z, ridge__sample_weight=weights)
    return model


def _cross_nuisance(x, y, w, cfg: ScoreConfig, seed: int):
    n = len(y)
    folds = np.array_split(np.random.default_rng(seed).permutation(n), cfg.n_folds)
    mu = np.zeros(n, dtype=float)
    propensity = np.zeros(n, dtype=float)
    for k, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        outcome = _rf_reg(seed + k, cfg)
        outcome.fit(x[train_idx], y[train_idx])
        mu[test_idx] = outcome.predict(x[test_idx])
        clf = _rf_clf(seed + 3 * k + 17, cfg)
        clf.fit(x[train_idx], w[train_idx])
        propensity[test_idx] = _predict_propensity(clf, x[test_idx])
    return mu, np.clip(propensity, 0.01, 0.99)


def _r_risk(x, y, w, cfg: ScoreConfig, seed: int) -> float:
    mu, propensity = _cross_nuisance(x, y, w, cfg, seed)
    y_tilde = y - mu
    w_tilde = w - propensity
    tau = _fit_tau_rlearner(x, y_tilde, w_tilde, seed)
    if tau is None:
        return float("nan")
    return float(np.mean((y_tilde - tau.predict(x) * w_tilde) ** 2))


def loco_r_risk_vimp(x, y, w, cfg: ScoreConfig, seed: int) -> np.ndarray:
    """Leave-one-covariate R-learner loss. The outcome model is a random forest."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    w = np.asarray(w, dtype=int).ravel()
    p = x.shape[1]
    base = _r_risk(x, y, w, cfg, seed)
    vimp = np.zeros(p, dtype=float)
    for j in range(p):
        x_loo = np.delete(x, j, axis=1)
        vimp[j] = _r_risk(x_loo, y, w, cfg, seed + 1 + j) - base
    return vimp


def grf_vimp(x, y, w, cfg: ScoreConfig, seed: int) -> np.ndarray:
    """Causal-forest split importance. Outcome model is a random forest."""
    from econml.dml import CausalForestDML

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    model_y = RandomForestRegressor(
        n_estimators=cfg.grf_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=max(cfg.min_samples_leaf, 5),
        n_jobs=1,
        random_state=seed,
    )
    model_t = LogisticRegression(max_iter=400, random_state=seed)
    forest = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        n_estimators=cfg.grf_estimators,
        min_samples_leaf=max(cfg.min_samples_leaf, 5),
        random_state=seed,
        discrete_treatment=True,
        cv=3,
        inference=False,
    )
    forest.fit(y, w, X=x)
    return np.asarray(forest.feature_importances_, dtype=float)


def _subsample(x, max_n, rng):
    x = np.asarray(x, dtype=float)
    if len(x) <= max_n:
        return x
    return x[rng.choice(len(x), max_n, replace=False)]


def _median_bandwidth(z):
    if len(z) < 2:
        return 1.0
    diff = z[:, None, :] - z[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    positive = dist[dist > 0]
    med = float(np.median(positive)) if positive.size else 1.0
    return med if med > 0 else 1.0


def _rbf_mmd2(x, y, gamma):
    n, m = len(x), len(y)
    if n < 2 or m < 2:
        return 0.0

    def sq(a, b):
        return np.sum(a * a, axis=1, keepdims=True) + np.sum(b * b, axis=1) - 2.0 * (a @ b.T)

    kxx = np.exp(-gamma * sq(x, x))
    kyy = np.exp(-gamma * sq(y, y))
    kxy = np.exp(-gamma * sq(x, y))
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    stat = kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2.0 * kxy.mean()
    return float(max(stat, 0.0))


def loco_mmd_vimp(x0, x1, cfg: ScoreConfig, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = _subsample(x0, cfg.max_kernel_n, rng)
    b = _subsample(x1, cfg.max_kernel_n, rng)
    p = a.shape[1]

    def mmd(u, v):
        med = _median_bandwidth(np.vstack([u, v]))
        gamma = 1.0 / (2.0 * med * med + 1e-12)
        return _rbf_mmd2(u, v, gamma)

    base = mmd(a, b)
    vimp = np.zeros(p, dtype=float)
    for j in range(p):
        kept = np.delete(np.arange(p), j)
        vimp[j] = base - mmd(a[:, kept], b[:, kept])
    return vimp


def _rbf(a, b, sigma):
    dist = np.sum(a * a, axis=1, keepdims=True) + np.sum(b * b, axis=1) - 2.0 * (a @ b.T)
    gamma = 1.0 / (2.0 * float(sigma) ** 2 + 1e-12)
    return np.exp(-gamma * np.maximum(dist, 0.0))


def _hsic(x, w, sigma):
    n = len(x)
    if n < 3:
        return 0.0
    eye = np.eye(n) - np.ones((n, n)) / n
    k = eye @ _rbf(x, x, sigma) @ eye
    ww = np.asarray(w, dtype=float).reshape(-1, 1)
    l = eye @ _rbf(ww, ww, sigma) @ eye
    return float(np.trace(k @ l) / ((n - 1) ** 2))


def loco_hsic_vimp(x0, x1, cfg: ScoreConfig, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = _subsample(x0, cfg.max_kernel_n, rng)
    b = _subsample(x1, cfg.max_kernel_n, rng)
    x = np.vstack([a, b])
    w = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
    p = x.shape[1]
    base = _hsic(x, w, cfg.hsic_sigma)
    vimp = np.zeros(p, dtype=float)
    for j in range(p):
        vimp[j] = base - _hsic(np.delete(x, j, axis=1), w, cfg.hsic_sigma)
    return vimp


def rf_domain_vimp(x0, x1, cfg: ScoreConfig, seed: int) -> np.ndarray:
    x = np.vstack([x0, x1])
    w = np.concatenate([np.zeros(len(x0)), np.ones(len(x1))]).astype(int)
    p = x.shape[1]
    clf = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=max(2, int(np.round(np.sqrt(p)))),
        min_samples_leaf=max(1, int(np.round(np.sqrt(len(x)) / 2.0))),
        n_jobs=1,
        random_state=seed,
    )
    clf.fit(x, w)
    return np.asarray(clf.feature_importances_, dtype=float)


def delta_shap_vimp(x0, y0, x1, y1, cfg: ScoreConfig, seed: int) -> np.ndarray:
    """Absolute change in mean TreeSHAP between the two batches."""
    import shap

    p = x0.shape[1]
    n = len(y0) + len(y1)
    kw = dict(
        max_depth=max(2, int(round(p / 3.0))),
        n_estimators=cfg.shap_estimators,
        min_samples_leaf=max(1, int(round(np.sqrt(n) / 2.0))),
        n_jobs=1,
    )
    m0 = RandomForestRegressor(**kw, random_state=seed)
    m1 = RandomForestRegressor(**kw, random_state=seed + 1)
    tr0, te0, ytr0, _ = train_test_split(x0, y0, test_size=0.3, random_state=seed + 10)
    tr1, te1, ytr1, _ = train_test_split(x1, y1, test_size=0.3, random_state=seed + 11)
    m0.fit(tr0, ytr0)
    m1.fit(tr1, ytr1)
    s0 = np.asarray(shap.TreeExplainer(m0).shap_values(te0), dtype=float)
    s1 = np.asarray(shap.TreeExplainer(m1).shap_values(te1), dtype=float)
    return np.abs(s1.mean(axis=0) - s0.mean(axis=0))


METHOD_NAMES = (
    "permucate",
    "loco_r_risk",
    "grf",
    "loco_mmd",
    "loco_hsic",
    "rf_domain",
    "delta_shap",
)


def score_all(x0, y0, x1, y1, cfg: ScoreConfig, seed: int) -> dict[str, np.ndarray]:
    x0 = np.asarray(x0, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    y0 = np.asarray(y0, dtype=float).ravel()
    y1 = np.asarray(y1, dtype=float).ravel()
    x = np.vstack([x0, x1])
    y = np.concatenate([y0, y1])
    w = np.concatenate([np.zeros(len(y0)), np.ones(len(y1))]).astype(int)
    return {
        "permucate": permucate_vimp(x, y, w, cfg, seed + 4),
        "loco_r_risk": loco_r_risk_vimp(x, y, w, cfg, seed + 4),
        "grf": grf_vimp(x, y, w, cfg, seed + 4),
        "loco_mmd": loco_mmd_vimp(x0, x1, cfg, seed),
        "loco_hsic": loco_hsic_vimp(x0, x1, cfg, seed + 1),
        "rf_domain": rf_domain_vimp(x0, x1, cfg, seed),
        "delta_shap": delta_shap_vimp(x0, y0, x1, y1, cfg, seed),
    }
