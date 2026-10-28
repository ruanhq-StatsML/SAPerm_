"""PermuCATE variable importance for the concept-drift arm.

Arithmetic follows ``permuCATE_vimp.py`` in
ruanhq-StatsML/FSDS-FeatureSelection_for_DistributionShift.

The domain indicator W marks the existing batch versus the new batch. It is
not the incentive. The outcome regressions (mu0, mu1), the pseudo-outcome
regression tau, and every conditional residual model are
``RandomForestRegressor``. The domain propensity is logistic, which is the
script default for a binary domain label.

phi(Z) = (Y - mu_W) * (W - pi) / (pi * (1 - pi)) + (mu1 - mu0)

VIMP_j is the average increase in E[(phi - tau(X))^2] after the residual of
X_j given X_{-j} is permuted on the held-out half.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def po_risk(tau, Y, T, mu0, mu1, pi, clip: float = 1e-3) -> float:
    """Doubly robust pseudo-outcome risk, E[(phi(Z) - tau(X))^2]."""
    tau = np.asarray(tau, dtype=float).ravel()
    outcome = np.asarray(Y, dtype=float).ravel()
    domain = np.asarray(T, dtype=float).ravel()
    mu0 = np.asarray(mu0, dtype=float).ravel()
    mu1 = np.asarray(mu1, dtype=float).ravel()
    pi_clip = np.clip(np.asarray(pi, dtype=float).ravel(), clip, 1.0 - clip)
    mu_w = np.where(domain == 1.0, mu1, mu0)
    pseudo = (outcome - mu_w) * (domain - pi_clip) / (pi_clip * (1.0 - pi_clip)) + (mu1 - mu0)
    return float(np.mean((pseudo - tau) ** 2))


def make_random_forest(n: int, p: int, n_estimators: int, seed: int) -> RandomForestRegressor:
    """Manuscript regression factory: max_features = p/3, leaf = sqrt(n)/2."""
    return RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_features=max(1, int(round(p / 3.0))),
        min_samples_leaf=max(2, int(round(np.sqrt(n) / 2.0))),
        random_state=int(seed),
        n_jobs=1,
    )


def _fit_forest(X, y, n_estimators: int, seed: int) -> RandomForestRegressor:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    model = make_random_forest(X.shape[0], X.shape[1], n_estimators, seed)
    model.fit(X, y)
    return model


def fit_nuisance(X, Y, W, n_estimators: int, seed: int, clip_e: float = 0.05) -> dict:
    """Fit outcome, domain propensity, tau, and one conditional forest per column.

    Outcome and conditional models are random-forest regressors, including
    columns that happen to be binary. That is the required PermuCATE setting.
    """
    X = np.asarray(X, dtype=float).copy()
    outcome = np.asarray(Y, dtype=float).ravel()
    domain = np.asarray(W, dtype=int).ravel()
    treated = domain == 1
    control = ~treated
    if int(treated.sum()) < 8 or int(control.sum()) < 8:
        raise ValueError("each domain needs at least 8 rows to fit PermuCATE")
    mu0 = _fit_forest(X[control], outcome[control], n_estimators, seed)
    mu1 = _fit_forest(X[treated], outcome[treated], n_estimators, seed + 1)
    mu0_hat = mu0.predict(X)
    mu1_hat = mu1.predict(X)
    propensity = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logit", LogisticRegression(C=1.0, solver="lbfgs", max_iter=250)),
        ]
    )
    propensity.fit(X, domain)
    pi = np.clip(propensity.predict_proba(X)[:, 1], clip_e, 1.0 - clip_e)
    pseudo = (outcome - np.where(treated, mu1_hat, mu0_hat)) * (domain - pi) / (pi * (1.0 - pi))
    pseudo = pseudo + (mu1_hat - mu0_hat)
    tau = _fit_forest(X, pseudo, n_estimators, seed + 2)
    nu_models = []
    p = X.shape[1]
    for j in range(p):
        others = np.delete(X, j, axis=1)
        nu_models.append(_fit_forest(others, X[:, j], n_estimators, seed + 100 + j))
    return {
        "mu0": mu0,
        "mu1": mu1,
        "tau": tau,
        "nu": nu_models,
        "propensity": propensity,
        "clip_e": clip_e,
    }


def permu_cate_vimp(
    X: np.ndarray,
    Y: np.ndarray,
    W: np.ndarray,
    n_perm: int = 6,
    n_estimators: int = 40,
    test_size: float = 0.5,
    seed: int = 0,
    clip_e: float = 0.05,
) -> np.ndarray:
    """Conditional-permutation VIMP. Higher means a larger PO-risk increase."""
    X = np.asarray(X, dtype=float)
    outcome = np.asarray(Y, dtype=float).ravel()
    domain = np.asarray(W, dtype=int).ravel()
    if len(np.unique(domain)) < 2:
        return np.zeros(X.shape[1], dtype=float)
    X_train, X_test, W_train, W_test, Y_train, Y_test = train_test_split(
        X,
        domain,
        outcome,
        test_size=test_size,
        stratify=domain,
        random_state=seed,
    )
    fit = fit_nuisance(X_train, Y_train, W_train, n_estimators=n_estimators, seed=seed, clip_e=clip_e)
    tau_hat = fit["tau"].predict(X_test)
    mu0_test = fit["mu0"].predict(X_test)
    mu1_test = fit["mu1"].predict(X_test)
    pi_test = np.clip(fit["propensity"].predict_proba(X_test)[:, 1], clip_e, 1.0 - clip_e)
    baseline = po_risk(tau_hat, Y_test, W_test, mu0_test, mu1_test, pi_test, clip=clip_e)
    rng = np.random.default_rng(seed)
    vimp = np.zeros(X.shape[1], dtype=float)
    for j, nu in enumerate(fit["nu"]):
        others = np.delete(X_test, j, axis=1)
        nu_hat = nu.predict(others)
        residual = X_test[:, j] - nu_hat
        increases = np.empty(n_perm, dtype=float)
        for k in range(n_perm):
            permuted = X_test.copy()
            permuted[:, j] = nu_hat + rng.permutation(residual)
            tau_perm = fit["tau"].predict(permuted)
            increases[k] = po_risk(tau_perm, Y_test, W_test, mu0_test, mu1_test, pi_test, clip=clip_e) - baseline
        vimp[j] = float(increases.mean())
    return vimp
