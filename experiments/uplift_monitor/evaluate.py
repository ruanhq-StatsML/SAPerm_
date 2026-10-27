"""Uplift evaluation, following the FSDS scripts step for step.

Two scores are the scripts themselves.

``r_learner_observed_risk`` is ``vimp_loco_r_risk`` up through ``observed_r_risk``
(``LOCO_vimp_r_risk.py``). Outcome model is ``rf_regressor``. Propensity model
is ``logistic_classifier``. Tau is ``_fit_tau_rlearner_weighted``. The loss is
``mean((Y_tilde - tau_hat * W_tilde)^2)``. Tau is fit inside this function.
An external uplift is not substituted.

``permucate_test_po_risk`` is ``permuCATE_vimp`` up through ``po_risk_original``.
The split is half the rows, stratified on treatment. ``mu0`` and ``mu1`` are
random forests fit on each arm. The propensity is logistic. The pseudo-outcome
is the doubly robust one in ``permuCATE_fit_nuisance``. Tau is a random forest
on that pseudo-outcome. ``po_risk`` is evaluated on the held-out half.

``deployed_po_risk`` calls the same ``po_risk`` function with a model that was
already fit. That is the monitor of a frozen uplift. It is not ``permuCATE_vimp``,
which refits.

The forest uses the ``make_rf_regression`` shape: ``max_features = round(p/3)``,
``min_samples_leaf = round(sqrt(n)/2)``. ``n_estimators`` is 40. The registry
default in ``vimp_loco_r_risk`` is 150.

What this file does not copy from the scripts:

- ``vimp_loco_r_risk`` calls ``cross_nuisance_fit`` without ``seed`` and without
  ``clip_e``. The helper then uses seed 2026 and clips at ``1e-3``, and the
  caller clips ``e_hat`` again at 0.01. The loss sees 0.01, which is the clip
  used here. The fold seed here is the ``seed`` argument, so each batch has
  its own split.
- The LOCO loop after ``observed_r_risk``, and the permutation loop after
  ``po_risk_original``, are not run on every batch.
- ``permuCATE_vimp`` calls ``train_test_split`` with no ``random_state``.
  This file passes ``random_state=seed``.
- A one-class propensity fold is replaced by the training mean. The script
  fits the logistic model anyway.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.uplift_monitor.fsds_vimp import po_risk

NTREE = 40
N_FOLDS = 5
CLIP_E = 0.01


def _rf_regressor(n: int, p: int, seed: int) -> RandomForestRegressor:
    leaf = max(1, int(round(np.sqrt(n) / 2)))
    return RandomForestRegressor(
        n_estimators=NTREE,
        max_features=max(1, int(round(p / 3))),
        min_samples_leaf=leaf,
        random_state=int(seed),
        n_jobs=1,
    )


def _logistic(seed: int) -> LogisticRegression:
    return LogisticRegression(solver="lbfgs", max_iter=200, random_state=int(seed))


def make_folds(n: int, n_folds: int = N_FOLDS, seed: int = 2026) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    return list(np.array_split(indices, n_folds))


def _predict_positive(model: LogisticRegression, x: np.ndarray) -> np.ndarray:
    probs = model.predict_proba(x)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(x))
    return probs[:, classes.index(1)]


def cross_nuisance_fit(x, y, w, n_folds: int = N_FOLDS, seed: int = 2026):
    """``cross_nuisance_fit`` with ``model_m='rf_regressor'`` and ``model_e='logistic_classifier'``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    w = np.asarray(w, dtype=int).reshape(-1)
    n, p = x.shape
    mu_hat = np.zeros(n, dtype=float)
    e_hat = np.zeros(n, dtype=float)
    for k, test_idx in enumerate(make_folds(n, n_folds, seed)):
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        y_tr = y[train_idx]
        w_tr = w[train_idx]
        if len(np.unique(y_tr)) < 2:
            mu_hat[test_idx] = float(np.mean(y_tr)) if len(y_tr) else 0.0
        else:
            outcome = _rf_regressor(len(train_idx), p, seed + k)
            outcome.fit(x[train_idx], y_tr)
            mu_hat[test_idx] = outcome.predict(x[test_idx])
        if len(np.unique(w_tr)) < 2:
            e_hat[test_idx] = float(np.mean(w_tr)) if len(w_tr) else 0.5
        else:
            propensity = _logistic(seed + 3 * k)
            propensity.fit(x[train_idx], w_tr)
            e_hat[test_idx] = _predict_positive(propensity, x[test_idx])
    return mu_hat, np.clip(e_hat, CLIP_E, 1.0 - CLIP_E)


def _fit_tau_rlearner_weighted(x, y_tilde, w_tilde, seed: int = 0, clip_wtilde: float = 1e-3):
    mask = np.abs(w_tilde) > clip_wtilde
    if int(mask.sum()) < 8:
        return None
    z = y_tilde[mask] / w_tilde[mask]
    weights = w_tilde[mask] ** 2
    tau_model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=1.0, random_state=seed)),
        ]
    )
    tau_model.fit(x[mask], z, ridge__sample_weight=weights)
    return tau_model


def r_learner_observed_risk(x, y, w, seed: int = 2026) -> float:
    """``observed_r_risk`` from ``vimp_loco_r_risk``. Tau is refit here."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    w = np.asarray(w, dtype=int).reshape(-1)
    if len(y) < N_FOLDS * 8 or len(np.unique(w)) < 2:
        return float("nan")
    mu_hat, e_hat = cross_nuisance_fit(x, y, w, seed=seed)
    y_tilde = y - mu_hat
    w_tilde = w - e_hat
    tau_model = _fit_tau_rlearner_weighted(x, y_tilde, w_tilde, seed=seed)
    if tau_model is None:
        return float("nan")
    tau_hat = tau_model.predict(x)
    return float(np.mean((y_tilde - tau_hat * w_tilde) ** 2))


def _arm_predict(model, constant: float | None, rows: np.ndarray) -> np.ndarray:
    if model is None:
        return np.full(len(rows), float(constant))
    return model.predict(rows)


def permucate_test_po_risk(x, y, w, seed: int = 0) -> float:
    """``po_risk_original`` from ``permuCATE_vimp``, before the permutation loop."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    w = np.asarray(w, dtype=int).reshape(-1)
    if len(y) < 40 or min(int((w == 0).sum()), int((w == 1).sum())) < 8:
        return float("nan")
    x_train, x_test, w_train, w_test, y_train, y_test = train_test_split(
        x, w, y, test_size=0.5, stratify=w, random_state=seed
    )

    def fit_arm(mask: np.ndarray, seed_arm: int):
        if len(np.unique(y_train[mask])) < 2:
            return None, float(np.mean(y_train[mask]))
        model = _rf_regressor(int(mask.sum()), x.shape[1], seed_arm)
        model.fit(x_train[mask], y_train[mask])
        return model, None

    mu0, c0 = fit_arm(w_train == 0, seed)
    mu1, c1 = fit_arm(w_train == 1, seed)
    mu0_train = _arm_predict(mu0, c0, x_train)
    mu1_train = _arm_predict(mu1, c1, x_train)
    mu0_test = _arm_predict(mu0, c0, x_test)
    mu1_test = _arm_predict(mu1, c1, x_test)
    propensity = _logistic(seed + 2)
    propensity.fit(x_train, w_train)
    pi_train = np.clip(_predict_positive(propensity, x_train), CLIP_E, 1.0 - CLIP_E)
    pi_test = np.clip(_predict_positive(propensity, x_test), CLIP_E, 1.0 - CLIP_E)
    treated = w_train == 1
    pseudo = (mu1_train - mu0_train) + (y_train - np.where(treated, mu1_train, mu0_train)) * (
        w_train - pi_train
    ) / (pi_train * (1.0 - pi_train))
    tau_model = _rf_regressor(len(y_train), x.shape[1], seed + 3)
    tau_model.fit(x_train, pseudo)
    tau_test = tau_model.predict(x_test)
    return po_risk(tau_test, y_test, w_test, mu0_test, mu1_test, pi_test)


def deployed_po_risk(tau, y, treat, mu0, mu1, pi: float = 0.5) -> float:
    """``po_risk`` with a model that was fit earlier. Not the PermuCATE refit."""
    return po_risk(tau, y, treat, mu0, mu1, np.full(len(y), pi), clip=1e-3)
