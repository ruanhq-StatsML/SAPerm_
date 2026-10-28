"""Stabilized longitudinal TMLE with a two-model Super Learner.

The target is E[Y^{a=1}] - E[Y^{a=0}], the contrast between always offering
the incentive and never offering it. Backward recursion follows Bang and
Robins: at time t the outcome regression is fluctuated with the current-step
clever covariate H = 1{A_t = a*_t} / g_t(a*_t | history). The cumulative
product clever covariate is the efficient one and is also the one that
explodes when a channel shift pushes propensity onto the boundary, so the
point estimate uses the current-step update. Propensity truncation is the
knob the covariate arm turns.

The Super Learner library is ridge plus a random forest. Cross-validated
squared error chooses the convex weights. A concept-drift penalty multiplies
the score of any candidate that puts weight on ridge. That penalty is a
prior from PermuCATE, not a second cross-validation fold.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def design_q(L: np.ndarray, A: np.ndarray, t: int) -> np.ndarray:
    """History through A_t. Linear blip terms A*redemption and A*severity are included.

    The quadratic redemption blip is not included, so ridge is correctly
    specified before concept drift and misspecified after it.
    """
    n = L.shape[0]
    action = A[:, t : t + 1].astype(float)
    redemption = L[:, t, 2:3]
    severity = L[:, t, 0:1]
    parts = [L[:, t, :], action, action * redemption, action * severity]
    if t == 0:
        parts.extend([np.zeros((n, 3)), np.zeros((n, 1)), np.zeros((n, 1))])
    else:
        parts.extend(
            [
                L[:, t - 1, :3],
                A[:, t - 1 : t].astype(float),
                A[:, :t].mean(axis=1, keepdims=True),
            ]
        )
    return np.hstack(parts)


def design_g(L: np.ndarray, A: np.ndarray, t: int) -> np.ndarray:
    n = L.shape[0]
    if t == 0:
        lag = np.zeros((n, 2))
    else:
        lag = np.column_stack([A[:, t - 1].astype(float), L[:, t - 1, 0]])
    return np.hstack([L[:, t, :], lag])


def _forest(n_estimators: int, seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_features="sqrt",
        min_samples_leaf=5,
        max_depth=12,
        random_state=int(seed),
        n_jobs=1,
    )


def _ridge() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )


def _fit_member(name: str, X, y, n_estimators: int, seed: int):
    model = _ridge() if name == "ridge" else _forest(n_estimators, seed)
    model.fit(X, y)
    return model


def select_weights(
    X: np.ndarray,
    y: np.ndarray,
    ridge_penalty: float,
    n_estimators: int,
    seed: int,
    n_folds: int = 2,
) -> tuple[dict[str, float], dict[str, float]]:
    """Convex Super Learner weights and the unpenalized weights.

    The penalized score is CV mean squared error times
    (1 + ridge_penalty * ridge weight). The penalty is zero when PermuCATE
    does not clear its null.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    names = ("ridge", "forest")
    n = len(y)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), n_folds)
    oof = {name: np.zeros(n, dtype=float) for name in names}
    for fold_id, test_idx in enumerate(folds):
        if len(test_idx) == 0:
            continue
        train_idx = np.setdiff1d(np.arange(n), test_idx, assume_unique=False)
        for name in names:
            model = _fit_member(name, X[train_idx], y[train_idx], n_estimators, seed + 17 * fold_id)
            oof[name][test_idx] = model.predict(X[test_idx])
    grid = np.linspace(0.0, 1.0, 21)
    best = None
    plain = None
    for forest_weight in grid:
        weights = {"forest": float(forest_weight), "ridge": float(1.0 - forest_weight)}
        blended = weights["forest"] * oof["forest"] + weights["ridge"] * oof["ridge"]
        mse = float(np.mean((blended - y) ** 2))
        penalized = mse * (1.0 + float(ridge_penalty) * weights["ridge"])
        if best is None or penalized < best[0]:
            best = (penalized, dict(weights))
        if plain is None or mse < plain[0]:
            plain = (mse, dict(weights))
    return best[1], plain[1]


class _Ensemble:
    def __init__(self, models: dict, weights: dict[str, float]):
        self.models = models
        self.weights = weights

    def predict(self, X) -> np.ndarray:
        out = np.zeros(len(X), dtype=float)
        for name, model in self.models.items():
            out += self.weights[name] * model.predict(X)
        return out


def _fit_ensemble(X, y, weights: dict[str, float], n_estimators: int, seed: int) -> _Ensemble:
    models = {name: _fit_member(name, X, y, n_estimators, seed + i) for i, name in enumerate(weights)}
    return _Ensemble(models, weights)


def propensity_models(L, A, seed: int) -> list:
    models = []
    for t in range(A.shape[1]):
        X = design_g(L, A, t)
        y = A[:, t].astype(int)
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("logit", LogisticRegression(C=1.0, solver="lbfgs", max_iter=250)),
            ]
        )
        if len(np.unique(y)) < 2:
            models.append(float(y.mean()))
        else:
            model.fit(X, y)
            models.append(model)
    return models


def _g1(model, L, A, t) -> np.ndarray:
    if isinstance(model, float):
        return np.full(len(A), model)
    return model.predict_proba(design_g(L, A, t))[:, 1]


def overlap_score(L, A, seed: int = 0) -> float:
    """Share of fitted propensities that sit in (0.1, 0.9), averaged over time."""
    shares = []
    for t, model in enumerate(propensity_models(L, A, seed)):
        prob = _g1(model, L, A, t)
        shares.append(float(np.mean((prob > 0.1) & (prob < 0.9))))
    return float(np.mean(shares)) if shares else 1.0


def _intervention_mean(
    L,
    A,
    Y,
    a_star: int,
    g_models: list,
    truncation: float,
    ridge_penalty: float,
    frozen_weights: dict[str, float] | None,
    n_estimators: int,
    seed: int,
) -> tuple[float, list[dict[str, float]]]:
    Q = np.asarray(Y, dtype=float).ravel().copy()
    traces = []
    T = A.shape[1]
    for step, t in enumerate(reversed(range(T))):
        X = design_q(L, A, t)
        if frozen_weights is None:
            weights, _ = select_weights(X, Q, ridge_penalty, n_estimators, seed + 30 * step)
        else:
            weights = dict(frozen_weights)
        traces.append(weights)
        ensemble = _fit_ensemble(X, Q, weights, n_estimators, seed + 100 * step)
        Q_hat = ensemble.predict(X)
        prob = np.clip(_g1(g_models[t], L, A, t), truncation, 1.0 - truncation)
        g_star = prob if a_star == 1 else 1.0 - prob
        followed = (A[:, t] == a_star).astype(float)
        clever = followed / g_star
        denom = float(np.dot(clever, clever))
        residual = Q - Q_hat
        epsilon = float(np.dot(clever, residual) / denom) if denom > 1e-8 else 0.0
        A_cf = np.array(A, copy=True)
        A_cf[:, t] = a_star
        Q = ensemble.predict(design_q(L, A_cf, t)) + epsilon / g_star
    return float(np.mean(Q)), traces


def estimate_ate(
    L,
    A,
    Y,
    truncation: float = 0.02,
    ridge_penalty: float = 0.0,
    frozen_weights: dict[str, float] | None = None,
    n_estimators: int = 40,
    seed: int = 0,
) -> dict:
    """Always-treat minus never-treat, with the Super Learner weights used on Q."""
    L = np.asarray(L, dtype=float)
    A = np.asarray(A, dtype=int)
    Y = np.asarray(Y, dtype=float).ravel()
    g_models = propensity_models(L, A, seed)
    treated, trace_1 = _intervention_mean(
        L, A, Y, 1, g_models, truncation, ridge_penalty, frozen_weights, n_estimators, seed
    )
    control, trace_0 = _intervention_mean(
        L, A, Y, 0, g_models, truncation, ridge_penalty, frozen_weights, n_estimators, seed + 1
    )
    traces = trace_1 + trace_0
    forest_weight = float(np.mean([row["forest"] for row in traces])) if traces else 0.5
    return {
        "psi": treated - control,
        "psi_treat": treated,
        "psi_control": control,
        "forest_weight": forest_weight,
        "truncation": float(truncation),
        "ridge_penalty": float(ridge_penalty),
    }


def baseline_weights(L, A, Y, n_estimators: int, seed: int) -> dict[str, float]:
    """Weights from the pre-shift terminal regression. The frozen estimator reuses them."""
    X = design_q(L, A, A.shape[1] - 1)
    weights, _ = select_weights(X, Y, ridge_penalty=0.0, n_estimators=n_estimators, seed=seed)
    return weights
