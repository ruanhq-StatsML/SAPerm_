"""Per-site MMD, HSIC, and R-risk, and the same three quantities on a pool.

MMD is the covariate arm: an RBF discrepancy between the reference rows and
the new rows. It does not see the label.

HSIC is the dependence between the reference residual and the environment
indicator. The residual is Y minus a regression fit on the reference only,
so a change in P(Y|X) moves it and a pure change in P(X) does not, once
that regression fits.

R-risk is the fraction of residual variance the R-learner explains with a
heterogeneous environment effect tau(X). A constant shift in the marginal
label rate is not required. The skill gate multiplies HSIC and R-risk by
how well Y is predictable from X on the new rows alone. A noise site can
have a large raw residual gap and still a gated score of zero.

Leave-one-coordinate versions attribute the same three quantities to one
column. The aggregate of those vectors is a weighted mean across sites.
The pooled versions recompute the three quantities on the stacked rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

@dataclass
class KernelConfig:
    n_estimators: int = 16
    max_depth: int = 5
    min_samples_leaf: int = 5
    mmd_size: int = 220
    hsic_size: int = 240
    perm_repeats: int = 2
    min_propensity: float = 0.05


def _regressor(seed: int, cfg: KernelConfig) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=1,
        random_state=seed,
    )


def _classifier(seed: int, cfg: KernelConfig) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=1,
        random_state=seed,
    )


def _fit_regressor(X: np.ndarray, y: np.ndarray, seed: int, cfg: KernelConfig, sample_weight: np.ndarray | None = None):
    y = np.asarray(y, dtype=float)
    if y.size < 8 or float(np.std(y)) < 1e-8:
        return ("constant", float(np.mean(y) if y.size else 0.0))
    model = _regressor(seed, cfg)
    if sample_weight is None:
        model.fit(X, y)
    else:
        model.fit(X, y, sample_weight=np.asarray(sample_weight, dtype=float))
    return ("model", model)


def _predict_reg(fit, X: np.ndarray) -> np.ndarray:
    kind, obj = fit
    if kind == "constant":
        return np.full(X.shape[0], float(obj))
    return np.asarray(fit[1].predict(X), dtype=float)


def _subsample_rows(n: int, size: int, rng: np.random.Generator) -> np.ndarray:
    if n <= size:
        return np.arange(n)
    return rng.choice(n, size=size, replace=False)


def normalized_hsic(a: np.ndarray, b: np.ndarray) -> float:
    """Kernel dependence in [0, 1] between two paired one-dimensional samples."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    n = int(a.size)
    if n < 12 or b.size != n:
        return 0.0
    if float(np.std(a)) < 1e-10 or float(np.std(b)) < 1e-10:
        return 0.0

    def kernel(z: np.ndarray) -> np.ndarray:
        d2 = (z[:, None] - z[None, :]) ** 2
        positive = d2[d2 > 0]
        med = float(np.median(positive)) if positive.size else 1.0
        gamma = 1.0 / (2.0 * med) if med > 0 else 1.0
        return np.exp(-gamma * d2)

    K = kernel(a)
    L = kernel(b)
    centering = np.eye(n) - np.ones((n, n)) / n
    Kc = centering @ K @ centering
    Lc = centering @ L @ centering
    scale = float((n - 1) ** 2)
    hsic = float(np.sum(Kc * Lc) / scale)
    hkk = float(np.sum(Kc * Kc) / scale)
    hll = float(np.sum(Lc * Lc) / scale)
    denom = np.sqrt(max(hkk, 0.0) * max(hll, 0.0))
    if denom < 1e-15:
        return 0.0
    return float(np.clip(hsic / denom, 0.0, 1.0))


def hsic_residual(residual: np.ndarray, environment: np.ndarray, seed: int, size: int) -> float:
    residual = np.asarray(residual, dtype=float).ravel()
    environment = np.asarray(environment, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    take = _subsample_rows(residual.size, size, rng)
    # Keep both environments in the subsample when the full sample has both.
    if environment.size > size and len(np.unique(environment)) > 1:
        groups = []
        for value in np.unique(environment):
            pool = np.flatnonzero(environment == value)
            n_take = max(8, int(round(size * pool.size / environment.size)))
            n_take = min(n_take, pool.size)
            groups.append(rng.choice(pool, size=n_take, replace=False))
        take = np.concatenate(groups)
    return normalized_hsic(residual[take], environment[take])


def skill_on_new(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    cfg: KernelConfig,
    *,
    binary: bool,
) -> float:
    """How much of the new-window outcome is predictable from X. OOF, so noise stays near zero."""
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    if n < 24:
        return 0.0
    if binary:
        y_int = y.astype(int)
        if len(np.unique(y_int)) < 2:
            return 0.0
        pred = np.full(n, float(np.mean(y_int)))
        splitter = KFold(n_splits=2, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(splitter.split(X)):
            if len(np.unique(y_int[tr])) < 2:
                pred[te] = float(np.mean(y_int[tr])) if tr.size else 0.5
                continue
            model = _classifier(seed + 11 * (fold + 1), cfg)
            model.fit(X[tr], y_int[tr])
            proba = model.predict_proba(X[te])
            classes = list(model.classes_)
            pred[te] = proba[:, classes.index(1)] if 1 in classes else 0.0
        if len(np.unique(y_int)) < 2:
            return 0.0
        auc = float(roc_auc_score(y_int, pred))
        return float(max(2.0 * (auc - 0.5), 0.0))
    pred = np.full(n, float(np.mean(y)))
    splitter = KFold(n_splits=2, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(splitter.split(X)):
        fit = _fit_regressor(X[tr], y[tr], seed + 11 * (fold + 1), cfg)
        pred[te] = _predict_reg(fit, X[te])
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot < 1e-10:
        return 0.0
    r2 = 1.0 - float(np.sum((y - pred) ** 2)) / ss_tot
    return float(max(r2, 0.0))


def _propensity(X_train: np.ndarray, w_train: np.ndarray, X_test: np.ndarray, seed: int) -> np.ndarray:
    w_train = np.asarray(w_train).astype(int)
    if len(np.unique(w_train)) < 2:
        return np.full(X_test.shape[0], float(np.mean(w_train)))
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=250, random_state=seed)
    model.fit(X_train, w_train)
    return model.predict_proba(X_test)[:, 1]


def r_risk_and_loco(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    seed: int,
    cfg: KernelConfig,
    *,
    with_loco: bool = True,
) -> tuple[float, np.ndarray]:
    """Cross-fit R-learner risk drop and permutation importance of tau.

    Outcome and propensity are fit on the complementary fold. Tau is fit once
    on that fold's residual-on-residual target. Coordinate importance permutes
    one column of the test fold and re-reads the same tau, so the loss drop
    is the share of explained residual variance that column was carrying.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    n, p = X.shape
    m_hat = np.zeros(n)
    e_hat = np.full(n, float(np.mean(w)))
    splitter = KFold(n_splits=2, shuffle=True, random_state=seed)
    folds = list(splitter.split(X))
    for fold, (tr, te) in enumerate(folds):
        outcome = _fit_regressor(X[tr], y[tr], seed + 3 * (fold + 1), cfg)
        m_hat[te] = _predict_reg(outcome, X[te])
        e_hat[te] = _propensity(X[tr], w[tr], X[te], seed + 9 + fold)
    y_res = y - m_hat
    w_res = w - e_hat
    l0 = float(np.sum(y_res ** 2))
    if l0 < 1e-10:
        return 0.0, np.zeros(p)

    tau_fits: list[tuple[np.ndarray, object]] = []
    l_full = 0.0
    for fold, (tr, te) in enumerate(folds):
        usable = np.abs(w_res[tr]) > cfg.min_propensity
        if int(usable.sum()) < 16:
            tau_fits.append((te, ("constant", 0.0)))
            l_full += float(np.sum(y_res[te] ** 2))
            continue
        target = y_res[tr][usable] / w_res[tr][usable]
        weight = w_res[tr][usable] ** 2
        fit = _fit_regressor(X[tr][usable], target, seed + 40 + fold, cfg, sample_weight=weight)
        tau_fits.append((te, fit))
        pred = w_res[te] * _predict_reg(fit, X[te])
        l_full += float(np.sum((y_res[te] - pred) ** 2))
    risk = float(max(l0 - l_full, 0.0) / l0)
    if (not with_loco) or risk <= 1e-8:
        return risk, np.zeros(p)

    rng = np.random.default_rng(seed + 17)
    loco = np.zeros(p)
    for j in range(p):
        gaps = []
        for _rep in range(cfg.perm_repeats):
            l_perm = 0.0
            for te, fit in tau_fits:
                Xp = X[te].copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                pred = w_res[te] * _predict_reg(fit, Xp)
                l_perm += float(np.sum((y_res[te] - pred) ** 2))
            gaps.append((l_perm - l_full) / l0)
        loco[j] = float(max(np.mean(gaps), 0.0))
    return risk, loco


def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    a2 = np.sum(A * A, axis=1)[:, None]
    b2 = np.sum(B * B, axis=1)[None, :]
    return np.maximum(a2 + b2 - 2.0 * A @ B.T, 0.0)


def _mmd_gamma(X: np.ndarray, Y: np.ndarray, gamma: float) -> float:
    kxx = np.exp(-gamma * _sqdist(X, X))
    kyy = np.exp(-gamma * _sqdist(Y, Y))
    kxy = np.exp(-gamma * _sqdist(X, Y))
    n, m = X.shape[0], Y.shape[0]
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    stat = float(kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2.0 * kxy.mean())
    return float(max(stat, 0.0))


def _median_gamma(X: np.ndarray, Y: np.ndarray, seed: int, size: int) -> float:
    rng = np.random.default_rng(seed)
    Z = np.vstack([X, Y])
    if Z.shape[0] > size:
        Z = Z[rng.choice(Z.shape[0], size=size, replace=False)]
    d2 = _sqdist(Z, Z)
    positive = d2[d2 > 0]
    med = float(np.median(positive)) if positive.size else 1.0
    return 1.0 / (2.0 * med) if med > 0 else 1.0


def mmd_loco(X0: np.ndarray, X1: np.ndarray, seed: int, cfg: KernelConfig) -> tuple[float, np.ndarray]:
    """RBF MMD and the drop when one column is removed.

    A within-sample permutation keeps each sample's marginal, so it does not
    remove a mean shift. The column is left out instead, and the kernel width
    stays the one estimated on the full matrix.
    """
    rng = np.random.default_rng(seed)
    if X0.shape[0] > cfg.mmd_size:
        X0 = X0[rng.choice(X0.shape[0], size=cfg.mmd_size, replace=False)]
    if X1.shape[0] > cfg.mmd_size:
        X1 = X1[rng.choice(X1.shape[0], size=cfg.mmd_size, replace=False)]
    gamma = _median_gamma(X0, X1, seed + 1, min(cfg.mmd_size, X0.shape[0] + X1.shape[0]))
    base = _mmd_gamma(X0, X1, gamma)
    p = X0.shape[1]
    loco = np.zeros(p)
    for j in range(p):
        cols = [k for k in range(p) if k != j]
        reduced = _mmd_gamma(X0[:, cols], X1[:, cols], gamma)
        loco[j] = float(max(base - reduced, 0.0))
    return float(base), loco


def _poly5(column: np.ndarray) -> np.ndarray:
    x = np.asarray(column, dtype=float).ravel()
    return np.column_stack([x ** k for k in range(1, 6)])


def _quadratic_features(X: np.ndarray) -> np.ndarray:
    parts = [X, X ** 2]
    inter = [(X[:, a] * X[:, b])[:, None] for a in range(X.shape[1]) for b in range(a + 1, X.shape[1])]
    return np.column_stack(parts + inter)


def _quadratic_keep(p: int, drop: int) -> np.ndarray:
    n_inter = p * (p - 1) // 2
    mask = np.ones(2 * p + n_inter, dtype=bool)
    mask[drop] = False
    mask[p + drop] = False
    cursor = 2 * p
    for a in range(p):
        for b in range(a + 1, p):
            if a == drop or b == drop:
                mask[cursor] = False
            cursor += 1
    return mask


def _skill_linear(X: np.ndarray, y: np.ndarray, seed: int) -> float:
    y = np.asarray(y, dtype=float).ravel()
    if y.size < 20 or float(np.std(y)) < 1e-8:
        return 0.0
    pred = np.zeros(y.size)
    splitter = KFold(n_splits=2, shuffle=True, random_state=seed)
    for tr, te in splitter.split(X):
        if float(np.std(y[tr])) < 1e-8:
            pred[te] = float(np.mean(y[tr]))
        else:
            pred[te] = LinearRegression().fit(X[tr], y[tr]).predict(X[te])
    ss = float(np.sum((y - y.mean()) ** 2))
    if ss < 1e-10:
        return 0.0
    return float(max(1.0 - float(np.sum((y - pred) ** 2)) / ss, 0.0))


def r_risk_linear(X: np.ndarray, y: np.ndarray, w: np.ndarray, seed: int) -> tuple[float, np.ndarray]:
    """Cross-fit R-learner risk for a continuous outcome with a linear reference.

    The reference regression is ordinary least squares, which matches the
    shared conditional mean of the Gaussian designs. Tau is the best of a
    linear effect, a quadratic effect, and a degree-5 effect of one coordinate.
    The coordinate vector is that univariate risk, with a quadratic deletion
    score used when the interaction model is the one that carries the risk.
    A pure covariate shift redraws labels from the same conditional, so this
    risk stays near zero. A noise site does not: its risk is large until the
    new-window skill multiplies it.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    n, p = X.shape
    folds = list(KFold(n_splits=2, shuffle=True, random_state=seed).split(X))
    m_hat = np.zeros(n)
    e_hat = np.full(n, float(np.mean(w)))
    for tr, te in folds:
        if float(np.std(y[tr])) < 1e-8:
            m_hat[te] = float(np.mean(y[tr]))
        else:
            m_hat[te] = LinearRegression().fit(X[tr], y[tr]).predict(X[te])
        e_hat[te] = _propensity(X[tr], w[tr], X[te], seed)
    y_res = y - m_hat
    w_res = w - e_hat
    l0 = float(np.sum(y_res ** 2))
    if l0 < 1e-10:
        return 0.0, np.zeros(p)

    def loss_of(features: np.ndarray, alpha: float) -> float:
        total = 0.0
        for tr, te in folds:
            usable = np.abs(w_res[tr]) > 0.08
            null = float(np.sum(y_res[te] ** 2))
            if int(usable.sum()) < 25:
                total += null
                continue
            target = y_res[tr][usable] / w_res[tr][usable]
            weight = w_res[tr][usable] ** 2
            if float(np.std(target)) < 1e-8:
                total += null
                continue
            model = Ridge(alpha=alpha).fit(features[tr][usable], target, sample_weight=weight)
            pred = w_res[te] * model.predict(features[te])
            total += min(float(np.sum((y_res[te] - pred) ** 2)), null)
        return total

    l_lin = loss_of(X, 1.0)
    quadratic = _quadratic_features(X)
    l_quad = loss_of(quadratic, 4.0)
    uni = np.zeros(p)
    for j in range(p):
        basis = _poly5(X[:, j])
        uni[j] = max(l0 - loss_of(basis, 1.0), 0.0) / l0
    lin_risk = max(l0 - l_lin, 0.0) / l0
    quad_risk = max(l0 - l_quad, 0.0) / l0
    vector = uni.copy()
    if quad_risk > float(uni.max()):
        for j in range(p):
            reduced = loss_of(quadratic[:, _quadratic_keep(p, j)], 4.0)
            vector[j] = max(vector[j], (reduced - l_quad) / l0)
    scalar = float(max(lin_risk, quad_risk, float(uni.max(initial=0.0))))
    return scalar, np.maximum(vector, 0.0)


def hsic_and_loco(
    X0: np.ndarray,
    y0: np.ndarray,
    X1: np.ndarray,
    y1: np.ndarray,
    seed: int,
    cfg: KernelConfig,
    *,
    linear_outcome: bool = False,
) -> tuple[float, np.ndarray]:
    """HSIC of the reference residual with the environment, and a per-coordinate gap.

    The coordinate score is the increase, from the reference window to the new
    window, of HSIC between that column and the reference residual. A column
    that already predicts the reference outcome does not get credit for the shift.
    """
    y0 = np.asarray(y0, dtype=float)
    y1 = np.asarray(y1, dtype=float)
    if linear_outcome:
        if float(np.std(y0)) < 1e-8:
            level = float(np.mean(y0))
            r0 = y0 - level
            r1 = y1 - level
        else:
            model = LinearRegression().fit(X0, y0)
            r0 = y0 - model.predict(X0)
            r1 = y1 - model.predict(X1)
    else:
        fit = _fit_regressor(X0, y0, seed, cfg)
        r0 = y0 - _predict_reg(fit, X0)
        r1 = y1 - _predict_reg(fit, X1)
    residual = np.concatenate([r0, r1])
    environment = np.concatenate([np.zeros(r0.size), np.ones(r1.size)])
    scalar = hsic_residual(residual, environment, seed + 5, cfg.hsic_size)
    rng = np.random.default_rng(seed + 6)
    if X0.shape[0] > cfg.hsic_size:
        take = rng.choice(X0.shape[0], size=cfg.hsic_size, replace=False)
        X0, r0 = X0[take], r0[take]
    if X1.shape[0] > cfg.hsic_size:
        take = rng.choice(X1.shape[0], size=cfg.hsic_size, replace=False)
        X1, r1 = X1[take], r1[take]
    p = X0.shape[1]
    loco = np.zeros(p)
    for j in range(p):
        new_dep = normalized_hsic(X1[:, j], r1)
        old_dep = normalized_hsic(X0[:, j], r0)
        loco[j] = float(max(new_dep - old_dep, 0.0))
    return float(scalar), loco


def score_pair(
    X0: np.ndarray,
    y0: np.ndarray,
    X1: np.ndarray,
    y1: np.ndarray,
    seed: int,
    cfg: KernelConfig,
    *,
    binary: bool,
    with_loco: bool = True,
) -> dict:
    """Score one site, or one pooled stack, against its own reference rows."""
    X0 = np.asarray(X0, dtype=float)
    X1 = np.asarray(X1, dtype=float)
    y0 = np.asarray(y0, dtype=float).ravel()
    y1 = np.asarray(y1, dtype=float).ravel()
    mmd, mmd_vec = mmd_loco(X0, X1, seed, cfg)
    hsic, hsic_vec = hsic_and_loco(X0, y0, X1, y1, seed + 30, cfg, linear_outcome=not binary)
    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])
    w = np.concatenate([np.zeros(y0.size), np.ones(y1.size)])
    if binary:
        risk, risk_vec = r_risk_and_loco(X, y, w, seed + 60, cfg, with_loco=with_loco)
        skill = skill_on_new(X1, y1, seed + 90, cfg, binary=True)
    else:
        risk, risk_vec = r_risk_linear(X, y, w, seed + 60)
        skill = _skill_linear(X1, y1, seed + 90)
    return {
        "mmd": float(mmd),
        "hsic": float(hsic),
        "r_risk": float(risk),
        "skill": float(skill),
        "hsic_gated": float(hsic * skill),
        "r_gated": float(risk * skill),
        "mmd_loco": mmd_vec,
        "hsic_loco": hsic_vec,
        "r_loco": risk_vec,
    }
