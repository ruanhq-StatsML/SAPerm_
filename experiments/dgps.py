"""Every data-generating process used on this machine, in one place.

    python3 -m experiments.dgps
    python3 -m experiments.dgps --smoke

Families
--------
site_batch
    Four Gaussian two-batch generators used by the site-level driver table.
    Each returns ``(X0, y0, X1, y1)``. The planted support is the first four
    coordinates.
fsds_stacked
    The manuscript generators. They return one stacked ``(X, Y, W)`` with
    ``W`` the batch indicator. Linear concept drift leaves coordinate 0
    unshifted, because its coefficient is ``arange(4) * delta``.
injection
    Shifts painted onto an existing table. Each returns ``(df1, df2, support)``
    with the outcome stacked as the last column.
feature_rank
    The TabReD plant: sort one column, cut in half, then add a linear signal,
    a nonlinear signal, a replaced regression, or a covariate shift.
ten_site
    Ten sites on one real covariate table. Four stay clean, three rewrite a
    tail with a halfspace, two rewrite with a product or a sine-square, and
    one replaces labels with coin flips.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Site-batch generators. These four are the ones behind the saved table.
# ---------------------------------------------------------------------------

P = 20
N_PER = 900
RHO = 0.3
GAMMA = 1.0
DELTA = 1.0
SUPPORT = np.arange(4)


def ar1_cov(p: int, rho: float) -> np.ndarray:
    idx = np.arange(p)
    cov = float(rho) ** np.abs(idx[:, None] - idx[None, :])
    return cov + 1e-6 * np.eye(p)


def draw_x(n: int, p: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    return rng.multivariate_normal(np.zeros(p), ar1_cov(p, rho), size=n)


def standardize_col(col: np.ndarray) -> np.ndarray:
    col = np.asarray(col, dtype=float).ravel()
    return (col - col.mean()) / (col.std() + 1e-8)


def linear_covariate(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, p = N_PER, P
    corr = ar1_cov(p, RHO)
    x = rng.multivariate_normal(np.zeros(p), corr, size=n)
    beta = np.zeros(p)
    beta[:4] = GAMMA
    x0 = x[: n // 2].copy()
    x1 = x[n // 2 :].copy()
    shift = x0[:, :4] @ beta[:4]
    x0[:, :4] = x0[:, :4] + shift[:, None]
    y0 = x0 @ beta + rng.normal(0, 1, len(x0))
    y1 = x1 @ beta + rng.normal(0, 1, len(x1))
    return x0, y0, x1, y1


def linear_concept(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, p = N_PER, P
    corr = ar1_cov(p, RHO)
    x = rng.multivariate_normal(np.zeros(p), corr, size=n)
    beta0 = np.ones(p)
    y = x @ beta0 + rng.normal(0, 1, n)
    beta = np.zeros(p)
    beta[:4] = np.array([1.0, 1.0, 1.0, 1.0]) * DELTA
    y = y.copy()
    y[n // 2 :] = y[n // 2 :] + x[n // 2 :] @ beta
    return x[: n // 2], y[: n // 2], x[n // 2 :], y[n // 2 :]


def nonlinear_delta(x: np.ndarray, rng: np.random.Generator, scale: float) -> np.ndarray:
    delta = np.zeros(x.shape[0], dtype=float)
    active = SUPPORT
    for j in active:
        z = standardize_col(x[:, j])
        others = active[active != j]
        other = int(rng.choice(others))
        z2 = standardize_col(x[:, other])
        delta += scale * (
            rng.uniform(0.5, 1.0) * np.sin(z)
            + rng.uniform(0.25, 0.5) * (z ** 2)
            + rng.uniform(0.2, 0.35) * z * z2
        )
    return delta


def nonlinear_concept(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    beta = np.zeros(P)
    beta[SUPPORT] = rng.uniform(0.5, 1.5, size=4)
    x0 = draw_x(N_PER, P, RHO, rng)
    x1 = draw_x(N_PER, P, RHO, rng)
    y0 = x0 @ beta + nonlinear_delta(x0, rng, 0.5) + rng.normal(0, 0.1, N_PER)
    y1 = x1 @ beta + nonlinear_delta(x1, rng, 0.5 + DELTA) + rng.normal(0, 0.1, N_PER)
    return x0, y0, x1, y1


def nonlinear_covariate(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    beta = np.zeros(P)
    beta[SUPPORT] = rng.uniform(0.5, 1.5, size=4)
    x0 = draw_x(N_PER, P, RHO, rng)
    x1 = draw_x(N_PER, P, RHO, rng)
    for j in SUPPORT:
        z = standardize_col(x1[:, j])
        x1[:, j] = x1[:, j] + GAMMA * (
            rng.uniform(0.5, 1.0) * np.sin(z) + rng.uniform(0.25, 0.5) * (z ** 2)
        )
    y0 = x0 @ beta + rng.normal(0, 0.1, N_PER)
    y1 = x1 @ beta + rng.normal(0, 0.1, N_PER)
    return x0, y0, x1, y1


GENERATORS = {
    "linear_covariate": ("covariate", linear_covariate, 1),
    "nonlinear_covariate": ("covariate", nonlinear_covariate, 2),
    "linear_concept": ("concept", linear_concept, 3),
    "nonlinear_concept": ("concept", nonlinear_concept, 4),
}


# ---------------------------------------------------------------------------
# Manuscript stacked generators. W marks the batch. Bandwidth-limited AR(1).
# ---------------------------------------------------------------------------

FSDS_P = 20
FSDS_SUPPORT = np.arange(4)


def banded_toeplitz_corr(rho: float, p: int = 20, bandwidth: int = 6) -> np.ndarray:
    corr = np.empty((p, p), dtype=float)
    lags = rho ** np.arange(p)
    for i in range(p):
        for j in range(p):
            corr[i, j] = lags[abs(i - j)] if abs(i - j) <= bandwidth else 0.0
    return corr


def fsds_stacked_linear_concept(
    n: int = 2000, p: int = 20, delta_beta: float = 1.0, rho: float = 0.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Second half of Y gains a drift. Coordinate 0 has coefficient 0."""
    rng = np.random.default_rng(seed)
    x = rng.multivariate_normal(np.zeros(p), banded_toeplitz_corr(float(rho), p=p), size=n)
    y = x @ np.ones(p) + rng.normal(0, 1, n)
    beta = np.zeros(p)
    beta[:4] = np.arange(4) * float(delta_beta)
    y = y.copy()
    y[n // 2 :] = y[n // 2 :] + x[n // 2 :] @ beta
    w = np.concatenate([np.zeros(n // 2), np.ones(n - n // 2)])
    return x, y, w


def fsds_stacked_linear_covariate(
    n: int = 2000, p: int = 20, gamma: float = 1.0, rho: float = 0.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First half of X is shifted on the first four columns. The regression stays linear."""
    rng = np.random.default_rng(seed)
    x = rng.multivariate_normal(np.zeros(p), banded_toeplitz_corr(float(rho), p=p), size=n)
    beta = np.zeros(p)
    beta[:4] = float(gamma)
    shift = x[: n // 2, :4] @ beta[:4]
    x = x.copy()
    x[: n // 2, :4] = x[: n // 2, :4] + shift[:, None]
    y = x @ beta + rng.normal(0, 1, n)
    w = np.concatenate([np.zeros(n // 2), np.ones(n - n // 2)])
    return x, y, w


def _fsds_base_beta(rng: np.random.Generator, p: int) -> np.ndarray:
    beta = np.zeros(p, dtype=float)
    beta[FSDS_SUPPORT] = rng.uniform(0.5, 1.5, size=len(FSDS_SUPPORT))
    return beta


def _fsds_nonlinear_delta(x: np.ndarray, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    delta = np.zeros(x.shape[0], dtype=float)
    active = FSDS_SUPPORT
    for j in active:
        z = standardize_col(x[:, j])
        others = active[active != j]
        other = int(rng.choice(others)) if len(others) else int(j)
        z2 = standardize_col(x[:, other])
        delta += scale * (
            rng.uniform(0.5, 1.0) * np.sin(z)
            + rng.uniform(0.25, 0.5) * (z ** 2)
            + rng.uniform(0.2, 0.35) * z * z2
        )
    return delta


def fsds_stacked_nonlinear_concept(
    delta_beta: float = 1.0,
    rho: float = 0.3,
    seed: int = 0,
    p: int = FSDS_P,
    n_per_domain: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    beta = _fsds_base_beta(rng, p)
    cov = rho ** np.abs(np.arange(p)[:, None] - np.arange(p)[None, :]) + 1e-6 * np.eye(p)
    x0 = rng.multivariate_normal(np.zeros(p), cov, size=n_per_domain)
    x1 = rng.multivariate_normal(np.zeros(p), cov, size=n_per_domain)
    y0 = x0 @ beta + _fsds_nonlinear_delta(x0, rng, 0.5) + rng.normal(0, 0.1, n_per_domain)
    y1 = x1 @ beta + _fsds_nonlinear_delta(x1, rng, 0.5 + float(delta_beta)) + rng.normal(0, 0.1, n_per_domain)
    x = np.vstack([x0, x1])
    y = np.concatenate([y0, y1])
    w = np.concatenate([np.zeros(len(y0), dtype=int), np.ones(len(y1), dtype=int)])
    return x, y, w


def fsds_stacked_nonlinear_covariate(
    gamma: float = 1.0,
    rho: float = 0.3,
    seed: int = 0,
    p: int = FSDS_P,
    n_per_domain: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    beta = _fsds_base_beta(rng, p)
    cov = rho ** np.abs(np.arange(p)[:, None] - np.arange(p)[None, :]) + 1e-6 * np.eye(p)
    x0 = rng.multivariate_normal(np.zeros(p), cov, size=n_per_domain)
    x1 = rng.multivariate_normal(np.zeros(p), cov, size=n_per_domain).copy()
    for j in FSDS_SUPPORT:
        z = standardize_col(x1[:, j])
        x1[:, j] = x1[:, j] + float(gamma) * (
            rng.uniform(0.5, 1.0) * np.sin(z) + rng.uniform(0.25, 0.5) * (z ** 2)
        )
        if rng.uniform() > 0.5:
            x1[:, j] = x1[:, j] + float(gamma) * 0.25 * (z ** 2)
    y0 = x0 @ beta + rng.normal(0, 0.1, n_per_domain)
    y1 = x1 @ beta + rng.normal(0, 0.1, n_per_domain)
    x = np.vstack([x0, x1])
    y = np.concatenate([y0, y1])
    w = np.concatenate([np.zeros(len(y0), dtype=int), np.ones(len(y1), dtype=int)])
    return x, y, w


# ---------------------------------------------------------------------------
# Injections onto a table that already exists.
# ---------------------------------------------------------------------------

def injection_nonlinear_interaction(x0, y0, x1, y1, n_prop: float = 0.35, seed: int = 2026):
    """Sin, square, and a product, plus a coin-flip quadratic nudge to X."""
    rng = np.random.default_rng(seed)
    x1 = np.asarray(x1, dtype=float).copy()
    y1 = np.asarray(y1, dtype=float).copy()
    n2, p = x1.shape
    n_shift = max(1, int(np.round(p * n_prop)))
    support = rng.choice(np.arange(p), size=n_shift, replace=False)
    delta = np.zeros(n2, dtype=float)
    for j in support:
        z = standardize_col(x1[:, j])
        others = np.setdiff1d(support, j)
        other = int(j) if len(others) == 0 else int(rng.choice(others))
        z2 = (x1[:, other] - x1[:, other].mean()) / (x1[:, other].std() + 1e-7)
        delta += (
            rng.uniform(0.5, 1.0) * np.sin(z)
            + rng.uniform(0.25, 0.5) * (z ** 2)
            + rng.uniform(0.2, 0.35) * z * z2
        )
        if rng.uniform(0, 1) > 0.5:
            x1[:, j] = x1[:, j] + (z ** 2) * 0.25
    y1 = y1 + delta + rng.normal(0, 0.05, n2)
    df1 = np.hstack([np.asarray(x0, dtype=float), np.asarray(y0, dtype=float).reshape(-1, 1)])
    df2 = np.hstack([x1, y1.reshape(-1, 1)])
    return df1, df2, support


def injection_linear(x0, y0, x1, y1, n_prop: float = 0.3, seed: int = 0):
    """Add a linear signal on a random subset of the new batch."""
    rng = np.random.default_rng(seed)
    x1 = np.asarray(x1, dtype=float)
    y1 = np.asarray(y1, dtype=float).copy()
    n2, p = x1.shape
    n_shift = max(1, int(np.round(p * n_prop)))
    support = rng.choice(p, size=n_shift, replace=False)
    beta = np.zeros(p)
    beta[support] = np.linspace(0, 1, len(support))
    y1 = y1 + x1 @ beta + rng.normal(0, 0.05, n2)
    df1 = np.hstack([np.asarray(x0, dtype=float), np.asarray(y0, dtype=float).reshape(-1, 1)])
    df2 = np.hstack([x1, y1.reshape(-1, 1)])
    return df1, df2, support


def injection_nonlinear(x0, y0, x1, y1, n_prop: float = 0.3, seed: int = 0, strength: float = 1.0):
    """Sin and square on a random subset. X itself is left alone."""
    rng = np.random.default_rng(seed)
    x1 = np.asarray(x1, dtype=float)
    y1 = np.asarray(y1, dtype=float).copy()
    n2, p = x1.shape
    n_shift = max(1, int(np.round(p * n_prop)))
    support = np.sort(rng.choice(p, size=n_shift, replace=False))
    delta = np.zeros(n2, dtype=float)
    for j in support:
        z = standardize_col(x1[:, j])
        delta += rng.uniform(0.3, strength) * np.sin(z) + rng.uniform(0.3, strength) * (z ** 2)
    y1 = y1 + delta + rng.normal(0, 0.05, n2)
    df1 = np.hstack([np.asarray(x0, dtype=float), np.asarray(y0, dtype=float).reshape(-1, 1)])
    df2 = np.hstack([x1, y1.reshape(-1, 1)])
    return df1, df2, support


def injection_interaction(x0, y0, x1, y1, n_prop: float = 0.3, seed: int = 0, strength: float = 1.0):
    """Pairwise products on a random subset of the new batch."""
    rng = np.random.default_rng(seed)
    x1 = np.asarray(x1, dtype=float)
    y1 = np.asarray(y1, dtype=float).copy()
    n2, p = x1.shape
    n_shift = max(2, int(np.round(p * n_prop)))
    support = np.sort(rng.choice(p, size=n_shift, replace=False))
    pairs = list(itertools.combinations(support, 2)) or [(int(support[0]), int(support[0]))]
    delta = np.zeros(n2, dtype=float)
    for (j, k), c in zip(pairs, rng.uniform(0.3, strength, size=len(pairs))):
        delta += c * x1[:, j] * x1[:, k]
    y1 = y1 + delta + rng.normal(0, 0.05, n2)
    df1 = np.hstack([np.asarray(x0, dtype=float), np.asarray(y0, dtype=float).reshape(-1, 1)])
    df2 = np.hstack([x1, y1.reshape(-1, 1)])
    return df1, df2, support


# ---------------------------------------------------------------------------
# TabReD feature-rank plant. Sort one column, cut in half, then shift.
# ---------------------------------------------------------------------------

FEATURE_RANK_SHIFTS = ("linear", "nonlinear", "conceptdrift", "covariateshift")
FEATURE_RANK_SCALE = 1.5


def feature_rank_indices(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x, kind="mergesort")
    mid = len(order) // 2
    return order[:mid], order[mid:]


def choose_ridge_support(x0: np.ndarray, y0: np.ndarray, banned: int, k: int, seed: int) -> np.ndarray:
    """Ten largest |ridge coefficients| on the low half, excluding the split column."""
    rng = np.random.default_rng(seed)
    n = x0.shape[0]
    take = min(n, 12000)
    idx = rng.choice(n, size=take, replace=False)
    z = StandardScaler().fit_transform(x0[idx])
    beta = Ridge(alpha=1.0).fit(z, y0[idx]).coef_
    chosen = [int(j) for j in np.argsort(-np.abs(beta)) if int(j) != banned][:k]
    if len(chosen) < k:
        raise RuntimeError(f"only found {len(chosen)} support features")
    return np.asarray(chosen, dtype=int)


def plant_feature_rank_shift(
    x0: np.ndarray,
    y0: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    support: np.ndarray,
    kind: str,
    rng: np.random.Generator,
    scale: float = FEATURE_RANK_SCALE,
) -> tuple[np.ndarray, np.ndarray]:
    x1 = x1.copy()
    y1 = y1.copy()
    mu = x0[:, support].mean(axis=0)
    sd = x0[:, support].std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    z = (x1[:, support] - mu) / sd
    y_sd = float(np.std(y0))
    if y_sd < 1e-8:
        y_sd = 1.0
    if kind == "linear":
        direction = np.linspace(1.0, 0.4, len(support))
        direction = direction / np.linalg.norm(direction)
        signal = z @ direction
        signal = signal / (np.std(signal) + 1e-8)
        y1 = y1 + scale * y_sd * signal
    elif kind == "nonlinear":
        parts = [np.sin(z[:, 0]), z[:, 1] ** 2]
        if z.shape[1] > 2:
            parts.append(z[:, 2] * z[:, min(3, z.shape[1] - 1)])
        signal = np.sum(np.column_stack(parts), axis=1)
        signal = (signal - signal.mean()) / (np.std(signal) + 1e-8)
        y1 = y1 + scale * y_sd * signal
    elif kind == "conceptdrift":
        raw = np.sin(1.5 * z[:, 0]) + z[:, 1] * z[:, min(2, z.shape[1] - 1)]
        raw = (raw - raw.mean()) / (np.std(raw) + 1e-8)
        noise = rng.normal(0.0, 0.15, size=len(y1))
        y1 = float(np.mean(y0)) + y_sd * raw + 0.15 * y_sd * noise
    elif kind == "covariateshift":
        x1[:, support] = x1[:, support] + scale * sd
    else:
        raise ValueError(kind)
    return x1, y1


# ---------------------------------------------------------------------------
# Ten sites on one covariate table. The noise site is not a primary driver.
# ---------------------------------------------------------------------------

SITE_PLAN: list[tuple[str, str, object]] = [
    ("S0", "clean", None),
    ("S1", "clean", None),
    ("S2", "clean", None),
    ("S3", "clean", None),
    ("S4", "linear", np.array([1.0, 0.35])),
    ("S5", "linear", np.array([0.25, 1.0])),
    ("S6", "linear", np.array([-0.85, 0.55])),
    ("S7", "nonlinear", "product"),
    ("S8", "nonlinear", "sin_square"),
    ("S9", "noise", None),
]


@dataclass
class TenSiteConfig:
    n_per_site: int = 1600
    n_batches: int = 16
    t_star: int = 8
    alpha: float = 1.0
    subgroup_q: float = 0.5
    mode: str = "abrupt"
    ramp: int = 4
    nonlinear_scope: str = "all"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_per_site % self.n_batches != 0:
            raise ValueError("n_per_site must be divisible by n_batches")
        if not 0 < self.t_star < self.n_batches:
            raise ValueError("t_star must lie strictly inside the batch grid")
        if self.mode not in {"abrupt", "gradual"}:
            raise ValueError(self.mode)
        if self.nonlinear_scope not in {"all", "subgroup"}:
            raise ValueError(self.nonlinear_scope)


def _winsorize_zscore(x: np.ndarray, lower: float = 0.01, upper: float = 0.99) -> np.ndarray:
    z = np.asarray(x, dtype=float).copy()
    for j in range(z.shape[1]):
        col = z[:, j]
        lo, hi = np.quantile(col, [lower, upper])
        col = np.clip(col, lo, hi)
        sd = float(col.std())
        z[:, j] = 0.0 if sd < 1e-8 else (col - col.mean()) / sd
    return z


def _activation_weight(batch: np.ndarray, cfg: TenSiteConfig) -> np.ndarray:
    weight = np.zeros(batch.shape[0], dtype=float)
    for t in range(cfg.n_batches):
        if t < cfg.t_star:
            wt = 0.0
        elif cfg.mode == "abrupt":
            wt = 1.0
        else:
            wt = min(1.0, (t - cfg.t_star + 1) / float(cfg.ramp))
        weight[batch == t] = wt
    return weight


def _rate_matched_labels(score: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float, int]:
    y_mask = y[mask]
    pi = float(np.mean(y_mask)) if mask.any() else 0.5
    best_agree = 2.0
    best_pred = np.zeros(int(mask.sum()), dtype=int)
    best_sign = 1
    n_mask = int(mask.sum())
    if n_mask == 0:
        return best_pred, 1.0, best_sign
    for sign in (1.0, -1.0):
        sc = sign * score[mask]
        if pi <= 0.0:
            pred = np.zeros(n_mask, dtype=int)
        elif pi >= 1.0:
            pred = np.ones(n_mask, dtype=int)
        else:
            cutoff = np.quantile(sc, 1.0 - pi)
            pred = (sc > cutoff).astype(int)
        agree = float(np.mean(pred == y_mask))
        if agree < best_agree:
            best_agree = agree
            best_pred = pred
            best_sign = int(sign)
    return best_pred, best_agree, best_sign


def _rule_score(z: np.ndarray, kind: str, beta: np.ndarray | None) -> np.ndarray:
    z1, z2, z3 = z[:, 0], z[:, 1], z[:, 2]
    if kind == "linear":
        b = np.asarray(beta, dtype=float)
        return b[0] * z1 + b[1] * z2
    if kind == "product":
        return z1 * z2
    if kind == "sin_square":
        return np.sin(z1) + np.square(z3)
    raise ValueError(kind)


def simulate_ten_sites(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    roles: dict[str, str],
    cfg: TenSiteConfig | None = None,
) -> dict:
    """Split one table into the ten-site plan and rewrite labels after the burn-in."""
    cfg = TenSiteConfig() if cfg is None else cfg
    rng = np.random.default_rng(cfg.seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int).ravel()
    missing = [name for name in roles.values() if name not in feature_names]
    if missing:
        raise ValueError(f"role columns not in the table: {missing}")
    need = cfg.n_per_site * len(SITE_PLAN)
    if x.shape[0] < need:
        raise ValueError(f"need at least {need} rows, got {x.shape[0]}")

    order = rng.permutation(x.shape[0])[:need]
    x_s = _winsorize_zscore(x[order])
    y_s = y[order]
    names = list(feature_names)
    x1 = names.index(roles["x1"])
    x2 = names.index(roles["x2"])
    x3 = names.index(roles["x3"])
    threshold = float(np.quantile(x_s[:, x1], cfg.subgroup_q))
    subgroup = x_s[:, x1] > threshold
    chunks = rng.permutation(need).reshape(len(SITE_PLAN), cfg.n_per_site)
    batch_size = cfg.n_per_site // cfg.n_batches
    sites = []
    planted: dict[str, list[int]] = {}
    for (site_id, site_type, spec), local_idx in zip(SITE_PLAN, chunks):
        local_idx = rng.permutation(local_idx)
        xs = x_s[local_idx]
        y_base = y_s[local_idx].copy()
        batch = np.arange(cfg.n_per_site) // batch_size
        sub = subgroup[local_idx]
        activate = rng.random(cfg.n_per_site) < (cfg.alpha * _activation_weight(batch, cfg))
        y_new = y_base.copy()
        if site_type == "clean":
            planted[site_id] = []
        elif site_type == "noise":
            planted[site_id] = []
            y_new[activate] = rng.integers(0, 2, size=int(activate.sum()))
        else:
            if site_type == "linear":
                kind, beta, rule_mask = "linear", np.asarray(spec, dtype=float), sub
                planted[site_id] = [x1, x2]
            else:
                kind, beta = str(spec), None
                rule_mask = sub if cfg.nonlinear_scope == "subgroup" else np.ones(cfg.n_per_site, dtype=bool)
                planted[site_id] = [x1, x2] if kind == "product" else [x1, x3]
            score = _rule_score(xs[:, [x1, x2, x3]], kind, beta)
            pred, _, _ = _rate_matched_labels(score, y_base, rule_mask)
            y_rule = y_base.copy()
            y_rule[rule_mask] = pred
            y_new[activate & rule_mask] = y_rule[activate & rule_mask]
        sites.append(
            {
                "site_id": site_id,
                "site_type": site_type,
                "X": xs,
                "y": y_new,
                "y_base": y_base,
                "batch": batch,
                "planted": planted[site_id],
            }
        )
    return {"sites": sites, "planted": planted, "roles": {"x1": x1, "x2": x2, "x3": x3}, "config": cfg, "threshold": threshold}


# ---------------------------------------------------------------------------
# The list. This is the page to read.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DGP:
    name: str
    family: str
    summary: str


CATALOG: tuple[DGP, ...] = (
    DGP("linear_covariate", "site_batch", "高斯两批。新批前四列被同一线性分数平移，条件均值不变。"),
    DGP("nonlinear_covariate", "site_batch", "高斯两批。新批前四列加 sin 和二次项，Y 仍是原来的线性函数。"),
    DGP("linear_concept", "site_batch", "同一批 X 切成两半。后一半 Y 加上前四列的线性漂移。"),
    DGP("nonlinear_concept", "site_batch", "两批同分布的 X。新批的 sin、二次项和交互项放大。"),
    DGP("fsds_stacked_linear_concept", "fsds_stacked", "手稿版概念漂移。返回 (X, Y, W)。第 0 列系数是 0。"),
    DGP("fsds_stacked_linear_covariate", "fsds_stacked", "手稿版协变量漂移。前一半 X 的前四列平移，带状相关。"),
    DGP("fsds_stacked_nonlinear_concept", "fsds_stacked", "手稿版非线性概念漂移。两域各抽一次，叠成 W。"),
    DGP("fsds_stacked_nonlinear_covariate", "fsds_stacked", "手稿版非线性协变量漂移。有一半概率再加一截二次项。"),
    DGP("injection_linear", "injection", "在已有表的新批上加一段线性信号。支持集随机。"),
    DGP("injection_nonlinear", "injection", "在已有表的新批上加 sin 和二次项。X 不动。"),
    DGP("injection_interaction", "injection", "在已有表的新批上加两两乘积。"),
    DGP("injection_nonlinear_interaction", "injection", "sin、二次项、交互，再掷硬币决定要不要把 X 也拧一下。"),
    DGP("feature_rank_indices", "feature_rank", "按某一列排序，从中间切开。TabReD 的九个 num_* 切分用的就是这个。"),
    DGP("choose_ridge_support", "feature_rank", "低半批上 ridge |β| 最大的 k 列，丢掉用来切开的那一列。"),
    DGP("plant_feature_rank_shift", "feature_rank", "四种种植：linear、nonlinear、conceptdrift、covariateshift。尺度 1.5。"),
    DGP("simulate_ten_sites", "ten_site", "十个 site：4 个干净，3 个半空间，2 个乘积或 sin 方，1 个扔硬币。"),
)


def smoke() -> None:
    """Draw one small copy of every family."""
    for name, (_, fn, _) in GENERATORS.items():
        x0, y0, x1, y1 = fn(0)
        print(f"site_batch {name:22s}  X0 {x0.shape}  X1 {x1.shape}")
    x, y, w = fsds_stacked_linear_concept(n=40, p=8, seed=0)
    print(f"fsds_stacked linear_concept   X {x.shape}  W {int(w.sum())}/{len(w)}")
    x, y, w = fsds_stacked_linear_covariate(n=40, p=8, seed=0)
    print(f"fsds_stacked linear_covariate X {x.shape}")
    x, y, w = fsds_stacked_nonlinear_concept(seed=0, p=8, n_per_domain=20)
    print(f"fsds_stacked nonlinear_concept X {x.shape}")
    x, y, w = fsds_stacked_nonlinear_covariate(seed=0, p=8, n_per_domain=20)
    print(f"fsds_stacked nonlinear_covariate X {x.shape}")
    rng = np.random.default_rng(0)
    base_x = rng.normal(size=(60, 8))
    base_y = base_x[:, 0] + rng.normal(size=60)
    for fn in (injection_linear, injection_nonlinear, injection_interaction, injection_nonlinear_interaction):
        df1, df2, support = fn(base_x[:30], base_y[:30], base_x[30:], base_y[30:], seed=0)
        print(f"injection {fn.__name__:32s}  support {np.sort(support).tolist()}  df2 {df2.shape}")
    order = feature_rank_indices(base_x[:, 0])
    support = choose_ridge_support(base_x[order[0]], base_y[order[0]], banned=0, k=3, seed=0)
    for kind in FEATURE_RANK_SHIFTS:
        x1, y1 = plant_feature_rank_shift(
            base_x[order[0]], base_y[order[0]], base_x[order[1]], base_y[order[1]], support, kind, rng
        )
        print(f"feature_rank {kind:16s}  support {support.tolist()}  y1 {y1.shape}")
    toy_x = rng.normal(size=(320, 5))
    toy_y = (toy_x[:, 0] > 0).astype(int)
    out = simulate_ten_sites(
        toy_x,
        toy_y,
        [f"v{i}" for i in range(5)],
        {"x1": "v0", "x2": "v1", "x3": "v2"},
        TenSiteConfig(n_per_site=32, n_batches=16, t_star=8, seed=0),
    )
    kinds = ",".join(site["site_type"] for site in out["sites"])
    print(f"ten_site  {kinds}")


def main() -> None:
    parser = argparse.ArgumentParser(description="List every DGP used on this machine.")
    parser.add_argument("--smoke", action="store_true", help="Draw one small copy of each family.")
    args = parser.parse_args()
    width = max(len(item.name) for item in CATALOG)
    current = ""
    for item in CATALOG:
        if item.family != current:
            print(f"\n[{item.family}]")
            current = item.family
        print(f"  {item.name:{width}s}  {item.summary}")
    if args.smoke:
        print("\n[smoke]")
        smoke()


if __name__ == "__main__":
    main()
