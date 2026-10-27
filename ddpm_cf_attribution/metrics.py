"""Distributional distances used by the attribution score and the protocol."""

from __future__ import annotations

import numpy as np
from scipy import stats


def median_bandwidth(x: np.ndarray, y: np.ndarray, cap: int = 256) -> float:
    z = np.concatenate([np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()])
    if z.size > cap:
        z = z[:cap]
    d = np.abs(z[:, None] - z[None, :])
    med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
    return float(max(med, 1e-3))


def mmd2_rbf(x: np.ndarray, y: np.ndarray, sigma: float | None = None) -> float:
    """Biased RBF MMD^2. Non-negative up to numerical error, unlike the U-statistic."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if sigma is None:
        sigma = median_bandwidth(x, y)
    def gram(a, b):
        d2 = (a[:, None] - b[None, :]) ** 2
        return np.exp(-d2 / (2.0 * sigma ** 2))
    return float(gram(x, x).mean() + gram(y, y).mean() - 2.0 * gram(x, y).mean())


def mmd2_rff(
    x: np.ndarray,
    y: np.ndarray,
    sigma: float | None = None,
    n_freq: int = 100,
    rng: np.random.Generator | None = None,
    omega: np.ndarray | None = None,
    bias: np.ndarray | None = None,
) -> float:
    """Random-Fourier MMD^2 for the RBF kernel. D=100 is the protocol default."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if sigma is None:
        sigma = median_bandwidth(x, y)
    rng = np.random.default_rng(0) if rng is None else rng
    if omega is None:
        omega = rng.normal(0.0, 1.0 / sigma, size=n_freq)
        bias = rng.uniform(0.0, 2.0 * np.pi, size=n_freq)
    zx = np.sqrt(2.0 / n_freq) * np.cos(np.outer(x, omega) + bias)
    zy = np.sqrt(2.0 / n_freq) * np.cos(np.outer(y, omega) + bias)
    diff = zx.mean(axis=0) - zy.mean(axis=0)
    return float(np.dot(diff, diff))


def draw_rff(sigma: float, n_freq: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    omega = rng.normal(0.0, 1.0 / sigma, size=n_freq)
    bias = rng.uniform(0.0, 2.0 * np.pi, size=n_freq)
    return omega, bias


def bootstrap_mmd(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 400,
    n_freq: int = 100,
    sigma_mult: float = 1.0,
    seed: int = 0,
) -> dict:
    """Percentile interval for RFF-MMD^2. Fourier features are held fixed."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    sigma = median_bandwidth(x, y) * sigma_mult
    omega, bias = draw_rff(sigma, n_freq, rng)
    point = mmd2_rff(x, y, sigma=sigma, n_freq=n_freq, omega=omega, bias=bias)
    stats_b = np.empty(n_boot)
    n, m = len(x), len(y)
    for b in range(n_boot):
        xb = x[rng.integers(0, n, size=n)]
        yb = y[rng.integers(0, m, size=m)]
        stats_b[b] = mmd2_rff(xb, yb, sigma=sigma, n_freq=n_freq, omega=omega, bias=bias)
    lo, hi = np.quantile(stats_b, [0.025, 0.975])
    return {
        "delta": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "sigma": float(sigma),
        "boot_mean": float(stats_b.mean()),
        "boot_std": float(stats_b.std()),
    }


def wasserstein1(x: np.ndarray, y: np.ndarray) -> float:
    return float(stats.wasserstein_distance(np.asarray(x).ravel(), np.asarray(y).ravel()))


def pairwise_concordance(score: np.ndarray, truth: np.ndarray, min_gap: float) -> dict:
    score = np.asarray(score, dtype=float)
    truth = np.asarray(truth, dtype=float)
    num = den = 0
    n = len(score)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(truth[i] - truth[j]) < min_gap:
                continue
            den += 1
            if (score[i] - score[j]) * (truth[i] - truth[j]) > 0:
                num += 1
    return {"concordance": float(num / den) if den else float("nan"), "n_pairs": int(den)}


def spearman(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return {"rho": float("nan"), "pvalue": float("nan")}
    rho, p = stats.spearmanr(x, y)
    return {"rho": float(rho), "pvalue": float(p)}


def coef_variation(values: list[float] | np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    mu = float(np.mean(v))
    if abs(mu) < 1e-8:
        return float("inf")
    return float(np.std(v, ddof=1) / abs(mu)) if len(v) > 1 else 0.0


def e_value_from_d(cohens_d: float) -> float:
    """VanderWeele–Ding approximation: RR ≈ exp(0.91 |d|), then the E-value."""
    d = abs(float(cohens_d))
    rr = float(np.exp(0.91 * d))
    if rr <= 1.0:
        return 1.0
    return float(rr + np.sqrt(rr * (rr - 1.0)))


def hsic_rbf(x: np.ndarray, y: np.ndarray) -> float:
    """Biased HSIC with median-bandwidth RBF kernels. x is (n, d), y is (n,)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = x.shape[0]
    if x.ndim == 1:
        x = x[:, None]

    def gram(a: np.ndarray) -> np.ndarray:
        # a: (n, d)
        a2 = np.sum(a * a, axis=1)
        d2 = a2[:, None] + a2[None, :] - 2.0 * a @ a.T
        d2 = np.maximum(d2, 0.0)
        flat = d2[np.triu_indices(n, k=1)]
        med = np.median(flat) if flat.size else 1.0
        sigma2 = max(med, 1e-6)
        return np.exp(-d2 / (2.0 * sigma2))

    K = gram(x)
    L = gram(y[:, None])
    H = np.eye(n) - 1.0 / n
    return float(np.trace(K @ H @ L @ H) / (n - 1) ** 2)


def ks_uniformity(pit: np.ndarray) -> dict:
    pit = np.asarray(pit, dtype=float)
    stat, p = stats.kstest(pit, "uniform")
    return {"statistic": float(stat), "pvalue": float(p)}


def knn_distances(query: np.ndarray, cloud: np.ndarray, k: int = 5) -> np.ndarray:
    """Euclidean distance from each query row to its k-th nearest cloud neighbor."""
    # chunk to keep memory bounded
    out = np.empty(query.shape[0])
    chunk = 256
    for i in range(0, query.shape[0], chunk):
        q = query[i:i + chunk]
        d2 = ((q[:, None, :] - cloud[None, :, :]) ** 2).sum(axis=-1)
        part = np.partition(d2, kth=min(k, d2.shape[1] - 1), axis=1)
        out[i:i + chunk] = np.sqrt(part[:, min(k, d2.shape[1] - 1)])
    return out


def fsds_scores(x: np.ndarray, cloud: np.ndarray, k: int = 5) -> np.ndarray:
    """How much each coordinate contributes to the distance from the reference cloud.

    FSDS_j = d(x, cloud) - d(x with coordinate j replaced by the cloud mean, cloud).
    The largest score is the coordinate driving the extrapolation.
    """
    x = np.asarray(x, dtype=float).ravel()
    base = float(knn_distances(x[None, :], cloud, k=k)[0])
    scores = np.zeros(x.size)
    center = cloud.mean(axis=0)
    for j in range(x.size):
        x2 = x.copy()
        x2[j] = center[j]
        scores[j] = base - float(knn_distances(x2[None, :], cloud, k=k)[0])
    return scores
