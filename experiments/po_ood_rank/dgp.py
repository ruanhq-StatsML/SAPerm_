"""Ranked concept-drift and covariate-shift generators.

The second half of each sample is the shifted batch. A ramp from 0 to the
shift level multiplies the planted perturbation, so the OOD magnitude on that
half is ordered by the row index. The first half is the reference batch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from experiments.dgps import _fsds_base_beta, _fsds_nonlinear_delta, banded_toeplitz_corr


def _ramp(n: int, level: float) -> np.ndarray:
    ramp = np.zeros(n, dtype=float)
    start = n // 2
    ramp[start:] = np.linspace(0.0, float(level), n - start)
    return ramp


def _batch_indicator(n: int) -> np.ndarray:
    return np.concatenate([np.zeros(n // 2, dtype=int), np.ones(n - n // 2, dtype=int)])


def apply_concept_drift(
    X: np.ndarray, delta_beta: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Same covariate law. The second half gains a ramped linear drift.

    The coefficient on coordinate 0 is zero (``arange(4) * delta``), matching
    the manuscript concept-drift generator. ``realized`` is the absolute
    outcome intervention. ``designed`` is the ramp itself.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    beta0 = np.ones(p)
    Y = X @ beta0 + rng.normal(0.0, 1.0, n)
    beta = np.zeros(p)
    beta[: min(4, p)] = np.arange(min(4, p)) * float(delta_beta)
    designed = _ramp(n, float(delta_beta))
    intervention = (X @ beta) * designed
    Y = Y + intervention
    return X, Y, _batch_indicator(n), designed, np.abs(intervention)


def apply_covariate_shift(
    X: np.ndarray, gamma: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fixed regression. The second half is pushed along a ramped score.

    The outcome is drawn after the shift, from the same coefficient vector
    that defines the shift, so the conditional mean given the observed X does
    not change. ``realized`` is the Euclidean displacement of X.
    """
    X = np.asarray(X, dtype=float).copy()
    n, p = X.shape
    beta = np.zeros(p)
    width = min(4, p)
    beta[:width] = float(gamma)
    designed = _ramp(n, float(gamma))
    shift_score = X[:, :width] @ beta[:width]
    delta = shift_score[:, None] * designed[:, None]
    X[:, :width] = X[:, :width] + delta
    realized = np.linalg.norm(delta, axis=1)
    Y = X @ beta + rng.normal(0.0, 1.0, n)
    return X, Y, _batch_indicator(n), designed, realized


def nonlinear_concept_ranked(
    delta_beta: float,
    *,
    n_per_domain: int = 1000,
    p: int = 20,
    rho: float = 0.3,
    seed: int = 0,
    X0: np.ndarray | None = None,
    X1: np.ndarray | None = None,
) -> dict:
    """Nonlinear concept drift with a within-batch ramp on the second domain.

    The second domain's sine, square, and interaction drift is multiplied by
    ``linspace(0, 5)``. ``delta_beta`` raises the scale of that drift from
    ``0.5`` to ``0.5 + delta_beta``. Covariates stay on the same law.
    ``designed`` on the second half is the ramp. ``realized`` is the absolute
    size of the ramped nonlinear term.
    """
    rng = np.random.default_rng(seed)
    beta = _fsds_base_beta(rng, p)
    if X0 is None or X1 is None:
        cov = float(rho) ** np.abs(np.arange(p)[:, None] - np.arange(p)[None, :]) + 1e-6 * np.eye(p)
        X0 = rng.multivariate_normal(np.zeros(p), cov, size=n_per_domain)
        X1 = rng.multivariate_normal(np.zeros(p), cov, size=n_per_domain)
    else:
        X0 = np.asarray(X0, dtype=float)
        X1 = np.asarray(X1, dtype=float)
    n0, n1 = X0.shape[0], X1.shape[0]
    y0 = X0 @ beta + _fsds_nonlinear_delta(X0, rng, 0.5) + rng.normal(0.0, 0.1, n0)
    drift = _fsds_nonlinear_delta(X1, rng, 0.5 + float(delta_beta))
    ramp = np.linspace(0.0, 5.0, n1)
    intervention = drift * ramp
    y1 = X1 @ beta + intervention + rng.normal(0.0, 0.1, n1)
    X = np.vstack([X0, X1])
    Y = np.concatenate([y0, y1])
    W = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
    designed = np.concatenate([np.zeros(n0), ramp])
    realized = np.concatenate([np.zeros(n0), np.abs(intervention)])
    return {
        "X": X,
        "Y": Y,
        "W": W,
        "designed": designed,
        "realized": realized,
        "eval_mask": np.arange(n0 + n1) >= n0,
    }


def gaussian_covariates(n: int, p: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    cov = banded_toeplitz_corr(float(rho), p=p)
    return rng.multivariate_normal(np.zeros(p), cov, size=n)


def draw_ranked(
    family: str,
    shift: float,
    n: int = 1000,
    p: int = 20,
    rho: float = 0.3,
    seed: int = 0,
    source: str = "synthetic",
    embedding_path: str | Path | None = None,
    concept: str = "linear",
) -> dict:
    """One ranked sample. ``family`` is ``concept`` or ``covariate``.

    ``concept="nonlinear"`` uses the ramped FSDS nonlinear drift on the
    second domain. ``n`` is the total sample, split in half.
    """
    if family not in {"concept", "covariate"}:
        raise ValueError(family)
    if source not in {"synthetic", "semisynthetic"}:
        raise ValueError(source)
    if concept not in {"linear", "nonlinear"}:
        raise ValueError(concept)
    if family == "concept" and concept == "nonlinear":
        n_per = n // 2
        if source == "synthetic":
            return nonlinear_concept_ranked(
                shift, n_per_domain=n_per, p=p, rho=rho, seed=seed,
            )
        block = biasfinder_components(n, p, seed, embedding_path)
        return nonlinear_concept_ranked(
            shift, n_per_domain=n_per, p=p, rho=rho, seed=seed,
            X0=block[:n_per], X1=block[n_per:],
        )
    rng = np.random.default_rng(seed)
    if source == "synthetic":
        X = gaussian_covariates(n, p, rho, rng)
    else:
        X = biasfinder_components(n, p, seed, embedding_path)
    if family == "concept":
        X, Y, W, designed, realized = apply_concept_drift(X, shift, rng)
    else:
        X, Y, W, designed, realized = apply_covariate_shift(X, shift, rng)
    return {
        "X": X,
        "Y": Y,
        "W": W,
        "designed": designed,
        "realized": realized,
        "eval_mask": np.arange(n) >= (n // 2),
    }


_EMBEDDING_CACHE: dict[str, np.ndarray] = {}


def load_biasfinder_matrix(path: str | Path | None = None) -> np.ndarray:
    """Numeric BERT embedding columns from the BiasFinder training table."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "datasets" / "train_biasfinder_bert_embedding.csv"
    path = str(path)
    cached = _EMBEDDING_CACHE.get(path)
    if cached is not None:
        return cached
    import pandas as pd

    frame = pd.read_csv(path)
    columns = [c for c in frame.columns if str(c).isdigit()]
    if len(columns) < 8:
        raise RuntimeError(f"no embedding columns in {path}")
    matrix = frame[columns].to_numpy(dtype=float)
    _EMBEDDING_CACHE[path] = matrix
    return matrix


def biasfinder_components(
    n: int,
    p: int,
    seed: int,
    path: str | Path | None = None,
) -> np.ndarray:
    """Whiten a seeded subsample of the BiasFinder embeddings down to ``p`` PCs."""
    matrix = load_biasfinder_matrix(path)
    if n > matrix.shape[0]:
        raise ValueError(f"asked for {n} rows, table has {matrix.shape[0]}")
    rng = np.random.default_rng(seed)
    take = rng.choice(matrix.shape[0], size=n, replace=False)
    block = matrix[take]
    components = min(p, block.shape[0], block.shape[1])
    scores = PCA(n_components=components, whiten=True, random_state=seed).fit_transform(block)
    if components < p:
        pad = np.zeros((n, p - components), dtype=float)
        scores = np.hstack([scores, pad])
    return scores
