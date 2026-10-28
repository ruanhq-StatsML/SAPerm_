"""Drop counterfactual rows that leave the reference support.

`density_tau` is relative to the reference cloud. A row is dense enough when
its k-NN distance is below `1 / density_tau` times the reference median
distance. The default 0.5 therefore keeps rows no farther than twice the
typical reference neighbor distance. Distances are computed after column
standardization, so a raw age column does not dominate a binary genre column.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


class QualityGate:
    def __init__(self, density_tau=0.5, range_q=(0.01, 0.99), min_in_range_frac=0.95):
        self.density_tau = density_tau
        self.range_q = range_q
        self.min_in_range_frac = min_in_range_frac

    def fit(self, X_ref):
        X_ref = np.asarray(X_ref, dtype=np.float64)
        self.q_lo = np.quantile(X_ref, self.range_q[0], axis=0)
        self.q_hi = np.quantile(X_ref, self.range_q[1], axis=0)
        self.mu = X_ref.mean(axis=0)
        self.sd = X_ref.std(axis=0) + 1e-6
        Z = self._standardize(X_ref)
        k = min(5, max(1, len(X_ref) - 1))
        self.k = k
        self.nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
        take = np.arange(min(len(X_ref), 800))
        dist, _ = self.nn.kneighbors(Z[take])
        # The fitted point is its own neighbor. Drop that zero.
        self.ref_scale = float(np.median(dist[:, 1:].mean(axis=1))) + 1e-8
        return self

    def _standardize(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mu) / self.sd

    def check(self, X_gen):
        X_gen = np.asarray(X_gen, dtype=np.float64)
        in_range_frac = ((X_gen >= self.q_lo) & (X_gen <= self.q_hi)).mean(axis=1)
        in_range = in_range_frac >= self.min_in_range_frac
        Z = self._standardize(X_gen)
        dist, _ = self.nn.kneighbors(Z)
        # Generated rows are not in the reference fit, so the first neighbor counts.
        rel = self.ref_scale / (dist[:, : self.k].mean(axis=1) + 1e-8)
        is_valid = in_range & (rel > self.density_tau)
        return is_valid, {
            "valid_ratio": float(is_valid.mean()) if len(is_valid) else 0.0,
            "in_range_ratio": float(in_range.mean()) if len(in_range) else 0.0,
        }
