"""Generator interface. fit on a reference batch, then generate a target block."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


class BaseEngine:
    def fit(self, X_ref: np.ndarray, col_names: list, discrete_cols=None):
        raise NotImplementedError

    def generate(
        self,
        X_cond: np.ndarray,
        n: int,
        col_names_cond: list,
        col_names_target: list,
    ) -> np.ndarray:
        """Draw the target columns under the reference conditional of X_cond."""
        raise NotImplementedError

    def match_to_entities(
        self, X_gen: np.ndarray, X_real: np.ndarray,
    ) -> np.ndarray:
        if len(X_gen) == len(X_real):
            return X_gen
        nn = NearestNeighbors(n_neighbors=1).fit(X_gen)
        idx = nn.kneighbors(X_real, return_distance=False).ravel()
        return X_gen[idx]
