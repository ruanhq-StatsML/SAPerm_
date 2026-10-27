"""One entry point for every generator backend."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from core.engines.base import BaseEngine


class CounterfactualGenerator:
    def __init__(self, backend="stub", seed=0, **kwargs):
        self.backend = backend
        self.seed = seed
        self._engine = None
        self._kwargs = kwargs

    def fit(self, X_ref, col_names, discrete_cols=None):
        if self.backend == "ctgan":
            from core.engines.ctgan_engine import CTGANEngine
            self._engine = CTGANEngine(seed=self.seed, **self._kwargs)
        elif self.backend == "tabforge":
            from core.engines.tabforge_engine import TabFORGEEngine
            self._engine = TabFORGEEngine(seed=self.seed, **self._kwargs)
        elif self.backend == "diffltm":
            from core.engines.diffltm_engine import DiffLTMEngine
            self._engine = DiffLTMEngine(seed=self.seed, **self._kwargs)
        elif self.backend == "stub":
            self._engine = _StubEngine(seed=self.seed)
        else:
            raise ValueError(self.backend)
        self._engine.fit(X_ref, col_names, discrete_cols)
        return self

    def generate(self, X_cond, X_target_ref, n, col_names_cond, col_names_target):
        raw = self._engine.generate(X_cond, n, col_names_cond, col_names_target)
        return self._engine.match_to_entities(raw, X_target_ref)


class _StubEngine(BaseEngine):
    """Donor from the reference, matched on the conditioning columns.

    For each current row, the nearest reference row in the complement of the
    target block supplies the target block. The condition is the rest of the
    row, not a single lag.
    """

    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)

    def fit(self, X_ref, col_names, discrete_cols=None):
        del discrete_cols
        self.ref = np.asarray(X_ref, dtype=np.float32)
        self.col_names = list(col_names)
        self.index = {name: i for i, name in enumerate(self.col_names)}
        return self

    def generate(self, X_cond, n, col_names_cond, col_names_target):
        tgt = [self.index[c] for c in col_names_target]
        out = np.zeros((n, len(tgt)), dtype=np.float32)
        if not col_names_cond:
            for j, col in enumerate(tgt):
                ranks = self.rng.random(n)
                out[:, j] = np.quantile(self.ref[:, col], ranks)
            return out
        cond = [self.index[c] for c in col_names_cond]
        ref_cond = self.ref[:, cond].astype(np.float64)
        mu = ref_cond.mean(axis=0)
        sd = ref_cond.std(axis=0) + 1e-6
        nn = NearestNeighbors(n_neighbors=1).fit((ref_cond - mu) / sd)
        query = np.asarray(X_cond, dtype=np.float64)
        if len(query) != n:
            take = self.rng.choice(len(query), size=n, replace=len(query) < n)
            query = query[take]
        matched = nn.kneighbors((query - mu) / sd, return_distance=False).ravel()
        return self.ref[matched][:, tgt].astype(np.float32)
