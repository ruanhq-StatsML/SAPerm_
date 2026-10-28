"""One entry point for every generator backend."""

from __future__ import annotations

import numpy as np

from core.engines.base import BaseEngine
from core.iterative import conditional_indices


def _build_ctgan(seed=0, **kwargs):
    from core.engines.ctgan_engine import CTGANEngine
    return CTGANEngine(seed=seed, **kwargs)


def _build_tabforge(seed=0, **kwargs):
    from core.engines.tabforge_engine import TabFORGEEngine
    return TabFORGEEngine(seed=seed, **kwargs)


def _build_diffltm(seed=0, **kwargs):
    from core.engines.diffltm_engine import DiffLTMEngine
    return DiffLTMEngine(seed=seed, **kwargs)


GENERATIVE_MODELS = {
    "stub": lambda seed=0, **kwargs: _StubEngine(seed=seed),
    "ctgan": _build_ctgan,
    "tabforge": _build_tabforge,
    "diffltm": _build_diffltm,
}


class CounterfactualGenerator:
    def __init__(self, backend="stub", seed=0, **kwargs):
        self.backend = backend
        self.seed = seed
        self._engine = None
        self._kwargs = kwargs

    def fit(self, X_ref, col_names, discrete_cols=None):
        try:
            build = GENERATIVE_MODELS[self.backend]
        except KeyError:
            raise ValueError(self.backend) from None
        self._engine = build(seed=self.seed, **self._kwargs)
        self._engine.fit(X_ref, col_names, discrete_cols)
        return self

    @property
    def methods(self):
        return getattr(self._engine, "methods", None)

    def generate(
        self,
        X_cond,
        X_target_ref,
        n,
        col_names_cond,
        col_names_target,
        X_yesterday=None,
        col_names_yesterday=None,
        method=None,
    ):
        raw = self._engine.generate(
            X_cond,
            n,
            col_names_cond,
            col_names_target,
            X_yesterday=X_yesterday,
            col_names_yesterday=col_names_yesterday,
            method=method,
        )
        return self._engine.match_to_entities(raw, X_target_ref)


class _StubEngine(BaseEngine):
    """Donor from the reference, matched on today and yesterday.

    For each current row, the nearest reference row supplies the target
    block. The match uses today's other columns and yesterday's full row.
    Yesterday on the reference side is the previous reference row.
    """

    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)

    def fit(self, X_ref, col_names, discrete_cols=None):
        del discrete_cols
        self.ref = np.asarray(X_ref, dtype=np.float32)
        self.col_names = list(col_names)
        self.index = {name: i for i, name in enumerate(self.col_names)}
        return self

    def generate(
        self,
        X_cond,
        n,
        col_names_cond,
        col_names_target,
        X_yesterday=None,
        col_names_yesterday=None,
        method=None,
    ):
        del col_names_yesterday, method
        tgt = [self.index[c] for c in col_names_target]
        if not col_names_cond and X_yesterday is None:
            out = np.zeros((n, len(tgt)), dtype=np.float32)
            for j, col in enumerate(tgt):
                ranks = self.rng.random(n)
                out[:, j] = np.quantile(self.ref[:, col], ranks)
            return out
        cond = [self.index[c] for c in col_names_cond]
        query = np.asarray(X_cond, dtype=np.float64)
        yesterday = None if X_yesterday is None else np.asarray(X_yesterday, dtype=np.float64)
        if len(query) != n:
            take = self.rng.choice(len(query), size=n, replace=len(query) < n)
            query = query[take]
            if yesterday is not None:
                yesterday = yesterday[take]
        matched = conditional_indices(self.ref, query, cond, yesterday)
        return self.ref[matched][:, tgt].astype(np.float32)
