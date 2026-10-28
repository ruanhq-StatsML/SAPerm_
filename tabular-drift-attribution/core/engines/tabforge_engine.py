"""TabFORGE backend.

The published generator is unconditional. This procedure still draws that
sample, then reorders each target column onto the ranks of the reference
donor matched on today's other features and yesterday's full features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..iterative import conditional_indices
from .base import BaseEngine


class TabFORGEEngine(BaseEngine):
    def __init__(self, seed=0):
        self.seed = seed
        self.model = None
        self.ref = None

    def fit(self, X_ref, col_names, discrete_cols=None):
        from tabforge import TabFORGEGenerator

        self.ref = np.asarray(X_ref, dtype=np.float64)
        self.col_names = list(col_names)
        df = pd.DataFrame(X_ref, columns=col_names)
        self.model = TabFORGEGenerator(task="unsupervision", random_state=self.seed)
        self.model.fit(df)
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
        generated = self.model.generate(n, random_state=self.seed)
        if not isinstance(generated, pd.DataFrame):
            generated = pd.DataFrame(generated, columns=self.col_names)
        donor = self._donor(X_cond, n, col_names_cond, col_names_target, X_yesterday)
        out = np.zeros((n, len(col_names_target)), dtype=np.float32)
        rng = np.random.RandomState(self.seed)
        for i, c in enumerate(col_names_target):
            j = self.col_names.index(c)
            if c in generated.columns and len(generated) >= n and donor is not None:
                order = np.argsort(donor[:, i], kind="mergesort")
                synthetic = np.sort(generated[c].to_numpy()[:n])
                out[order, i] = synthetic
            elif donor is not None:
                out[:, i] = donor[:, i]
            else:
                out[:, i] = np.quantile(self.ref[:, j], rng.random(n))
        return out.astype(np.float32)

    def _donor(self, X_cond, n, col_names_cond, col_names_target, X_yesterday):
        if not col_names_cond and X_yesterday is None:
            return None
        cond = [self.col_names.index(c) for c in col_names_cond]
        query = np.asarray(X_cond, dtype=np.float64)
        yesterday = None if X_yesterday is None else np.asarray(X_yesterday, dtype=np.float64)
        if len(query) != n:
            rng = np.random.RandomState(self.seed)
            take = rng.choice(len(query), size=n, replace=len(query) < n)
            query = query[take]
            if yesterday is not None:
                yesterday = yesterday[take]
        matched = conditional_indices(self.ref, query, cond, yesterday)
        tgt = [self.col_names.index(c) for c in col_names_target]
        return self.ref[matched][:, tgt].astype(np.float32)
