"""TabFORGE is unconditional. Rank-match each target column onto the donor."""

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

        del discrete_cols
        self.ref = np.asarray(X_ref, dtype=np.float64)
        self.col_names = list(col_names)
        self.model = TabFORGEGenerator(task="unsupervision", random_state=self.seed)
        self.model.fit(pd.DataFrame(X_ref, columns=col_names))
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
        matched = conditional_indices(
            self.ref,
            np.asarray(X_cond),
            [self.col_names.index(c) for c in col_names_cond],
            None if X_yesterday is None else np.asarray(X_yesterday),
        )
        donor = self.ref[matched]
        out = np.empty((n, len(col_names_target)), dtype=np.float32)
        for i, name in enumerate(col_names_target):
            order = np.argsort(donor[:, self.col_names.index(name)], kind="mergesort")
            out[order, i] = np.sort(np.asarray(generated[name])[:n])
        return out
