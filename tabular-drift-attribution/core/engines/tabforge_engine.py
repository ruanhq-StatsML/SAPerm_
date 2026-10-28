"""TabFORGE backend.

The published generator is unconditional. Conditioning is approximated by
rank-matching each target column to the reference marginal. A drift that
changes only P(block | rest) and leaves the marginal alone is invisible here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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

    def generate(self, X_cond, n, col_names_cond, col_names_target):
        del X_cond, col_names_cond
        generated = self.model.generate(n, random_state=self.seed)
        if not isinstance(generated, pd.DataFrame):
            generated = pd.DataFrame(generated, columns=self.col_names)
        out = np.zeros((n, len(col_names_target)), dtype=np.float32)
        rng = np.random.RandomState(self.seed)
        for i, c in enumerate(col_names_target):
            j = self.col_names.index(c)
            if c in generated.columns and len(generated) >= n:
                order = np.argsort(generated[c].to_numpy()[:n])
                donor = np.quantile(self.ref[:, j], np.linspace(0, 1, n, endpoint=False))
                out[order, i] = donor
            else:
                out[:, i] = np.quantile(self.ref[:, j], rng.random(n))
        return out.astype(np.float32)
