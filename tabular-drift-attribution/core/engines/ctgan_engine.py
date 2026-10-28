"""CTGAN backend. Discrete conditions are a single batch-level value.

CTGAN samples from one condition dictionary, not a different condition per
row. Continuous conditioning columns are reduced to the batch median bin.
The dictionary is today's other features and yesterday's full features.
Yesterday on the fitted table is the previous reference row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseEngine


class CTGANEngine(BaseEngine):
    def __init__(self, epochs=300, n_bins=10, seed=0):
        self.epochs = epochs
        self.n_bins = n_bins
        self.seed = seed
        self.model = None
        self.bin_edges = {}
        self.discrete_cols = []

    def fit(self, X_ref, col_names, discrete_cols=None):
        from ctgan import CTGAN

        df = pd.DataFrame(X_ref, columns=col_names)
        self.discrete_cols = list(discrete_cols or [])
        original_discrete = set(self.discrete_cols)
        lag = np.roll(np.asarray(X_ref, dtype=np.float64), 1, axis=0)
        for i, c in enumerate(col_names):
            df[f"yest__{c}"] = lag[:, i]
        for c in list(df.columns):
            base = c[len("yest__"):] if c.startswith("yest__") else c
            if c in self.discrete_cols or base in original_discrete:
                if c.startswith("yest__") and c not in self.discrete_cols:
                    self.discrete_cols.append(c)
                continue
            bin_col = f"{c}_bin"
            df[bin_col] = pd.qcut(
                df[c].rank(method="first"),
                q=min(self.n_bins, max(2, df[c].nunique())),
                labels=False,
                duplicates="drop",
            ).astype(int)
            n_edges = int(df[bin_col].max()) + 2
            self.bin_edges[c] = np.quantile(df[c], np.linspace(0, 1, n_edges))
            self.discrete_cols.append(bin_col)
        self.model = CTGAN(epochs=self.epochs, cuda=False)
        self.model.fit(df, self.discrete_cols)
        self.col_names = list(col_names)
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
        del method
        df_cond = pd.DataFrame(X_cond, columns=col_names_cond)
        if X_yesterday is not None:
            df_cond = pd.concat(
                [df_cond, pd.DataFrame(X_yesterday, columns=col_names_yesterday)],
                axis=1,
            )
        conditions = {}
        for c in list(col_names_cond) + list(col_names_yesterday or []):
            if c in self.bin_edges:
                med = float(df_cond[c].median())
                bin_id = int(np.digitize(med, self.bin_edges[c][1:-1]))
                conditions[f"{c}_bin"] = bin_id
            elif c in self.discrete_cols and c in df_cond.columns:
                conditions[c] = df_cond[c].mode().iloc[0]
        gen_df = self.model.sample(n, conditions if conditions else None)
        out = np.zeros((n, len(col_names_target)), dtype=np.float32)
        rng = np.random.RandomState(self.seed)
        for i, c in enumerate(col_names_target):
            if c in self.bin_edges and f"{c}_bin" in gen_df.columns:
                edges = self.bin_edges[c]
                bins = gen_df[f"{c}_bin"].to_numpy()
                bins = np.clip(bins, 0, len(edges) - 2)
                lo = edges[bins]
                hi = edges[bins + 1]
                out[:, i] = rng.uniform(lo, np.maximum(hi, lo + 1e-6))
            elif c in gen_df.columns:
                out[:, i] = gen_df[c].to_numpy()
            else:
                raise KeyError(f"CTGAN sample has no column {c}")
        return out.astype(np.float32)
