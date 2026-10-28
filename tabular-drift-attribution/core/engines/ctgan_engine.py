"""CTGAN backend.

``CTGAN.sample(n, condition_column, condition_value)`` conditions on one
discrete column. A continuous condition uses its training bin as that column.
The returned frame is already inverse-transformed.
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
        discrete = set(discrete_cols or [])
        lag = np.roll(np.asarray(X_ref, dtype=np.float64), 1, axis=0)
        for i, c in enumerate(col_names):
            df[f"yest__{c}"] = lag[:, i]
        discrete_cols_yes = [c for c in df.columns if str(c).split("__")[0] == "yest" and str(c).split("__", 1)[-1] in discrete]
        self.discrete_cols = list(discrete) + discrete_cols_yes
        for c in df.columns:
            if c in self.discrete_cols:
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
        self.model.set_random_state(self.seed)
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
        frame = pd.DataFrame(X_cond, columns=col_names_cond)
        if X_yesterday is not None:
            frame = pd.concat(
                [frame, pd.DataFrame(X_yesterday, columns=col_names_yesterday)],
                axis=1,
            )
        column, value = None, None
        names = list(col_names_cond) + list(col_names_yesterday or [])
        for c in names:
            if c in self.discrete_cols and c not in self.bin_edges and c in frame.columns:
                column, value = c, frame[c].mode().iloc[0]
                break
        if column is None:
            for c in names:
                if c in self.bin_edges:
                    med = float(frame[c].median())
                    column, value = f"{c}_bin", int(np.digitize(med, self.bin_edges[c][1:-1]))
                    break
        self.model.set_random_state(self.seed)
        if column is None:
            generated = self.model.sample(n)
        else:
            generated = self.model.sample(n, condition_column=column, condition_value=value)
        if not isinstance(generated, pd.DataFrame):
            generated = pd.DataFrame(generated)
        missing = [c for c in col_names_target if c not in generated.columns]
        if missing:
            raise KeyError(f"CTGAN sample has no column {missing[0]}")
        return generated[list(col_names_target)].to_numpy(dtype=np.float32)
