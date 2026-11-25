"""Reference rating scorer used by OnlineRFPerm.

The monitor only needs ``fit`` and ``predict``. Random forest is the default
because OOB error gives a natural ``E_ref`` without a held-out split; Ridge is
kept as a fast fallback for smoke tests.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class RatingModel:
    def __init__(self, kind: str = "rf", seed: int = 0):
        self.kind = kind
        self.seed = seed
        if kind == "rf":
            self.model = RandomForestRegressor(
                n_estimators=80,
                max_depth=8,
                min_samples_leaf=5,
                max_features="sqrt",
                n_jobs=1,
                oob_score=True,
                bootstrap=True,
                random_state=seed,
            )
        elif kind == "ridge":
            self.model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("ridge", Ridge(alpha=1.0)),
                ]
            )
        else:
            raise ValueError(kind)
        self.e_ref: float | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, holdout: float = 0.25) -> float:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.kind == "rf":
            self.model.fit(x, y)
            # OOB MSE when available; else in-sample residual.
            if getattr(self.model, "oob_prediction_", None) is not None:
                oob = self.model.oob_prediction_
                self.e_ref = float(np.mean((y - oob) ** 2))
            else:
                pred = self.model.predict(x)
                self.e_ref = float(np.mean((y - pred) ** 2))
            return self.e_ref

        n = len(y)
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(n)
        n_te = max(20, int(n * holdout))
        te, tr = idx[:n_te], idx[n_te:]
        self.model.fit(x[tr], y[tr])
        pred = self.model.predict(x[te])
        self.e_ref = float(np.mean((y[te] - pred) ** 2))
        # Refit on all data for production scoring.
        self.model.fit(x, y)
        return self.e_ref

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(np.asarray(x, dtype=float)), dtype=float)

    def batch_mse(self, x: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(x)
        return float(np.mean((np.asarray(y, dtype=float) - pred) ** 2))
