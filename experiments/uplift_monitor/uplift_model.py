"""Two-model uplift: separate logistic regressions for treated and control."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class UpliftModel:
    def __init__(self, seed: int):
        self.seed = seed
        self._t1: LogisticRegression | None = None
        self._t0: LogisticRegression | None = None
        self.ready = False

    def fit(self, x: np.ndarray, treat: np.ndarray, y: np.ndarray) -> bool:
        treat = np.asarray(treat).astype(int)
        y = np.asarray(y).astype(int)
        ok = True
        for arm, slot in ((1, "_t1"), (0, "_t0")):
            mask = treat == arm
            if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
                ok = False
                continue
            model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200, random_state=self.seed)
            model.fit(x[mask], y[mask])
            setattr(self, slot, model)
        self.ready = self._t1 is not None and self._t0 is not None and ok
        return self.ready

    def predict_parts(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.ready:
            zeros = np.zeros(len(x))
            return zeros, zeros, zeros
        mu1 = self._t1.predict_proba(x)[:, 1]
        mu0 = self._t0.predict_proba(x)[:, 1]
        return mu0, mu1, mu1 - mu0

    def predict_tau(self, x: np.ndarray) -> np.ndarray:
        return self.predict_parts(x)[2]
