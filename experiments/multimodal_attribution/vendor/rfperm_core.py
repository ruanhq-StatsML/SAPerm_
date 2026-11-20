"""RFPerm core aligned with ``ruanhq-StatsML/RFPerm``.

Sources
-------
- ``Python/RFPerm_randomForest.py`` — continuous RF + OOB vs new-batch loss
- ``R/R/perm_oobtest.R`` + ``R/R/permTest_.R`` — canonical one-sided p-value

The upstream Python ``permTest`` loop had a bug (permuted vector unused). This
module implements the R statistic:

    d0 = mean(test_error) - mean(oob_error)
    p  = (# {d_perm >= d0} + 1) / (n_perm + 1)

with exchangeable group sizes matching the OOB / test lengths (same orientation
as ``d0``). Returns a **single p-value** (for Monte Carlo power boards), not the
``B``-resample power wrapper in the demo scripts.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ArrayLike = Union[np.ndarray, pd.DataFrame]


def mse_loss(y_hat, y_target) -> np.ndarray:
    return (np.asarray(y_hat, dtype=float).reshape(-1) - np.asarray(y_target, dtype=float).reshape(-1)) ** 2


def perm_test_oob_vs_pred(
    oob_errors: np.ndarray,
    pred_errors: np.ndarray,
    *,
    n_perm: int = 5000,
    seed: int = 2026,
) -> float:
    """One-sided p-value: is mean(pred) larger than mean(oob)? (R ``perm_oobtest``)."""
    oob = np.asarray(oob_errors, dtype=float).reshape(-1)
    pred = np.asarray(pred_errors, dtype=float).reshape(-1)
    d0 = float(np.mean(pred) - np.mean(oob))
    n_oob = int(oob.size)
    pooled = np.concatenate([oob, pred])
    rng = np.random.default_rng(int(seed))
    exceed = 0
    for _ in range(int(n_perm)):
        perm = rng.permutation(pooled)
        # Same orientation as d0: first block = oob-sized, second = test-sized.
        d_perm = float(np.mean(perm[n_oob:]) - np.mean(perm[:n_oob]))
        if d_perm >= d0 - 1e-12:
            exceed += 1
    return float((exceed + 1.0) / (int(n_perm) + 1.0))


def RFPerm(
    df_exist: ArrayLike,
    df_new: ArrayLike,
    loss: Optional[Callable] = None,
    *,
    n_perm: int = 150,
    seed: int = 0,
    n_estimators: int = 150,
    sample_fraction: float = 0.8,
) -> float:
    """OOB vs new-domain prediction-error permutation test.

    Matches ``Python/RFPerm_randomForest.py`` (continuous branch) hyperparameters
    and ``R/perm_oobtest.R`` p-value semantics.
    """
    loss = loss or mse_loss
    if isinstance(df_exist, np.ndarray):
        df_exist = pd.DataFrame(np.asarray(df_exist, dtype=float))
        df_new = pd.DataFrame(np.asarray(df_new, dtype=float))
    X_exist = np.asarray(df_exist.iloc[:, :-1], dtype=float)
    Y_exist = np.asarray(df_exist.iloc[:, -1], dtype=float).reshape(-1)
    X_new = np.asarray(df_new.iloc[:, :-1], dtype=float)
    Y_new = np.asarray(df_new.iloc[:, -1], dtype=float).reshape(-1)
    n_exist, p = X_exist.shape
    # Python RFPerm_randomForest: min_samples_leaf = round(sqrt(n)/2), max_features=0.33
    # R perm_oobtest: sample.fraction=0.8, mtry ≈ round(ncol/3)
    leaf = max(1, int(round(np.sqrt(n_exist) / 2.0)))
    mtry_frac = float(max(1, int(round((p + 1) / 3.0)))) / float(p)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "RF",
                RandomForestRegressor(
                    n_estimators=int(n_estimators),
                    min_samples_leaf=leaf,
                    max_features=min(1.0, max(mtry_frac, 0.33)),
                    oob_score=True,
                    bootstrap=True,
                    max_samples=float(sample_fraction),
                    random_state=int(seed),
                    n_jobs=1,
                ),
            ),
        ]
    )
    pipe.fit(X_exist, Y_exist)
    rf = pipe.named_steps["RF"]
    y_oob = np.asarray(rf.oob_prediction_, dtype=float).reshape(-1)
    y_hat_new = np.asarray(pipe.predict(X_new), dtype=float).reshape(-1)
    # Official Python call order was (OOB, NEW) but its mean_dif sign flipped vs R.
    # We follow R: compare NEW errors against OOB errors.
    return perm_test_oob_vs_pred(
        loss(y_oob, Y_exist),
        loss(y_hat_new, Y_new),
        n_perm=int(n_perm),
        seed=int(seed) + 17,
    )
