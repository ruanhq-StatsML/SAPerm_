"""The ready multi-armed bandit model pool.

Nineteen arms, in the benchmark order. ``model_pool()`` returns the first
twelve, which is the default arm count. PermuCATE uses the first ten:
RF1–RF6 and XGB1–XGB4.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor


POOL_NAMES = [
    "RF1", "RF2", "RF3", "RF4", "RF5", "RF6",
    "XGB1", "XGB2", "XGB3", "XGB4", "XGB5", "XGB6",
    "Ridge_a1", "Ridge_a10", "LM",
    "MLP_32_16", "MLP_64_32", "MLP_128_64",
    "KRR",
]


def model_pool(n_arms: int | None = 12) -> list[tuple[str, object]]:
    """Fresh estimators. ``n_arms=None`` returns all nineteen."""
    pool = [
        ("RF1", RandomForestRegressor(n_estimators=50, max_features=0.25, max_depth=3, min_samples_leaf=50, random_state=2020, n_jobs=1)),
        ("RF2", RandomForestRegressor(n_estimators=70, max_features=0.75, max_depth=4, min_samples_leaf=40, random_state=2021, n_jobs=1)),
        ("RF3", RandomForestRegressor(n_estimators=90, max_features=0.25, max_depth=5, min_samples_leaf=30, random_state=2022, n_jobs=1)),
        ("RF4", RandomForestRegressor(n_estimators=110, max_features=0.75, max_depth=6, min_samples_leaf=20, random_state=2023, n_jobs=1)),
        ("RF5", RandomForestRegressor(n_estimators=130, max_features=0.25, max_depth=7, min_samples_leaf=10, random_state=2024, n_jobs=1)),
        ("RF6", RandomForestRegressor(n_estimators=500, max_features=0.75, max_depth=8, min_samples_leaf=5, random_state=2025, n_jobs=1)),
        ("XGB1", XGBRegressor(n_estimators=50, max_depth=3, colsample_bytree=0.33, subsample=0.2, random_state=2012, verbosity=0, n_jobs=1)),
        ("XGB2", XGBRegressor(n_estimators=100, max_depth=5, colsample_bytree=0.4, subsample=0.3, random_state=2014, verbosity=0, n_jobs=1)),
        ("XGB3", XGBRegressor(n_estimators=150, max_depth=7, colsample_bytree=0.5, subsample=0.4, random_state=2016, verbosity=0, n_jobs=1)),
        ("XGB4", XGBRegressor(n_estimators=200, max_depth=9, colsample_bytree=0.6, subsample=0.5, random_state=2018, verbosity=0, n_jobs=1)),
        ("XGB5", XGBRegressor(n_estimators=250, max_depth=11, colsample_bytree=0.7, subsample=0.6, random_state=2020, verbosity=0, n_jobs=1)),
        ("XGB6", XGBRegressor(n_estimators=300, max_depth=13, colsample_bytree=0.8, subsample=0.7, random_state=2022, verbosity=0, n_jobs=1)),
        ("Ridge_a1", Ridge(alpha=1.0, random_state=2026)),
        ("Ridge_a10", Ridge(alpha=10.0, random_state=2026)),
        ("LM", LinearRegression()),
        ("MLP_32_16", MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=200, random_state=2022)),
        ("MLP_64_32", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=2022)),
        ("MLP_128_64", MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=200, random_state=2022)),
        ("KRR", KernelRidge(kernel="rbf", gamma=0.1, alpha=0.15)),
    ]
    if n_arms is None:
        return pool
    return pool[:n_arms]
