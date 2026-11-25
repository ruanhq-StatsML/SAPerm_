"""Vendored RFPerm from github.com/ruanhq-StatsML/RFPerm (Python/).

Primary: ``RFPerm_Generalized.PermValTest`` — fit RF on train split of exist,
compare val losses vs new-batch losses, ``permTest`` on pooled per-sample L2.

Alternate: ``RFPerm_randomForest.RFPerm`` OOB vs new (``rfperm_oob_single``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _infer_response_type(Y: np.ndarray) -> str:
    y = np.asarray(Y).reshape(-1)
    n_cls = len(np.unique(y))
    if n_cls == 2:
        return "binary"
    if n_cls > 6:
        return "continuous"
    return "multinomial"


def L2(y_pred: np.ndarray, y_true: np.ndarray) -> list[float]:
    return ((np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)) ** 2).tolist()


def permTest(
    mse_list_val: np.ndarray | list,
    mse_list_pred: np.ndarray | list,
    n_perm: int = 5000,
    seed: int = 2026,
) -> tuple[float, float]:
    """Fixed: store each permuted mean diff (upstream used stale ``pooled``)."""
    rng = np.random.default_rng(seed)
    val = np.asarray(mse_list_val, dtype=float)
    pred = np.asarray(mse_list_pred, dtype=float)
    n_val, n_pred = val.shape[0], pred.shape[0]
    mean_dif_original = float(np.mean(val) - np.mean(pred))
    pooled = np.empty(n_val + n_pred, dtype=float)
    pooled[:n_val] = val
    pooled[n_val:] = pred
    null = np.zeros(n_perm, dtype=float)
    for i in range(n_perm):
        sh = rng.permutation(pooled)
        null[i] = float(np.mean(sh[:n_val]) - np.mean(sh[n_val:]))
    p_one = (1.0 + np.sum(mean_dif_original > null)) / (n_perm + 1)
    p_two = (1.0 + np.sum(np.abs(mean_dif_original) > np.abs(null))) / (n_perm + 1)
    return float(p_one), float(p_two)


def rfperm_permval_single(
    X_exist: np.ndarray,
    Y_exist: np.ndarray,
    X_new: np.ndarray,
    Y_new: np.ndarray,
    *,
    n_perm: int = 150,
    seed: int = 2026,
    n_estimators: int = 150,
    test_size: float = 0.3,
    alpha: float = 0.05,
) -> dict:
    """One replicate of ``PermValTest`` (Generalized.py) with RandomForest."""
    X_exist = np.asarray(X_exist, dtype=float)
    X_new = np.asarray(X_new, dtype=float)
    Y_exist = np.asarray(Y_exist).reshape(-1)
    Y_new = np.asarray(Y_new).reshape(-1)
    n_exist, p = X_exist.shape
    response_type = _infer_response_type(Y_exist)
    msl = max(1, round(np.sqrt(n_exist) / 2))

    X_train, X_val, y_train, y_val = train_test_split(
        X_exist, Y_exist, test_size=test_size, random_state=seed
    )
    if response_type == "continuous":
        model = RandomForestRegressor(
            min_samples_leaf=msl,
            max_features=0.33,
            n_estimators=n_estimators,
            random_state=seed,
            n_jobs=1,
        )
        model.fit(X_train, y_train)
        pred_val = model.predict(X_val)
        pred_test = model.predict(X_new)
        loss_test = L2(pred_test, Y_new)
        loss_val = L2(pred_val, y_val)
    else:
        model = RandomForestClassifier(
            min_samples_leaf=msl,
            max_features=max(1, int(round(np.sqrt(p)))),
            n_estimators=n_estimators,
            random_state=seed + 1,
            n_jobs=1,
        )
        model.fit(X_train, y_train.astype(int))
        classes = list(model.classes_)
        proba_val = model.predict_proba(X_val)
        proba_test = model.predict_proba(X_new)
        col = classes.index(1) if 1 in classes else 0
        pred_val = proba_val[:, col] if proba_val.ndim == 2 else proba_val
        pred_test = proba_test[:, col] if proba_test.ndim == 2 else proba_test
        loss_test = L2(pred_test, Y_new)
        loss_val = L2(pred_val, y_val)

    # permTest: first arg = test losses, second = val losses (Generalized.py order)
    p_one, p_two = permTest(loss_test, loss_val, n_perm=n_perm, seed=seed + 11)
    stat = float(np.mean(loss_test) - np.mean(loss_val))
    return {
        "method": "RFPerm",
        "variant": "PermValTest",
        "statistic": stat,
        "p_value": float(p_one),
        "p_value_two_sided": float(p_two),
        "reject": bool(p_one < alpha),
        "response_type": response_type,
    }


def rfperm_oob_single(
    X_exist: np.ndarray,
    Y_exist: np.ndarray,
    X_new: np.ndarray,
    Y_new: np.ndarray,
    *,
    n_perm: int = 150,
    seed: int = 2026,
    n_estimators: int = 150,
    alpha: float = 0.05,
) -> dict:
    """One RFPerm replicate (one row of official ``RFPerm``'s ``pval_list``)."""
    X_exist = np.asarray(X_exist, dtype=float)
    X_new = np.asarray(X_new, dtype=float)
    Y_exist = np.asarray(Y_exist).reshape(-1)
    Y_new = np.asarray(Y_new).reshape(-1)
    n_exist, p = X_exist.shape
    response_type = _infer_response_type(Y_exist)
    msl = round(np.sqrt(n_exist) / 2)

    if response_type == "continuous":
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "RF",
                    RandomForestRegressor(
                        min_samples_leaf=max(1, msl),
                        max_features=0.33,
                        n_estimators=n_estimators,
                        oob_score=True,
                        bootstrap=True,
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        pipe.fit(X_exist, Y_exist)
        rf = pipe.named_steps["RF"]
        y_hat_oob = np.asarray(rf.oob_prediction_, dtype=float)
        y_hat_test = pipe.predict(X_new)
        loss_oob = L2(y_hat_oob, Y_exist)
        loss_new = L2(y_hat_test, Y_new)
    else:
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "RF",
                    RandomForestClassifier(
                        min_samples_leaf=max(1, msl),
                        max_features=max(1, int(round(np.sqrt(p)))),
                        n_estimators=n_estimators,
                        oob_score=True,
                        bootstrap=True,
                        random_state=seed + 1,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        pipe.fit(X_exist, Y_exist.astype(int))
        rf = pipe.named_steps["RF"]
        classes = list(rf.classes_)
        oob = rf.oob_decision_function_
        if oob.ndim == 2:
            y_hat_oob = oob[:, classes.index(1)] if 1 in classes else 1.0 - oob[:, 0]
        else:
            y_hat_oob = np.asarray(oob, dtype=float)
        proba = pipe.predict_proba(X_new)
        y_hat_test = proba[:, classes.index(1)] if 1 in classes else 1.0 - proba[:, 0]
        loss_oob = L2(y_hat_oob, Y_exist)
        loss_new = L2(y_hat_test, Y_new)

    p_one, p_two = permTest(loss_oob, loss_new, n_perm=n_perm, seed=seed + 11)
    stat = float(np.mean(loss_oob) - np.mean(loss_new))
    return {
        "method": "RFPerm",
        "variant": "OOB",
        "statistic": stat,
        "power_loss": stat,
        "p_value": float(p_one),
        "p_value_two_sided": float(p_two),
        "reject": bool(p_one < alpha),
        "response_type": response_type,
    }


def rfperm_single_pvalue(*args, **kwargs) -> dict:
    """Default: ``RFPerm_Generalized.PermValTest`` (train/val vs new) + fixed ``permTest``."""
    return rfperm_permval_single(*args, **kwargs)