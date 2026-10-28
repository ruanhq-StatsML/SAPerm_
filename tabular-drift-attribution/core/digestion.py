"""Discrete iff the reference column is a low-cardinality integer code."""

from __future__ import annotations

import numpy as np

from .iterative import lag_column_names


def is_discrete(values, max_cardinality=32):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0 or max_cardinality < 1:
        return False
    rounded = np.round(values)
    return bool(np.all(np.abs(values - rounded) <= 1e-6) and np.unique(rounded).size <= max_cardinality)


def digest(X, col_names, max_cardinality=32):
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != len(col_names):
        raise ValueError("X columns must match col_names")
    discrete = [n for i, n in enumerate(col_names) if is_discrete(X[:, i], max_cardinality)]
    rest = [n for n in col_names if n not in set(discrete)]
    return {"discrete": discrete, "continuous": rest, "max_cardinality": max_cardinality}


def block_account(col_names, target_cols, discrete):
    names, target, discrete = list(col_names), set(target_cols), set(discrete or [])
    if any(i < 0 or i >= len(names) for i in target):
        raise ValueError("target column out of range")
    today = [names[i] for i in range(len(names)) if i not in target]
    yesterday = lag_column_names(names)
    cond = today + yesterday
    equality = [n for n in cond if n.split("__")[-1] in discrete]
    return {
        "free": [names[i] for i in target_cols],
        "conditioned_today": today,
        "conditioned_yesterday": yesterday,
        "equality": equality,
        "ranges": [n for n in cond if n not in set(equality)],
    }


def generator_plan(name, account, digested):
    del digested
    n_eq, n_rg = len(account["equality"]), len(account["ranges"])
    n = n_eq + n_rg
    if name not in ("stub", "ctgan", "tabforge", "diffltm"):
        raise ValueError(name)
    return {
        "generator": name,
        "discretize": n_rg if name == "ctgan" else [],
        "keep_continuous": 0 if name == "ctgan" else n_rg if name == "diffltm" else n,
        "condition_inputs": n,
        "note": f"equality={n_eq}, ranges={n_rg}, free={len(account['free'])}",
    }
