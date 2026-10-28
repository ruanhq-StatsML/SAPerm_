"""Which columns are conditioned, and which of those are discrete.

A block update conditions on today's other columns and on yesterday's full
row. The block itself is free. A column is discrete when the reference batch
is integer-valued and has at most `max_cardinality` distinct values.
Everything else stays continuous.

That split is the data digestion. It is not the same as a model that bins
continuous inputs so it can take a discrete condition. Only CTGAN does that.
"""

from __future__ import annotations

import numpy as np

from .iterative import LAG_PREFIX, lag_column_names


def _base_name(name: str) -> str:
    if name.startswith(LAG_PREFIX):
        return name[len(LAG_PREFIX):]
    return name


def is_discrete(values, max_cardinality: int = 32) -> bool:
    """True when `values` is a low-cardinality integer code."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0 or max_cardinality < 1:
        return False
    if not np.all(np.abs(values - np.round(values)) <= 1e-6):
        return False
    return int(np.unique(np.round(values)).size) <= max_cardinality


def digest(X, col_names, max_cardinality: int = 32) -> dict:
    """Split a reference batch into discrete and continuous column names."""
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != len(col_names):
        raise ValueError("X columns must match col_names")
    discrete, continuous = [], []
    for i, name in enumerate(col_names):
        if is_discrete(X[:, i], max_cardinality):
            discrete.append(name)
        else:
            continuous.append(name)
    return {"discrete": discrete, "continuous": continuous, "max_cardinality": max_cardinality}


def _is_discrete_name(name: str, discrete: set[str]) -> bool:
    return name in discrete or _base_name(name) in discrete


def block_account(col_names, target_cols, discrete) -> dict:
    """Column lists for one block update.

    `conditioned_today` is every current column outside the block.
    `conditioned_yesterday` is the full lag, including yesterday's copy of
    the free block. Equality names are the discrete ones among those.
    Range names are the continuous ones. `free` is today's block only.
    """
    names = list(col_names)
    target = set(target_cols)
    if any(i < 0 or i >= len(names) for i in target_cols):
        raise ValueError("target column out of range")
    today = [names[i] for i in range(len(names)) if i not in target]
    yesterday = lag_column_names(names)
    discrete = set(discrete or [])
    conditioned = today + yesterday
    equality = [name for name in conditioned if _is_discrete_name(name, discrete)]
    ranges = [name for name in conditioned if name not in set(equality)]
    return {
        "free": [names[i] for i in target_cols],
        "conditioned_today": today,
        "conditioned_yesterday": yesterday,
        "equality": equality,
        "ranges": ranges,
    }


def generator_plan(name: str, account: dict, digested: dict) -> dict:
    """How one generator uses the account. Counts only, no fitting."""
    n_eq = len(account["equality"])
    n_rg = len(account["ranges"])
    n_cond = n_eq + n_rg
    n_free = len(account["free"])
    n_disc = len(digested["discrete"])
    n_cont = len(digested["continuous"])
    if name == "stub":
        return {
            "generator": name,
            "discretize": [],
            "keep_continuous": n_cond,
            "condition_inputs": n_cond,
            "note": f"不离散。在全部 {n_cond} 个条件列上找最近的参考行",
        }
    if name == "ctgan":
        return {
            "generator": name,
            "discretize": n_rg,
            "keep_continuous": 0,
            "condition_inputs": n_cond,
            "note": (
                f"条件里的 {n_eq} 个离散列保持类别；{n_rg} 个连续条件列做分位箱。"
                f"拟合时把 {n_cont} 个连续列和它们的滞后分成 {n_cont * 2} 个箱，"
                f"{n_disc} 个离散列和滞后仍是类别"
            ),
        }
    if name == "tabforge":
        return {
            "generator": name,
            "discretize": [],
            "keep_continuous": n_cond,
            "condition_inputs": n_cond,
            "note": f"不离散。无条件样本按这 {n_cond} 个条件列的参考捐赠行重排名次",
        }
    if name == "diffltm":
        return {
            "generator": name,
            "discretize": [],
            "keep_continuous": n_rg,
            "condition_inputs": n_cond,
            "note": (
                f"不离散。schema equality={n_eq}, ranges={n_rg}, free={n_free}，"
                "weights 空，bounds 空。SMC、Harpoon、rejection 读同一份"
            ),
        }
    raise ValueError(name)
