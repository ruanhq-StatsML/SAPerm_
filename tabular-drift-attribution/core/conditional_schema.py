"""DiffLTM conditional-sampling schema for one block update.

The wire format is ``Condition.dumps``: ``bounds``, ``equality``, ``ranges``,
and ``weights``, with keys sorted. The block being redrawn is ``free`` and is
absent from those four fields. Today's other columns and yesterday's full row
are the conditioned columns. Categorical ones are equalities, using DiffLTM's
string category codes. Numerical ones are ranges centered on the observed
value. This update does not add a linear constraint, so ``weights`` is empty
and ``bounds`` is null.
"""

from __future__ import annotations

import json
import math

import numpy as np

from .iterative import LAG_PREFIX, lag_column_names


CONDITION_KEYS = ("bounds", "equality", "ranges", "weights")


def _base_name(name: str) -> str:
    if name.startswith(LAG_PREFIX):
        return name[len(LAG_PREFIX):]
    return name


def category_code(value):
    """Match ``diffltm.data.core.category_values`` for one entry."""
    if value is None:
        return "__NA__"
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value):
            return "__NA__"
        if value.is_integer():
            return str(int(value))
        return str(value)
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    text = str(value)
    return "__NA__" if text == "<NA>" else text


def panel_columns(col_names, discrete):
    """Column roles a checkpoint must contain, numerical first.

    Today and yesterday are both in the modeled table. Yesterday's name is
    ``yest__`` plus the current name. A column is categorical when it, or its
    unlagged name, is discrete.
    """
    discrete = set(discrete or [])
    names = list(col_names) + lag_column_names(col_names)
    numerical, categorical = [], []
    for name in names:
        if name in discrete or _base_name(name) in discrete:
            categorical.append(name)
        else:
            numerical.append(name)
    return {"numerical": numerical, "categorical": categorical, "columns": numerical + categorical}


def dumps(condition: dict) -> str:
    payload = {key: condition[key] for key in CONDITION_KEYS}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def row_schema(
    values_today,
    names_today,
    values_yest,
    names_yest,
    discrete,
    scales,
    range_scale,
    free,
):
    """One row: the Condition object, plus the free target columns."""
    discrete = set(discrete or [])
    scales = scales or {}
    free = list(free)
    equality = {}
    ranges = {}

    def put(name, value):
        if name in free:
            return
        base = _base_name(name)
        if name in discrete or base in discrete:
            equality[name] = category_code(value)
            return
        width = max(float(range_scale) * float(scales.get(base, 1.0)), 1e-6)
        center = float(value)
        ranges[name] = [center - width, center + width]

    for name, value in zip(names_today, values_today):
        put(name, value)
    for name, value in zip(names_yest, values_yest):
        put(name, value)
    if not equality and not ranges:
        raise ValueError("a conditional sample needs today's other features or yesterday")
    overlap = set(free) & (set(equality) | set(ranges))
    if overlap:
        raise ValueError(f"free columns are also conditioned: {sorted(overlap)}")
    return {
        "condition": {
            "bounds": None,
            "equality": equality,
            "ranges": ranges,
            "weights": {},
        },
        "free": free,
    }


def batch_schemas(
    X_cond,
    names_cond,
    X_yesterday,
    names_yest,
    discrete,
    scales,
    range_scale,
    free,
):
    X_cond = np.asarray(X_cond)
    X_yesterday = np.asarray(X_yesterday)
    if len(X_cond) != len(X_yesterday):
        raise ValueError("today and yesterday must have the same number of rows")
    return [
        row_schema(
            X_cond[i],
            names_cond,
            X_yesterday[i],
            names_yest,
            discrete,
            scales,
            range_scale,
            free,
        )
        for i in range(len(X_cond))
    ]
