"""One sweep that rewrites each feature block in place.

Block b is redrawn from the reference conditional of two things: the other
columns of the working row (today's other features, with earlier blocks
already replaced) and yesterday's full feature vector. The next block sees
that write. Yesterday stays the observed previous day.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


LAG_PREFIX = "yest__"


def lag_column_names(col_names: list[str]) -> list[str]:
    return [f"{LAG_PREFIX}{name}" for name in col_names]


def block_update(generate_block, today, yesterday, blocks, n_sweeps: int = 1):
    """Return today's rows after `n_sweeps` passes over `blocks`.

    `generate_block(state, yesterday, cols)` returns a row-aligned array
    with shape `(n, len(cols))`.
    """
    state = np.array(today, dtype=np.float32, copy=True)
    yesterday = np.asarray(yesterday, dtype=np.float32)
    if yesterday.shape != state.shape:
        raise ValueError(
            f"yesterday shape {yesterday.shape} does not match today {state.shape}"
        )
    if n_sweeps < 1:
        raise ValueError("n_sweeps must be positive")
    for _ in range(n_sweeps):
        for cols in blocks.values():
            cols = list(cols)
            drawn = np.asarray(generate_block(state, yesterday, cols), dtype=np.float32)
            if drawn.shape != (len(state), len(cols)):
                raise ValueError(
                    f"block draw has shape {drawn.shape}, expected {(len(state), len(cols))}"
                )
            state[:, cols] = drawn
    return state


def conditional_indices(ref, query_today, today_idx, query_yesterday=None):
    """Nearest reference row on today's selected columns and a rolled lag.

    The lag attached to reference row i is reference row i-1 (cyclic). The
    index that comes back points at the current reference row, which is the
    donor for the target block.
    """
    ref = np.asarray(ref, dtype=np.float64)
    pieces_ref = []
    pieces_query = []
    if len(today_idx):
        pieces_ref.append(ref[:, list(today_idx)])
        pieces_query.append(np.asarray(query_today, dtype=np.float64))
    if query_yesterday is not None:
        pieces_ref.append(np.roll(ref, 1, axis=0))
        pieces_query.append(np.asarray(query_yesterday, dtype=np.float64))
    if not pieces_ref:
        raise ValueError("a conditional match needs today's other features or yesterday")
    ref_cond = np.hstack(pieces_ref)
    query = np.hstack(pieces_query)
    if len(query) != len(pieces_query[0]):
        raise ValueError("conditioning blocks must share a row count")
    mu = ref_cond.mean(axis=0)
    sd = ref_cond.std(axis=0) + 1e-6
    dist = cdist((query - mu) / sd, (ref_cond - mu) / sd)
    return dist.argmin(axis=1)
