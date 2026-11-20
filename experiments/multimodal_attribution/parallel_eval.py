"""Joblib helpers for overnight power boards (method-level parallelism).

Outer Monte Carlo reps stay sequential for checkpointing; methods inside a rep
are embarrassingly parallel. Set ``OMP_NUM_THREADS=1`` / sklearn ``n_jobs=1``
when using these workers to avoid nested oversubscription.
"""

from __future__ import annotations

import os
import traceback
from typing import Any, Callable, Dict, List, Sequence, Tuple

# Avoid BLAS/OpenMP fighting joblib workers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def _eval_one(name: str, fn: Callable, X1, X2, batch) -> Tuple[str, float, str]:
    try:
        return name, float(fn(X1, X2, batch)), ""
    except Exception as exc:  # noqa: BLE001
        return name, float("nan"), f"{exc}\n{traceback.format_exc()}"


def eval_methods_parallel(
    method_items: Sequence[Tuple[str, Callable]],
    X1,
    X2,
    batch,
    *,
    n_jobs: int = 1,
) -> List[Tuple[str, float, str]]:
    """Evaluate ``[(name, fn), ...]``; returns ``[(name, pvalue, err), ...]``."""
    items = list(method_items)
    if not items:
        return []
    jobs = max(1, int(n_jobs))
    if jobs == 1 or len(items) == 1:
        return [_eval_one(n, f, X1, X2, batch) for n, f in items]
    from joblib import Parallel, delayed

    return Parallel(n_jobs=jobs, prefer="processes")(
        delayed(_eval_one)(n, f, X1, X2, batch) for n, f in items
    )
