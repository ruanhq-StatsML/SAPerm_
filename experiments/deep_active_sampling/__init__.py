"""One-pass deep active sampling.

Heavy torch / dataloader imports are lazy so label-free monitors such as
``dependence_strength_shift`` can reuse ``ood_vimp`` and ``model_registry``
without installing the sampling stack.
"""

from __future__ import annotations

__all__ = [
    "QUERY_STRATEGIES",
    "continuous_loader",
    "deep_active_sampling",
    "one_pass",
]


def __getattr__(name: str):
    if name == "QUERY_STRATEGIES":
        from experiments.deep_active_sampling.query_strategies import QUERY_STRATEGIES

        return QUERY_STRATEGIES
    if name == "continuous_loader":
        from experiments.deep_active_sampling.data import continuous_loader

        return continuous_loader
    if name in {"deep_active_sampling", "one_pass"}:
        from experiments.deep_active_sampling import sampling

        return getattr(sampling, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
