"""One-pass deep active sampling.

A continuous loader yields ``(x, y, index)`` in row order. ``one_pass`` reads
it once. The prefix is the reference window. The remainder is the pool.
Digits, pen digits, and satimage use that same digestion.
"""

from experiments.deep_active_sampling.data import continuous_loader
from experiments.deep_active_sampling.query_strategies import QUERY_STRATEGIES
from experiments.deep_active_sampling.sampling import deep_active_sampling, one_pass

__all__ = [
    "QUERY_STRATEGIES",
    "continuous_loader",
    "deep_active_sampling",
    "one_pass",
]
