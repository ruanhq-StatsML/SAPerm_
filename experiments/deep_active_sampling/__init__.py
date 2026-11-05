"""Weighted deep active sampling.

Query rules follow the supervised strategies in cure-lab/deep-active-learning.
Each rule takes a sample-weight vector. The comparison weight is the per-row
pseudo-outcome residual from the reference nuisance fit.
"""

from experiments.deep_active_sampling.query_strategies import ACTIVE_SAMPLING_SCHEMES
from experiments.deep_active_sampling.sampling import deep_active_sampling

__all__ = ["ACTIVE_SAMPLING_SCHEMES", "deep_active_sampling"]
