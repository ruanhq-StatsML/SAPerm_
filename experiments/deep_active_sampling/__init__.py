"""Weighted deep active sampling.

Query rules follow the supervised strategies in
cure-lab/deep-active-learning (entropy, margin, least confidence, core-set,
k-means, BADGE, random). Each rule takes a sample-weight vector. The weights
come from the out-of-distribution scores in ``ood_vimp``: PermuCATE
pseudo-outcome importance, a random-forest domain classifier, and MMD
leave-one-covariate-out.
"""

from experiments.deep_active_sampling.query_strategies import ACTIVE_SAMPLING_SCHEMES
from experiments.deep_active_sampling.sampling import deep_active_sampling

__all__ = ["ACTIVE_SAMPLING_SCHEMES", "deep_active_sampling"]
