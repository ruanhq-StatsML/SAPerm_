"""Query rules from cure-lab/deep-active-learning, each with a sample weight.

Rules that only replace the trainer (VAAL, WAAL, learning-loss, the GCN pair,
and the semi-supervised loops) are not registered. They do not share this
network. Every registered rule accepts ``weights``. ``po_risk`` is the
per-row pseudo-outcome residual from the reference nuisance fit.
"""

from experiments.deep_active_sampling.query_strategies.adversarial import (
    AdversarialBIM,
    AdversarialDeepFool,
)
from experiments.deep_active_sampling.query_strategies.albl import ActiveLearningByLearning
from experiments.deep_active_sampling.query_strategies.diversity import (
    BadgeSampling,
    BaselineSampling,
    ClusterMarginSampling,
    CoreSet,
    KCenterGreedy,
    KMeansSampling,
    RandomSampling,
    TopWeight,
)
from experiments.deep_active_sampling.query_strategies.scoring import (
    bald_scores,
    combine_acquisition,
    entropy_scores,
    rank_pct,
    top_indices,
)
from experiments.deep_active_sampling.query_strategies.uncertainty import (
    BALDDropout,
    BatchBALD,
    EntropySampling,
    EntropySamplingDropout,
    LeastConfidence,
    LeastConfidenceDropout,
    MarginSampling,
    MarginSamplingDropout,
)

STRATEGIES = {
    "random": RandomSampling,
    "entropy": EntropySampling,
    "margin": MarginSampling,
    "least_confidence": LeastConfidence,
    "entropy_dropout": EntropySamplingDropout,
    "margin_dropout": MarginSamplingDropout,
    "least_confidence_dropout": LeastConfidenceDropout,
    "bald": BALDDropout,
    "batch_bald": BatchBALD,
    "coreset": CoreSet,
    "kcenter": KCenterGreedy,
    "kmeans": KMeansSampling,
    "cluster_margin": ClusterMarginSampling,
    "badge": BadgeSampling,
    "baseline": BaselineSampling,
    "adversarial_bim": AdversarialBIM,
    "adversarial_deepfool": AdversarialDeepFool,
    "albl": ActiveLearningByLearning,
    "top_weight": TopWeight,
}


def build_strategy(name: str):
    try:
        return STRATEGIES[name]()
    except KeyError as exc:
        known = ", ".join(sorted(STRATEGIES))
        raise KeyError(f"unknown query strategy {name!r}; known strategies: {known}") from exc


def schemes_with_po_risk(names=None) -> dict:
    """Each rule once unweighted and once with the pseudo-outcome residual."""
    if names is None:
        names = [name for name in STRATEGIES if name != "top_weight"]
    schemes = {}
    for name in names:
        schemes[name] = {"strategy": name, "weight": "uniform"}
        schemes[f"{name}_po"] = {"strategy": name, "weight": "po_risk"}
    return schemes


# Default dictionary passed into ``deep_active_sampling``.
ACTIVE_SAMPLING_SCHEMES = schemes_with_po_risk()

__all__ = [
    "ACTIVE_SAMPLING_SCHEMES",
    "STRATEGIES",
    "build_strategy",
    "schemes_with_po_risk",
    "bald_scores",
    "combine_acquisition",
    "entropy_scores",
    "rank_pct",
    "top_indices",
]
