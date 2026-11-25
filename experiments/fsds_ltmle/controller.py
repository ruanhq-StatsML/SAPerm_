"""Turn the two FSDS arms into a truncation and a Super Learner penalty.

The output is a priority and a knob change. It is not a license to drop a
confounder. ``DAG_FEATURES`` is copied into the action and nothing is removed
from it, including a feature whose VIMP is zero or whose drift rank is high.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from experiments.fsds_ltmle.covariate_arm import rf_domain_vimp
from experiments.fsds_ltmle.dgp import DAG_FEATURES, monitoring_matrix
from experiments.fsds_ltmle.ltmle import overlap_score
from experiments.fsds_ltmle.permucate import permu_cate_vimp

TRUNC_LOOSE = 0.02
TRUNC_TIGHT = 0.10
TRUNC_RED = 0.12
# The new peak must be at least this many times the stable-batch peak.
# A ratio near 1 treats ordinary permutation noise (a peak of 0.03 against a
# null of 0.02) as concept drift and will hand the Super Learner to the forest
# on a still-linear blip. Real concept drift in this panel moves PO-risk by
# an order of magnitude more than that.
ARM_RATIO = 5.0
OVERLAP_FLOOR = 0.50
# Maps an excess concept signal onto the ridge penalty used in the CV score.
PENALTY_SCALE = 0.75
PENALTY_CAP = 3.0


@dataclass
class Diagnostic:
    names: list[str]
    covariate_vimp: np.ndarray
    concept_vimp: np.ndarray
    overlap: float

    def ranking(self, arm: str) -> list[dict]:
        scores = self.covariate_vimp if arm == "covariate" else self.concept_vimp
        order = np.argsort(-np.asarray(scores, dtype=float))
        return [
            {"feature": self.names[int(i)], "vimp": float(scores[int(i)])}
            for i in order
        ]

    def peak(self, arm: str) -> float:
        scores = self.covariate_vimp if arm == "covariate" else self.concept_vimp
        return float(np.max(scores))


@dataclass
class Action:
    level: str
    truncation: float
    ridge_penalty: float
    pause_dynamic_incentives: bool
    threatened_nuisance: str
    reasons: list[str]
    covariate_ranking: list[dict]
    concept_ranking: list[dict]
    overlap: float
    adjustment_set: list[str]
    dropped_features: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _excess_ratio(peak: float, null_peak: float) -> float:
    scale = max(abs(null_peak), 1e-3)
    return max(0.0, peak - null_peak) / scale


def diagnose(
    batch_exist: dict,
    batch_new: dict,
    seed: int = 0,
    n_estimators: int = 40,
    n_perm: int = 6,
) -> Diagnostic:
    X0, names = monitoring_matrix(batch_exist, seed=seed + 11)
    X1, _ = monitoring_matrix(batch_new, seed=seed + 29)
    Y = np.concatenate([batch_exist["Y"], batch_new["Y"]])
    W = np.concatenate(
        [
            np.zeros(len(batch_exist["Y"]), dtype=int),
            np.ones(len(batch_new["Y"]), dtype=int),
        ]
    )
    X = np.vstack([X0, X1])
    covariate = rf_domain_vimp(X0, X1, n_estimators=n_estimators, n_perm=n_perm, seed=seed)
    concept = permu_cate_vimp(
        X,
        Y,
        W,
        n_perm=n_perm,
        n_estimators=n_estimators,
        seed=seed + 3,
    )
    overlap = overlap_score(batch_new["L"], batch_new["A"], seed=seed)
    return Diagnostic(names=names, covariate_vimp=covariate, concept_vimp=concept, overlap=overlap)


def decide_action(
    diagnostic: Diagnostic,
    null: Diagnostic,
    adjustment_set: tuple[str, ...] | list[str] = DAG_FEATURES,
) -> Action:
    """Map arm peaks to truncation, a ridge penalty, and a human-review flag.

    A red light pauses automatic use of the dynamic regime. The adjustment
    set is returned unchanged. Unique attribution is not claimed: the lists
    are a sort of the two VIMP vectors.
    """
    kept = list(adjustment_set)
    cov_ratio = _excess_ratio(diagnostic.peak("covariate"), null.peak("covariate"))
    concept_ratio = _excess_ratio(diagnostic.peak("concept"), null.peak("concept"))
    covariate_on = cov_ratio >= ARM_RATIO or diagnostic.overlap < OVERLAP_FLOOR
    concept_on = concept_ratio >= ARM_RATIO
    cov_top = diagnostic.ranking("covariate")[:3]
    concept_top = diagnostic.ranking("concept")[:3]
    reasons = [
        (
            f"Covariate arm peak {diagnostic.peak('covariate'):.4f} "
            f"versus null {null.peak('covariate'):.4f} (excess ratio {cov_ratio:.2f}). "
            f"Top features: {', '.join(row['feature'] for row in cov_top)}."
        ),
        (
            f"Concept arm peak {diagnostic.peak('concept'):.4f} "
            f"versus null {null.peak('concept'):.4f} (excess ratio {concept_ratio:.2f}). "
            f"Top features: {', '.join(row['feature'] for row in concept_top)}."
        ),
        f"Share of propensities inside (0.1, 0.9) is {diagnostic.overlap:.3f}.",
    ]
    if covariate_on and concept_on:
        level = "RED"
        truncation = TRUNC_RED
        pause = True
        threatened = "g_and_Q"
        reasons.append(
            "Both arms cleared the null. Pause the dynamic incentive and wait for a person "
            "to confirm before the next automatic update. Do not edit the adjustment set."
        )
    elif concept_on:
        level = "ORANGE_CONCEPT"
        truncation = TRUNC_LOOSE
        pause = False
        threatened = "Q"
        reasons.append(
            "Concept arm only. Penalize ridge inside the Q Super Learner so the forest "
            "carries the blip. Keep every confounder in g and Q."
        )
    elif covariate_on:
        level = "ORANGE_COVARIATE"
        truncation = TRUNC_TIGHT
        pause = False
        threatened = "g"
        reasons.append(
            "Covariate arm or thin overlap. Tighten propensity truncation. "
            "Send the top covariate-shift features to the data owner before retuning Q. "
            "Do not drop them from the adjustment set."
        )
    else:
        level = "GREEN"
        truncation = TRUNC_LOOSE
        pause = False
        threatened = "none"
        reasons.append("Neither arm cleared its null. Keep loose truncation and ordinary cross-validation.")
    penalty = 0.0
    if concept_on:
        penalty = min(PENALTY_CAP, PENALTY_SCALE * concept_ratio)
    reasons.append(
        f"Adjustment set retained: {', '.join(kept)}. Dropped features: none. "
        "Ranks are a priority for review, not a unique causal attribution."
    )
    return Action(
        level=level,
        truncation=truncation,
        ridge_penalty=float(penalty),
        pause_dynamic_incentives=pause,
        threatened_nuisance=threatened,
        reasons=reasons,
        covariate_ranking=diagnostic.ranking("covariate"),
        concept_ranking=diagnostic.ranking("concept"),
        overlap=float(diagnostic.overlap),
        adjustment_set=kept,
        dropped_features=[],
    )
