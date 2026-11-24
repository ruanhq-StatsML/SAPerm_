"""Prespecified update-selection rules.

None of these rules reads the locked final test. Tie-breaks follow block order
text, audio, visual, then fusion, then head. A missing rejection or a failed
overlap check falls back to no-update. A rejection without a positive block
score falls back to head-only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.cfperm_recsys_adaptation.dgp import BLOCKS
from experiments.cfperm_recsys_adaptation.model import ARM_GROUPS

ALPHA = 0.05
RISK_EPS = 0.01
BLOCK_ARM = {"text": "text_only", "audio": "audio_only", "visual": "visual_only"}
ARM_BLOCK = {arm: block for block, arm in BLOCK_ARM.items()}
INTERACTION_ARM = {
    "I_TA": "pair_TA",
    "I_TV": "pair_TV",
    "I_AV": "pair_AV",
    "I_TAV": "fusion_only",
}
RISK_ORDER = ("text", "audio", "visual", "fusion")
INTERACTION_ORDER = ("I_TA", "I_TV", "I_AV", "I_TAV")
EXHAUSTIVE_ORDER = (
    "no_update",
    "head_only",
    "text_only",
    "audio_only",
    "visual_only",
    "fusion_only",
    "pair_TA",
    "pair_TV",
    "pair_AV",
    "full",
    "targeted_data",
)
RANDOM_ARMS = ("head_only", "text_only", "audio_only", "visual_only", "fusion_only")


@dataclass
class Decision:
    policy: str
    arm: str
    status: str
    groups: tuple[str, ...]
    feature_index: int | None = None
    bin_id: int | None = None
    cut_low: float | None = None
    cut_high: float | None = None
    block: str = ""


def _adapter_group(block: str) -> tuple[str, ...]:
    return ARM_GROUPS[BLOCK_ARM[block]]


def cfperm_guided(diag: dict) -> Decision:
    if diag["insufficient_overlap"]:
        return Decision("cfperm_guided", "no_update", "insufficient_evidence", ())
    p_value = diag["global_pvalue"]
    if not np.isfinite(p_value) or p_value >= ALPHA:
        return Decision("cfperm_guided", "no_update", "no_rejection", ())
    scores = diag["shift_score"]
    best = max(BLOCKS, key=lambda block: (scores[block], -BLOCKS.index(block)))
    if not np.isfinite(scores[best]) or scores[best] <= 0.0:
        return Decision("cfperm_guided", "head_only", "nonpositive_shift", ARM_GROUPS["head_only"], block=best)
    return Decision(
        "cfperm_guided",
        BLOCK_ARM[best],
        "block",
        _adapter_group(best),
        block=best,
    )


def cfperm_interaction(diag: dict) -> Decision:
    base = cfperm_guided(diag)
    base.policy = "cfperm_interaction"
    if base.status in ("insufficient_evidence", "no_rejection"):
        return base
    contrasts = diag["interactions"]
    magnitude = {name: abs(float(contrasts[name])) for name in INTERACTION_ORDER}
    best_name = max(INTERACTION_ORDER, key=lambda name: (magnitude[name], -INTERACTION_ORDER.index(name)))
    best_shift = max(float(diag["shift_score"][block]) for block in BLOCKS)
    if np.isfinite(magnitude[best_name]) and magnitude[best_name] > best_shift and magnitude[best_name] > 0.0:
        arm = INTERACTION_ARM[best_name]
        return Decision("cfperm_interaction", arm, "interaction", ARM_GROUPS[arm], block=best_name)
    return base


def _best_bin(rows: list[dict]) -> dict | None:
    finite = [row for row in rows if np.isfinite(row["priority"])]
    if not finite:
        return None
    return max(finite, key=lambda row: (row["priority"], -row["bin"], -row["feature_index"]))


def cfperm_localization(diag: dict) -> Decision:
    base = cfperm_guided(diag)
    if base.arm not in BLOCK_ARM.values():
        return Decision("cfperm_localization", base.arm, base.status, base.groups, block=base.block)
    rows = [row for row in diag["localization"] if row["block"] == base.block]
    best = _best_bin(rows)
    if best is None:
        return Decision("cfperm_localization", base.arm, base.status, base.groups, block=base.block)
    priorities = [row["priority"] for row in rows if np.isfinite(row["priority"])]
    median = float(np.median(priorities))
    confirmed = np.isfinite(best["confirm_loss_gap"]) and best["confirm_loss_gap"] > 0.0
    concentrated = best["priority"] > 0.0 and best["priority"] + 1e-12 >= 2.0 * median
    if confirmed and concentrated and best["n_target"] > 0:
        return Decision(
            "cfperm_localization",
            "targeted_data",
            "localized",
            _adapter_group(base.block),
            feature_index=int(best["feature_index"]),
            bin_id=int(best["bin"]),
            cut_low=float(best["cut_low"]),
            cut_high=float(best["cut_high"]),
            block=base.block,
        )
    return Decision("cfperm_localization", base.arm, base.status, base.groups, block=base.block)


def mmd_only(diag: dict) -> Decision:
    significant = [block for block in BLOCKS if diag["mmd_pvalue"][block] < ALPHA]
    if not significant:
        return Decision("mmd_only", "no_update", "no_rejection", ())
    best = min(significant, key=lambda block: (diag["mmd_pvalue"][block], BLOCKS.index(block)))
    return Decision("mmd_only", BLOCK_ARM[best], "mmd", _adapter_group(best), block=best)


def predictive_risk_only(diag: dict) -> Decision:
    gap = {
        name: diag["risk_target"][name] - diag["risk_source"][name]
        for name in ("T", "A", "V", "TAV")
    }
    if not np.isfinite(gap["TAV"]) or gap["TAV"] <= RISK_EPS:
        return Decision("predictive_risk_only", "no_update", "no_degradation", ())
    scores = {
        "text": gap["T"],
        "audio": gap["A"],
        "visual": gap["V"],
        "fusion": gap["TAV"] - max(gap["T"], gap["A"], gap["V"], 0.0),
    }
    best = max(RISK_ORDER, key=lambda name: (scores[name], -RISK_ORDER.index(name)))
    if not np.isfinite(scores[best]) or scores[best] <= RISK_EPS:
        return Decision("predictive_risk_only", "head_only", "diffuse_risk", ARM_GROUPS["head_only"])
    if best == "fusion":
        return Decision("predictive_risk_only", "fusion_only", "risk", ARM_GROUPS["fusion_only"], block="fusion")
    return Decision("predictive_risk_only", BLOCK_ARM[best], "risk", _adapter_group(best), block=best)


def random_module(rng: np.random.Generator) -> Decision:
    arm = RANDOM_ARMS[int(rng.integers(len(RANDOM_ARMS)))]
    return Decision("random_module", arm, "random", ARM_GROUPS[arm], block=ARM_BLOCK.get(arm, ""))


def periodic_full() -> Decision:
    return Decision("periodic_full", "full", "periodic", ARM_GROUPS["full"], block="all")


def exhaustive_validation(val_brier: dict[str, float]) -> Decision:
    def key(arm: str):
        risk = val_brier.get(arm, np.inf)
        if not np.isfinite(risk):
            risk = np.inf
        return (risk, EXHAUSTIVE_ORDER.index(arm))

    arm = min(EXHAUSTIVE_ORDER, key=key)
    block = ARM_BLOCK.get(arm, "")
    groups = ARM_GROUPS[arm] if arm in ARM_GROUPS else ()
    return Decision("exhaustive_validation", arm, "validation", groups, block=block)


def targeted_training_decision(diag: dict) -> Decision:
    """Arm that adapts the rank-1 loss block on its highest-priority bin.

    This is a candidate for the localization policy and for exhaustive
    validation. It does not itself decide that the bin is worth selecting.
    """
    block = diag["top_block"]
    rows = [row for row in diag["localization"] if row["block"] == block]
    best = _best_bin(rows)
    if best is None or best["n_target"] <= 0:
        return Decision("targeted_training", "no_update", "no_bin", (), block=block)
    return Decision(
        "targeted_training",
        "targeted_data",
        "candidate",
        _adapter_group(block),
        feature_index=int(best["feature_index"]),
        bin_id=int(best["bin"]),
        cut_low=float(best["cut_low"]),
        cut_high=float(best["cut_high"]),
        block=block,
    )


def matches_planted(scenario: str, policy: str, arm: str) -> bool:
    """Descriptive hit against the planted mechanism. Not a guarantee of lower risk."""
    if scenario == "null":
        return arm in ("no_update", "head_only")
    if scenario == "benign_covariate":
        if policy == "mmd_only":
            return arm == "visual_only"
        return arm in ("no_update", "head_only")
    if scenario == "modality_concept":
        return arm == "visual_only"
    if scenario == "cross_modal":
        return arm in ("pair_TV", "fusion_only")
    if scenario == "localized":
        return arm in ("audio_only", "targeted_data")
    if scenario == "mixed":
        return arm in ("visual_only", "pair_TV", "fusion_only", "full")
    return False
