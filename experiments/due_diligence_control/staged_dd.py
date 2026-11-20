"""Staged due-diligence coverage release.

Do not pre-commit "read the whole data room". Release batches and diagnose:

- Supply Shortage: not enough signal / docs reviewed
- Matching Failure: lots of coverage, weak confirmation → retarget questions,
  not more undifferentiated reading
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .types import Deal, DealStage


class StageDiagnosis(str, Enum):
    SUPPLY_SHORTAGE = "supply_shortage"
    MATCHING_FAILURE = "matching_failure"
    ENOUGH_PIPELINE = "enough_pipeline"
    CONTINUE = "continue"


@dataclass(frozen=True)
class StagePlan:
    stage: DealStage
    coverage_batch: int
    decide_on: str


DEFAULT_STAGES: List[StagePlan] = [
    StagePlan(DealStage.SCREEN, 40, "flag_rate_and_volume"),
    StagePlan(DealStage.RED_FLAG, 40, "confirmed_risk_rate"),
    StagePlan(DealStage.DEEP_DIVE, 40, "ic_readiness_and_backlog"),
    StagePlan(DealStage.IC, -1, "residual_mandate"),
]


@dataclass
class StageDecision:
    release_coverage: int
    diagnosis: StageDiagnosis
    next_stage: DealStage
    note: str


def decide_next_batch(deal: Deal, stages: Optional[List[StagePlan]] = None) -> StageDecision:
    stages = stages or DEFAULT_STAGES
    stage_order = [s.stage for s in stages]
    try:
        idx = stage_order.index(deal.stage)
    except ValueError:
        idx = 0
    plan = stages[idx]
    if deal.stage in (DealStage.PASSED, DealStage.CLOSED):
        return StageDecision(0, StageDiagnosis.ENOUGH_PIPELINE, deal.stage, "done")

    flags = deal.total_flags
    confirmed = deal.total_confirmed_risks
    flag_rate = flags / max(deal.total_coverage, 1)
    confirm_rate = confirmed / max(flags, 1)

    if deal.stage == DealStage.SCREEN and deal.stage_coverage >= plan.coverage_batch:
        if flags < 3:
            return StageDecision(
                plan.coverage_batch,
                StageDiagnosis.SUPPLY_SHORTAGE,
                DealStage.SCREEN,
                "too_few_flags_widen_or_reread",
            )
        if flags >= 8 and confirm_rate < 0.15:
            return StageDecision(
                0,
                StageDiagnosis.MATCHING_FAILURE,
                DealStage.SCREEN,
                "noisy_flags_retarget_not_more_spend",
            )
        return StageDecision(stages[1].coverage_batch, StageDiagnosis.CONTINUE, DealStage.RED_FLAG, "to_red_flag")

    if deal.stage == DealStage.RED_FLAG and deal.stage_coverage >= plan.coverage_batch:
        if confirmed >= 3 and deal.pending_review < deal.daily_review_capacity:
            return StageDecision(
                0,
                StageDiagnosis.ENOUGH_PIPELINE,
                DealStage.DEEP_DIVE,
                "pause_enough_confirmed_risks",
            )
        if flags >= 12 and confirm_rate < 0.1:
            return StageDecision(
                0,
                StageDiagnosis.MATCHING_FAILURE,
                DealStage.RED_FLAG,
                "model_noise_change_prompt_or_corpus",
            )
        return StageDecision(stages[2].coverage_batch, StageDiagnosis.CONTINUE, DealStage.DEEP_DIVE, "to_deep_dive")

    if deal.stage == DealStage.DEEP_DIVE and deal.stage_coverage >= plan.coverage_batch:
        if deal.total_ic_ready >= 1 and deal.pending_review <= deal.daily_review_capacity:
            return StageDecision(0, StageDiagnosis.ENOUGH_PIPELINE, DealStage.IC, "ic_ready_pause_new_coverage")
        if deal.total_coverage > 100 and confirmed < 2:
            return StageDecision(
                0,
                StageDiagnosis.MATCHING_FAILURE,
                DealStage.DEEP_DIVE,
                "coverage_without_confirmation",
            )
        return StageDecision(-1, StageDiagnosis.CONTINUE, DealStage.IC, "to_ic_dynamic")

    if deal.stage == DealStage.IC or plan.coverage_batch < 0:
        residual = max(int(8 / max(deal.days_to_ic, 1)), 0)
        if deal.pending_review > deal.daily_review_capacity * 3:
            return StageDecision(0, StageDiagnosis.ENOUGH_PIPELINE, DealStage.IC, "backlog_blocks_ic_coverage")
        return StageDecision(residual, StageDiagnosis.CONTINUE, DealStage.IC, "ic_dynamic")

    left = max(plan.coverage_batch - deal.stage_coverage, 0)
    return StageDecision(left, StageDiagnosis.CONTINUE, deal.stage, "continue_stage")


def note_coverage(deal: Deal, n: int) -> None:
    deal.total_coverage += n
    deal.stage_coverage += n


def advance_stage(deal: Deal, decision: StageDecision) -> None:
    if decision.next_stage != deal.stage:
        deal.stage = decision.next_stage
        deal.stage_coverage = 0
