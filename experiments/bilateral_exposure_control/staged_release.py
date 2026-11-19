"""Two-stage / multi-stage traffic release without pre-committing total impressions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .types import Job


class StageDiagnosis(str, Enum):
    SUPPLY_SHORTAGE = "supply_shortage"  # need more exposure
    MATCHING_FAILURE = "matching_failure"  # enough apps, poor quality → retarget
    ENOUGH_PIPELINE = "enough_pipeline"  # pause further stages
    CONTINUE = "continue"


@dataclass(frozen=True)
class StagePlan:
    stage_index: int
    name: str
    impression_batch: int
    decide_on: str


DEFAULT_STAGES: List[StagePlan] = [
    StagePlan(0, "probe", 200, "application_count_and_quality"),
    StagePlan(1, "qualify", 200, "review_pass_rate"),
    StagePlan(2, "interview", 200, "effective_interviews_and_capacity"),
    StagePlan(3, "fill", -1, "remaining_openings"),  # dynamic
]


@dataclass
class StageDecision:
    release_impressions: int
    diagnosis: StageDiagnosis
    next_stage: int
    note: str


def _apply_rate(job: Job) -> float:
    if job.total_impressions <= 0:
        return 0.0
    return job.total_apps_from_marketing / max(job.total_impressions, 1)


def _qualify_rate(job: Job) -> float:
    if job.total_apps_from_marketing <= 0:
        return 0.0
    return job.total_qualified / max(job.total_apps_from_marketing, 1)


def decide_next_batch(
    job: Job,
    stages: Optional[List[StagePlan]] = None,
    company_impression_cap: int = 10_000,
) -> StageDecision:
    stages = stages or DEFAULT_STAGES
    idx = min(max(job.stage, 0), len(stages) - 1)
    plan = stages[idx]
    openings = max(job.openings - job.filled, 0)
    if openings <= 0:
        return StageDecision(0, StageDiagnosis.ENOUGH_PIPELINE, idx, "filled")

    apps = job.total_apps_from_marketing
    qualified = job.total_qualified
    interviews = job.total_interviews
    apply_r = _apply_rate(job)
    qual_r = _qualify_rate(job)

    # Stage 0: after probe impressions, check volume vs quality.
    if idx == 0 and job.stage_impressions >= plan.impression_batch:
        if apps < 8:
            return StageDecision(
                plan.impression_batch,
                StageDiagnosis.SUPPLY_SHORTAGE,
                0,
                "probe_low_apps_repeat_or_widen",
            )
        if apps >= 8 and qual_r < 0.15:
            return StageDecision(
                0,
                StageDiagnosis.MATCHING_FAILURE,
                0,
                "enough_apps_poor_quality_retarget_not_more_spend",
            )
        return StageDecision(stages[1].impression_batch, StageDiagnosis.CONTINUE, 1, "advance_qualify")

    if idx == 1 and job.stage_impressions >= plan.impression_batch:
        if qual_r >= 0.25 and qualified >= openings * 3:
            return StageDecision(0, StageDiagnosis.ENOUGH_PIPELINE, 2, "pause_enough_qualified")
        if apps >= 20 and qual_r < 0.12:
            return StageDecision(
                0,
                StageDiagnosis.MATCHING_FAILURE,
                1,
                "review_pass_low_retarget",
            )
        return StageDecision(stages[2].impression_batch, StageDiagnosis.CONTINUE, 2, "advance_interview")

    if idx == 2 and job.stage_impressions >= plan.impression_batch:
        if interviews >= openings and job.pending_review < job.daily_review_capacity:
            return StageDecision(0, StageDiagnosis.ENOUGH_PIPELINE, 3, "interview_capacity_ok_pause")
        if interviews < openings and apply_r < 0.03:
            return StageDecision(
                stages[2].impression_batch,
                StageDiagnosis.SUPPLY_SHORTAGE,
                2,
                "need_more_interviewable_supply",
            )
        return StageDecision(-1, StageDiagnosis.CONTINUE, 3, "advance_fill_dynamic")

    # Dynamic fill stage or mid-stage continuation.
    if plan.impression_batch < 0 or idx >= 3:
        # Scale residual release to remaining seats and calendar.
        residual = max(int(openings * 40 * (1.0 / max(job.hiring_days_left, 1))), 0)
        residual = min(residual, company_impression_cap)
        if job.pending_review > job.daily_review_capacity * 3:
            return StageDecision(0, StageDiagnosis.ENOUGH_PIPELINE, idx, "backlog_blocks_fill")
        return StageDecision(residual, StageDiagnosis.CONTINUE, idx, "dynamic_fill")

    # Still inside current stage batch.
    left = max(plan.impression_batch - job.stage_impressions, 0)
    return StageDecision(left, StageDiagnosis.CONTINUE, idx, "continue_stage")


def note_impressions(job: Job, n: int, stages: Optional[List[StagePlan]] = None) -> None:
    stages = stages or DEFAULT_STAGES
    job.total_impressions += n
    job.stage_impressions += n
    plan = stages[min(job.stage, len(stages) - 1)]
    if plan.impression_batch > 0 and job.stage_impressions >= plan.impression_batch:
        # Freeze counters until decide_next_batch advances stage.
        pass


def advance_stage(job: Job, decision: StageDecision) -> None:
    if decision.next_stage != job.stage:
        job.stage = decision.next_stage
        job.stage_impressions = 0
