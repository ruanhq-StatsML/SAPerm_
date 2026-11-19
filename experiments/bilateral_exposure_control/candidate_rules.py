"""Six candidate-side direct strategies + dynamic frequency table.

Only gates *proactive* marketing. Organic search / apply is never blocked here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .types import Candidate, CandidateStatus, Job


@dataclass(frozen=True)
class CandidatePolicyConfig:
    # Frequency Cap: marketing *touches* (emails/pushes), not job slots.
    max_touches_per_day: int = 1
    max_touches_per_week: int = 5
    # Job slots inside a touch (e.g. one email with three jobs).
    max_job_slots_per_touch: int = 3
    # Cooldown: same job must wait this many days between proactive touches.
    job_cooldown_days: int = 7
    # Company-level Cap: max similar-family jobs from one company per week.
    company_family_cap_per_week: int = 2
    # Pipeline-aware: reduce slots when candidate already has interviews/apps.
    pipeline_interview_soft_cap: int = 2
    # Negative feedback.
    hard_stop_on_unsubscribe: bool = True


# Dynamic frequency: daily *job slot* budget by status (proactive only).
DAILY_JOB_SLOT_BUDGET: Dict[CandidateStatus, int] = {
    CandidateStatus.ACTIVE_HIGH: 5,
    CandidateStatus.ACTIVE_COLD: 3,
    CandidateStatus.MULTI_INTERVIEW: 2,
    CandidateStatus.LOW_RESPONSE: 1,
    CandidateStatus.PAUSED: 0,
}


def _sum_window(by_day: Dict[int, int], day: int, window: int) -> int:
    return sum(by_day.get(d, 0) for d in range(day - window + 1, day + 1) if d >= 0)


def dynamic_daily_job_slots(c: Candidate, cfg: CandidatePolicyConfig) -> int:
    if c.unsubscribed or c.status == CandidateStatus.PAUSED:
        return 0
    base = DAILY_JOB_SLOT_BUDGET.get(c.status, 0)
    if c.consecutive_no_response >= 3:
        base = min(base, 1)
    if c.consecutive_no_response >= 5:
        base = 0
    if c.open_interviews >= cfg.pipeline_interview_soft_cap:
        base = min(base, DAILY_JOB_SLOT_BUDGET[CandidateStatus.MULTI_INTERVIEW])
    if c.open_apps >= 5:
        base = min(base, 1)
    return max(0, base)


def remaining_touch_budget(c: Candidate, day: int, cfg: CandidatePolicyConfig) -> int:
    if c.unsubscribed and cfg.hard_stop_on_unsubscribe:
        return 0
    day_used = c.marketing_touches_by_day.get(day, 0)
    week_used = _sum_window(c.marketing_touches_by_day, day, 7)
    return max(
        0,
        min(
            cfg.max_touches_per_day - day_used,
            cfg.max_touches_per_week - week_used,
        ),
    )


def remaining_job_slot_budget(c: Candidate, day: int, cfg: CandidatePolicyConfig) -> int:
    dyn = dynamic_daily_job_slots(c, cfg)
    used = c.job_slots_by_day.get(day, 0)
    return max(0, dyn - used)


def cooldown_ok(c: Candidate, job_id: str, day: int, cfg: CandidatePolicyConfig) -> bool:
    last = c.last_job_touch_day.get(job_id)
    if last is None:
        return True
    return (day - last) >= cfg.job_cooldown_days


def negative_feedback_blocks(c: Candidate, job_id: str) -> bool:
    if c.unsubscribed:
        return True
    return job_id in c.ignored_jobs


def company_family_ok(
    c: Candidate,
    job: Job,
    day: int,
    recent_pairs: Sequence[Tuple[str, str, int]],
    cfg: CandidatePolicyConfig,
) -> bool:
    """recent_pairs: (candidate_id, company_id|family, day) for this candidate."""
    count = 0
    for cid, fam, d in recent_pairs:
        if cid != c.id:
            continue
        if fam != f"{job.company_id}:{job.family}":
            continue
        if day - d < 7:
            count += 1
    return count < cfg.company_family_cap_per_week


@dataclass
class CandidateGateResult:
    allow: bool
    reason: str
    touch_budget: int
    job_slot_budget: int


def gate_candidate_job(
    c: Candidate,
    job: Job,
    day: int,
    channel: str,
    shared_channel_days: Set[Tuple[str, int]],
    recent_company_family: Sequence[Tuple[str, str, int]],
    cfg: CandidatePolicyConfig,
    enabled: bool = True,
) -> CandidateGateResult:
    """Cross-channel dedup: shared_channel_days holds (candidate_id, day) already touched."""
    touch_budget = remaining_touch_budget(c, day, cfg) if enabled else 99
    slot_budget = remaining_job_slot_budget(c, day, cfg) if enabled else 99

    if not enabled:
        return CandidateGateResult(True, "rules_off", touch_budget, slot_budget)

    if negative_feedback_blocks(c, job.id):
        return CandidateGateResult(False, "negative_feedback", touch_budget, slot_budget)
    if slot_budget <= 0:
        return CandidateGateResult(False, "job_slot_cap", touch_budget, slot_budget)
    # Cross-channel dedup: if any channel already marketed today, further jobs
    # must ride the same touch budget (slot budget still applies).
    already = (c.id, day) in shared_channel_days
    if not already and touch_budget <= 0:
        return CandidateGateResult(False, "frequency_cap", touch_budget, slot_budget)
    if not cooldown_ok(c, job.id, day, cfg):
        return CandidateGateResult(False, "cooldown", touch_budget, slot_budget)
    if not company_family_ok(c, job, day, recent_company_family, cfg):
        return CandidateGateResult(False, "company_family_cap", touch_budget, slot_budget)
    if c.open_interviews >= 4:
        return CandidateGateResult(False, "pipeline_saturated", touch_budget, slot_budget)
    return CandidateGateResult(True, "ok", touch_budget, slot_budget)


def pack_jobs_into_touches(
    candidate_id: str,
    job_ids: List[str],
    day: int,
    channel: str,
    cfg: CandidatePolicyConfig,
) -> List:
    """One email may carry up to max_job_slots_per_touch jobs."""
    from .types import DayBundle

    bundles: List[DayBundle] = []
    for i in range(0, len(job_ids), cfg.max_job_slots_per_touch):
        chunk = job_ids[i : i + cfg.max_job_slots_per_touch]
        bundles.append(DayBundle(candidate_id, channel, day, chunk))
    return bundles
