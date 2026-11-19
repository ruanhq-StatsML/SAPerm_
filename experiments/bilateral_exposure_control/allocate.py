"""Bilateral exposure allocation under hard candidate/company caps.

Phase-1 score (rule-based proxies, not learned):

    Score_ij = delta_v_ij - C_i^fatigue - C_j^congestion - C_ij^redundancy

Hard constraints (unsubscribe, backlog hard stop, frequency cap) always win
over a high score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .candidate_rules import (
    CandidatePolicyConfig,
    gate_candidate_job,
    pack_jobs_into_touches,
)
from .company_rules import (
    CompanyBudget,
    CompanyPolicyConfig,
    apply_family_dedup,
    company_day_budget,
)
from .staged_release import StageDecision, advance_stage, decide_next_batch, note_impressions
from .types import Candidate, ControlFlags, DayBundle, ExposureDecision, Job


@dataclass
class PairScore:
    candidate_id: str
    job_id: str
    delta_v: float
    fatigue: float
    congestion: float
    redundancy: float
    score: float


def match_delta(c: Candidate, job: Job) -> float:
    # Higher when candidate skill is near / above the bar.
    gap = c.skill - job.bar
    base = 0.08 + 0.25 * max(0.0, min(1.0, 0.5 + gap))
    # Slight boost if candidate is active and responsive.
    if c.status.value.startswith("active"):
        base *= 1.15
    return float(base)


def fatigue_cost(c: Candidate, day: int) -> float:
    touches = c.marketing_touches_by_day.get(day, 0) + sum(
        c.marketing_touches_by_day.get(d, 0) for d in range(max(0, day - 6), day)
    )
    return 0.01 * touches + 0.02 * c.consecutive_no_response


def congestion_cost(job: Job, budget: CompanyBudget) -> float:
    return 0.04 * budget.backlog_days + (0.05 if budget.impression_budget == 0 else 0.0)


def redundancy_cost(c: Candidate, job: Job, day: int) -> float:
    last = c.last_job_touch_day.get(job.id)
    if last is None:
        return 0.0
    gap = day - last
    if gap < 3:
        return 0.05
    if gap < 7:
        return 0.02
    return 0.0


def score_pair(c: Candidate, job: Job, day: int, budget: CompanyBudget) -> PairScore:
    dv = match_delta(c, job)
    f = fatigue_cost(c, day)
    cong = congestion_cost(job, budget)
    red = redundancy_cost(c, job, day)
    return PairScore(c.id, job.id, dv, f, cong, red, dv - f - cong - red)


@dataclass
class AllocationResult:
    decisions: List[ExposureDecision]
    bundles: List[DayBundle]
    budgets: Dict[str, CompanyBudget]
    stage_decisions: Dict[str, StageDecision]
    blocked: Dict[str, int]


def allocate_day(
    day: int,
    candidates: Dict[str, Candidate],
    jobs: Dict[str, Job],
    flags: ControlFlags,
    candidate_cfg: Optional[CandidatePolicyConfig] = None,
    company_cfg: Optional[CompanyPolicyConfig] = None,
    channel: str = "email",
    recent_company_family: Optional[Sequence[Tuple[str, str, int]]] = None,
    expected_apply_rate: float = 0.08,
) -> AllocationResult:
    candidate_cfg = candidate_cfg or CandidatePolicyConfig()
    company_cfg = company_cfg or CompanyPolicyConfig()
    recent_company_family = list(recent_company_family or [])

    raw_budgets = {
        jid: company_day_budget(job, company_cfg, enabled=flags.company_rules, expected_apply_rate=expected_apply_rate)
        for jid, job in jobs.items()
    }
    budgets = apply_family_dedup(raw_budgets, jobs, company_cfg) if flags.company_rules else raw_budgets

    stage_decisions: Dict[str, StageDecision] = {}
    for jid, job in jobs.items():
        sd = decide_next_batch(job)
        stage_decisions[jid] = sd
        if flags.company_rules:
            # Staged release caps the day's impressions.
            release = sd.release_impressions
            if release >= 0:
                b = budgets[jid]
                budgets[jid] = CompanyBudget(
                    b.job_id,
                    min(b.impression_budget, release),
                    b.marketing_app_room,
                    b.backlog_days,
                    b.pace_factor,
                    b.reason if release > 0 else f"stage:{sd.note}",
                )
            advance_stage(job, sd)

    remaining_imps = {jid: b.impression_budget for jid, b in budgets.items()}
    shared_channel_days: Set[Tuple[str, int]] = {
        (cid, day) for cid, c in candidates.items() if c.marketing_touches_by_day.get(day, 0) > 0
    }
    blocked: Dict[str, int] = {}
    scored: List[PairScore] = []

    for c in candidates.values():
        for job in jobs.values():
            gate = gate_candidate_job(
                c,
                job,
                day,
                channel,
                shared_channel_days,
                recent_company_family,
                candidate_cfg,
                enabled=flags.candidate_rules,
            )
            if not gate.allow:
                blocked[gate.reason] = blocked.get(gate.reason, 0) + 1
                continue
            if remaining_imps.get(job.id, 0) <= 0:
                blocked["company_budget"] = blocked.get("company_budget", 0) + 1
                continue
            scored.append(score_pair(c, job, day, budgets[job.id]))

    scored.sort(key=lambda s: s.score, reverse=True)

    # Per-candidate accepted jobs today (before packing into touches).
    accepted: Dict[str, List[str]] = {cid: [] for cid in candidates}
    decisions: List[ExposureDecision] = []

    for s in scored:
        c = candidates[s.candidate_id]
        job = jobs[s.job_id]
        if remaining_imps.get(job.id, 0) <= 0:
            continue
        gate = gate_candidate_job(
            c,
            job,
            day,
            channel,
            shared_channel_days,
            recent_company_family,
            candidate_cfg,
            enabled=flags.candidate_rules,
        )
        if not gate.allow:
            blocked[gate.reason] = blocked.get(gate.reason, 0) + 1
            continue
        if s.score <= 0 and flags.candidate_rules and flags.company_rules:
            blocked["nonpositive_score"] = blocked.get("nonpositive_score", 0) + 1
            continue

        # Commit one impression / job slot.
        accepted[c.id].append(job.id)
        remaining_imps[job.id] -= 1
        note_impressions(job, 1)
        decisions.append(
            ExposureDecision(c.id, job.id, channel, day, s.score, reason="allocated")
        )
        # Update ephemeral candidate counters so subsequent gates see the spend.
        c.job_slots_by_day[day] = c.job_slots_by_day.get(day, 0) + 1
        c.last_job_touch_day[job.id] = day
        fam = f"{job.company_id}:{job.family}"
        recent_company_family = list(recent_company_family) + [(c.id, fam, day)]
        if (c.id, day) not in shared_channel_days:
            # First job today opens one marketing touch.
            c.marketing_touches_by_day[day] = c.marketing_touches_by_day.get(day, 0) + 1
            shared_channel_days.add((c.id, day))

    bundles: List[DayBundle] = []
    for cid, jids in accepted.items():
        if not jids:
            continue
        bundles.extend(pack_jobs_into_touches(cid, jids, day, channel, candidate_cfg))

    return AllocationResult(
        decisions=decisions,
        bundles=bundles,
        budgets=budgets,
        stage_decisions=stage_decisions,
        blocked=blocked,
    )
