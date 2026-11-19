"""Five company-side direct strategies. Prefer Backlog Control + Pacing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .types import Job


@dataclass(frozen=True)
class CompanyPolicyConfig:
    # Application capping: max marketing-sourced apps still useful today.
    max_marketing_apps_per_open_seat: float = 8.0
    # Interview capacity: stop when interview funnel is full vs openings.
    interview_buffer_per_open_seat: float = 3.0
    # Backlog: if pending_review / daily_review_capacity exceeds this, throttle.
    backlog_days_threshold: float = 3.0
    # Soft throttle when backlog days in (soft, hard].
    backlog_soft_days: float = 2.0
    # Pacing: fraction of remaining calendar used to shape daily budget.
    pacing_enabled: bool = True
    # Company-level dedup: treat same-family jobs as one exposure pool share.
    family_share_cap: float = 0.6


@dataclass
class CompanyBudget:
    job_id: str
    impression_budget: int
    marketing_app_room: int
    backlog_days: float
    pace_factor: float
    reason: str


def backlog_days(job: Job) -> float:
    cap = max(job.daily_review_capacity, 1)
    return float(job.pending_review) / cap


def remaining_openings(job: Job) -> int:
    return max(job.openings - job.filled, 0)


def interview_room(job: Job, cfg: CompanyPolicyConfig) -> int:
    need = remaining_openings(job)
    soft = int(need * cfg.interview_buffer_per_open_seat)
    return max(job.interview_slots_remaining, 0) + max(soft - job.total_interviews, 0)


def marketing_app_capacity(job: Job, cfg: CompanyPolicyConfig) -> int:
    """Closed-loop room after organic inflow and review backlog."""
    need = remaining_openings(job)
    if need <= 0 or job.paused:
        return 0
    # Review clears some pending; organic fills some of tomorrow's capacity.
    clearable = max(job.daily_review_capacity - job.expected_organic_apps_per_day, 0.0)
    # Desired marketing apps today: do not refill faster than reviewers can clear.
    desired = clearable
    # Also respect headcount-scaled cap.
    headcount_cap = need * cfg.max_marketing_apps_per_open_seat - job.pending_review
    room = min(desired, max(headcount_cap, 0.0))
    bd = backlog_days(job)
    if bd >= cfg.backlog_days_threshold:
        return 0
    if bd >= cfg.backlog_soft_days:
        room *= 0.35
    return max(int(room), 0)


def pace_factor(job: Job, cfg: CompanyPolicyConfig) -> float:
    if not cfg.pacing_enabled:
        return 1.0
    days = max(job.hiring_days_left, 1)
    # Front-load slightly when far from deadline and backlog is calm; otherwise slow.
    bd = backlog_days(job)
    if bd >= cfg.backlog_days_threshold:
        return 0.0
    # Remaining share of calendar: release roughly 1/days of residual budget.
    base = min(1.0, 1.5 / days + 0.15)
    if bd >= cfg.backlog_soft_days:
        base *= 0.4
    return max(0.0, min(base, 1.0))


def impression_budget_from_apps(
    app_room: int,
    expected_apply_rate: float = 0.08,
    pace: float = 1.0,
) -> int:
    rate = max(expected_apply_rate, 1e-3)
    raw = app_room / rate
    return max(int(raw * pace), 0)


def company_day_budget(
    job: Job,
    cfg: CompanyPolicyConfig,
    enabled: bool = True,
    expected_apply_rate: float = 0.08,
) -> CompanyBudget:
    bd = backlog_days(job)
    if not enabled:
        # Uncontrolled: chase applications with a loose fixed tap.
        need = remaining_openings(job)
        apps = int(need * cfg.max_marketing_apps_per_open_seat)
        return CompanyBudget(
            job_id=job.id,
            impression_budget=impression_budget_from_apps(apps, expected_apply_rate, 1.0),
            marketing_app_room=apps,
            backlog_days=bd,
            pace_factor=1.0,
            reason="rules_off",
        )

    if remaining_openings(job) <= 0:
        return CompanyBudget(job.id, 0, 0, bd, 0.0, "filled")
    if interview_room(job, cfg) <= 0 and job.interview_slots_remaining <= 0:
        # Interview Capacity Capping
        return CompanyBudget(job.id, 0, 0, bd, 0.0, "interview_cap")

    apps = marketing_app_capacity(job, cfg)
    pace = pace_factor(job, cfg)
    imps = impression_budget_from_apps(apps, expected_apply_rate, pace)
    reason = "ok"
    if apps == 0 and bd >= cfg.backlog_days_threshold:
        reason = "backlog_hard"
    elif apps == 0:
        reason = "no_app_room"
    elif pace < 1.0:
        reason = "pacing"
    return CompanyBudget(
        job_id=job.id,
        impression_budget=imps,
        marketing_app_room=apps,
        backlog_days=bd,
        pace_factor=pace,
        reason=reason,
    )


def apply_family_dedup(
    budgets: Dict[str, CompanyBudget],
    jobs: Dict[str, Job],
    cfg: CompanyPolicyConfig,
) -> Dict[str, CompanyBudget]:
    """Company-level Deduplication: cap total impressions inside a job family."""
    by_family: Dict[str, int] = {}
    for jid, b in budgets.items():
        job = jobs[jid]
        key = f"{job.company_id}:{job.family}"
        by_family[key] = by_family.get(key, 0) + b.impression_budget

    out: Dict[str, CompanyBudget] = {}
    for jid, b in budgets.items():
        job = jobs[jid]
        key = f"{job.company_id}:{job.family}"
        total = by_family.get(key, 0)
        if total <= 0:
            out[jid] = b
            continue
        # Soft share: no single job eats more than family_share_cap of family pool
        # when siblings exist; rescale if over.
        siblings = [x for x, j in jobs.items() if f"{j.company_id}:{j.family}" == key]
        if len(siblings) <= 1:
            out[jid] = b
            continue
        cap = max(int(total * cfg.family_share_cap), 1)
        if b.impression_budget > cap:
            out[jid] = CompanyBudget(
                b.job_id,
                cap,
                b.marketing_app_room,
                b.backlog_days,
                b.pace_factor,
                "family_dedup",
            )
        else:
            out[jid] = b
    return out
