"""Desk / IC-side direct strategies. Prefer Backlog Control + Pacing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .types import Deal, DealStage


@dataclass(frozen=True)
class DeskPolicyConfig:
    max_new_flags_per_open_mandate: float = 6.0
    backlog_days_threshold: float = 3.0
    backlog_soft_days: float = 1.5
    pacing_enabled: bool = True
    sector_share_cap: float = 0.55


@dataclass
class DeskBudget:
    deal_id: str
    coverage_budget: int
    review_room: int
    backlog_days: float
    pace_factor: float
    reason: str


def backlog_days(deal: Deal) -> float:
    return float(deal.pending_review) / max(deal.daily_review_capacity, 1)


def review_room(deal: Deal, cfg: DeskPolicyConfig) -> int:
    if deal.stage in (DealStage.PASSED, DealStage.CLOSED):
        return 0
    # Soft-halt (paused) keeps a narrow critical lane. Zeroing room entirely
    # froze planted/high-risk recovery under sparse (v3 regression).
    if deal.paused:
        return max(int(cfg.max_new_flags_per_open_mandate * 0.35), 1)
    clearable = max(deal.daily_review_capacity - deal.expected_organic_docs_per_day, 0.0)
    headcount = max(deal.openings - deal.filled, 0) * cfg.max_new_flags_per_open_mandate
    room = min(clearable, max(headcount - deal.pending_review, 0.0))
    bd = backlog_days(deal)
    if bd >= cfg.backlog_days_threshold:
        return 0
    if bd >= cfg.backlog_soft_days:
        room *= 0.35
    return max(int(room), 0)


def pace_factor(deal: Deal, cfg: DeskPolicyConfig) -> float:
    if not cfg.pacing_enabled:
        return 1.0
    bd = backlog_days(deal)
    if bd >= cfg.backlog_days_threshold:
        return 0.0
    days = max(deal.days_to_ic, 1)
    base = min(1.0, 1.4 / days + 0.2)
    if bd >= cfg.backlog_soft_days:
        base *= 0.4
    if deal.stage == DealStage.IC:
        base *= 0.5  # IC week: less new coverage, more synthesis
    return max(0.0, min(base, 1.0))


def desk_day_budget(deal: Deal, cfg: DeskPolicyConfig, enabled: bool = True) -> DeskBudget:
    bd = backlog_days(deal)
    if not enabled:
        room = int(max(deal.openings - deal.filled, 0) * cfg.max_new_flags_per_open_mandate)
        return DeskBudget(deal.id, max(room * 12, 0), room, bd, 1.0, "rules_off")
    if deal.stage in (DealStage.PASSED, DealStage.CLOSED) or max(deal.openings - deal.filled, 0) <= 0:
        return DeskBudget(deal.id, 0, 0, bd, 0.0, "closed")
    room = review_room(deal, cfg)
    pace = pace_factor(deal, cfg)
    # Soft-halt: ignore pacing collapse so the critical lane still fires.
    if deal.paused:
        coverage = max(int(room / max(0.25, 0.2) * max(pace, 0.5)), 2)
        return DeskBudget(deal.id, coverage, room, bd, max(pace, 0.5), "critical_lane")
    # Translate review room into coverage impressions (docs pushed to analysts).
    coverage = int(room / max(0.25, 0.2) * pace)  # ~4–5 docs per review slot
    reason = "ok"
    if room == 0 and bd >= cfg.backlog_days_threshold:
        reason = "backlog_hard"
    elif room == 0:
        reason = "no_review_room"
    elif pace < 1.0:
        reason = "pacing"
    return DeskBudget(deal.id, coverage, room, bd, pace, reason)


def apply_sector_dedup(
    budgets: Dict[str, DeskBudget],
    deals: Dict[str, Deal],
    cfg: DeskPolicyConfig,
) -> Dict[str, DeskBudget]:
    by_sector: Dict[str, int] = {}
    for did, b in budgets.items():
        sec = deals[did].sector
        by_sector[sec] = by_sector.get(sec, 0) + b.coverage_budget
    out: Dict[str, DeskBudget] = {}
    for did, b in budgets.items():
        sec = deals[did].sector
        total = by_sector.get(sec, 0)
        siblings = [x for x, d in deals.items() if d.sector == sec]
        if len(siblings) <= 1 or total <= 0:
            out[did] = b
            continue
        cap = max(int(total * cfg.sector_share_cap), 1)
        if b.coverage_budget > cap:
            out[did] = DeskBudget(b.deal_id, cap, b.review_room, b.backlog_days, b.pace_factor, "sector_dedup")
        else:
            out[did] = b
    return out
