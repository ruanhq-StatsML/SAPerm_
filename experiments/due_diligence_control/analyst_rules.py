"""Analyst-side direct strategies (attention protection)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

from .types import Analyst, AnalystLoad, Deal


@dataclass(frozen=True)
class AnalystPolicyConfig:
    max_touches_per_day: int = 2
    max_touches_per_week: int = 8
    max_work_items_per_touch: int = 3  # docs per digest
    deal_cooldown_days: int = 2
    sector_cap_per_week: int = 4
    hard_stop_on_mute: bool = True


DAILY_WORK_ITEM_BUDGET: Dict[AnalystLoad, int] = {
    AnalystLoad.FRESH: 8,
    AnalystLoad.STEADY: 5,
    AnalystLoad.HEAVY: 2,
    AnalystLoad.SATURATED: 1,
    AnalystLoad.OFF: 0,
}


def _sum_window(by_day: Dict[int, int], day: int, window: int) -> int:
    return sum(by_day.get(d, 0) for d in range(day - window + 1, day + 1) if d >= 0)


def dynamic_daily_work_items(a: Analyst, cfg: AnalystPolicyConfig) -> int:
    if a.load == AnalystLoad.OFF:
        return 0
    base = DAILY_WORK_ITEM_BUDGET.get(a.load, 0)
    if a.consecutive_ignores >= 3:
        base = min(base, 1)
    if a.consecutive_ignores >= 5:
        base = 0
    if a.open_reviews >= 6:
        base = min(base, 1)
    return max(0, base)


def remaining_touch_budget(a: Analyst, day: int, cfg: AnalystPolicyConfig) -> int:
    day_used = a.touches_by_day.get(day, 0)
    week_used = _sum_window(a.touches_by_day, day, 7)
    return max(
        0,
        min(cfg.max_touches_per_day - day_used, cfg.max_touches_per_week - week_used),
    )


def remaining_work_budget(a: Analyst, day: int, cfg: AnalystPolicyConfig) -> int:
    return max(0, dynamic_daily_work_items(a, cfg) - a.work_items_by_day.get(day, 0))


@dataclass
class AnalystGateResult:
    allow: bool
    reason: str


def gate_analyst_deal(
    a: Analyst,
    deal: Deal,
    day: int,
    touched_today: Set[str],
    recent_sector: Sequence[Tuple[str, str, int]],
    cfg: AnalystPolicyConfig,
    enabled: bool = True,
) -> AnalystGateResult:
    if not enabled:
        return AnalystGateResult(True, "rules_off")
    if cfg.hard_stop_on_mute and deal.id in a.muted_deals:
        return AnalystGateResult(False, "muted")
    if a.load == AnalystLoad.OFF:
        return AnalystGateResult(False, "off")
    if remaining_work_budget(a, day, cfg) <= 0:
        return AnalystGateResult(False, "work_item_cap")
    already = a.id in touched_today
    if not already and remaining_touch_budget(a, day, cfg) <= 0:
        return AnalystGateResult(False, "frequency_cap")
    last = a.last_deal_touch_day.get(deal.id)
    if last is not None and (day - last) < cfg.deal_cooldown_days:
        return AnalystGateResult(False, "cooldown")
    sector_hits = sum(
        1
        for aid, sector, d in recent_sector
        if aid == a.id and sector == deal.sector and day - d < 7
    )
    if sector_hits >= cfg.sector_cap_per_week:
        return AnalystGateResult(False, "sector_cap")
    if a.open_reviews >= 8:
        return AnalystGateResult(False, "pipeline_saturated")
    return AnalystGateResult(True, "ok")


def pack_docs(analyst_id: str, doc_ids: List[str], cfg: AnalystPolicyConfig) -> List[List[str]]:
    """One digest may carry several work items — not several digests."""
    out: List[List[str]] = []
    for i in range(0, len(doc_ids), cfg.max_work_items_per_touch):
        out.append(doc_ids[i : i + cfg.max_work_items_per_touch])
    return out
