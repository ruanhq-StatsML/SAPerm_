"""Bilateral analyst × deal coverage under hard caps.

Score_ij = ΔV_ij - C_i^fatigue - C_j^congestion - C_ij^redundancy

Model risk scores only enter ΔV (and staged diagnosis). They never bypass
mute / backlog_hard / frequency caps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .analyst_rules import AnalystPolicyConfig, gate_analyst_deal
from .desk_rules import DeskBudget, DeskPolicyConfig, apply_sector_dedup, desk_day_budget
from .staged_dd import StageDecision, advance_stage, decide_next_batch, note_coverage
from .types import Analyst, ControlFlags, CoverageDecision, Deal, Document


@dataclass
class AllocationResult:
    decisions: List[CoverageDecision]
    budgets: Dict[str, DeskBudget]
    stage_decisions: Dict[str, StageDecision]
    blocked: Dict[str, int]


def _delta_v(
    a: Analyst,
    deal: Deal,
    doc: Document,
    use_model: bool,
    sparse_priority: bool = False,
    open_docs_for_deal: int = 0,
) -> float:
    fit = 0.05 + 0.2 * (1.0 - abs(a.expertise - 0.5))  # mild expertise prior
    if a.desk == deal.sector:
        fit += 0.08
    risk = doc.risk_score if use_model else 0.35
    # High risk raises value of attention — up to a point.
    value = fit + 0.4 * risk + 0.05 * deal.value
    if sparse_priority and open_docs_for_deal <= 3:
        # Thin room: lean harder into high-risk / planted critical docs.
        value += 0.35 * risk
        if doc.planted_risk and not doc.redacted:
            value += 0.2
        # Prefer anything live over empty attention when the stream is thin.
        if open_docs_for_deal <= 2:
            value += 0.15
    return float(value)


def allocate_day(
    day: int,
    analysts: Dict[str, Analyst],
    deals: Dict[str, Deal],
    documents: Dict[str, Document],
    flags: ControlFlags,
    analyst_cfg: Optional[AnalystPolicyConfig] = None,
    desk_cfg: Optional[DeskPolicyConfig] = None,
    recent_sector: Optional[Sequence[Tuple[str, str, int]]] = None,
) -> AllocationResult:
    analyst_cfg = analyst_cfg or AnalystPolicyConfig()
    desk_cfg = desk_cfg or DeskPolicyConfig()
    recent_sector = list(recent_sector or [])

    raw = {
        did: desk_day_budget(deal, desk_cfg, enabled=flags.desk_rules) for did, deal in deals.items()
    }
    budgets = apply_sector_dedup(raw, deals, desk_cfg) if flags.desk_rules else raw

    stage_decisions: Dict[str, StageDecision] = {}
    for did, deal in deals.items():
        sd = decide_next_batch(deal)
        stage_decisions[did] = sd
        if flags.desk_rules and sd.release_coverage >= 0:
            b = budgets[did]
            budgets[did] = DeskBudget(
                b.deal_id,
                min(b.coverage_budget, sd.release_coverage),
                b.review_room,
                b.backlog_days,
                b.pace_factor,
                b.reason if sd.release_coverage > 0 else f"stage:{sd.note}",
            )
        advance_stage(deal, sd)

    remaining = {did: b.coverage_budget for did, b in budgets.items()}
    touched_today: Set[str] = {aid for aid, a in analysts.items() if a.touches_by_day.get(day, 0) > 0}
    blocked: Dict[str, int] = {}

    # Soft halt (paused): still allow high-risk / planted critical docs through;
    # block low-signal filler so escalation does not freeze the whole mandate.
    open_docs = []
    for d in documents.values():
        if d.reviewed:
            continue
        deal = deals[d.deal_id]
        if deal.paused:
            risk = d.risk_score if flags.use_model_signals else 0.35
            if not (d.planted_risk or risk >= deal.risk_bar):
                continue
        open_docs.append(d)

    raw_open_by_deal: Dict[str, int] = {}
    for d in open_docs:
        raw_open_by_deal[d.deal_id] = raw_open_by_deal.get(d.deal_id, 0) + 1

    # v10 critical-first (soft): when a crowded deal (≥4 open) still has unread
    # non-redacted planted critical, penalize filler scores instead of hard-gating
    # — preserves sparse thin-stream recall while clearing dense rooms first.
    unread_critical_deals: Set[str] = set()
    if flags.critical_first:
        for d in open_docs:
            if d.planted_risk and d.available and not d.redacted and not d.found:
                unread_critical_deals.add(d.deal_id)

    open_by_deal: Dict[str, int] = {}
    for d in open_docs:
        open_by_deal[d.deal_id] = open_by_deal.get(d.deal_id, 0) + 1
    for did, deal in deals.items():
        deal.open_docs_recent = open_by_deal.get(did, 0)

    # v4 thin-stream absorb: when the live room is thin, raise remaining so
    # arriving docs are not left unread behind a pacing / sector cap.
    if flags.thin_stream_absorb:
        for did, n_open in open_by_deal.items():
            if n_open <= 0:
                continue
            if n_open <= 4 or deals[did].paused:
                remaining[did] = max(remaining.get(did, 0), n_open)

    # v10 even-effort: score bump only (no free remaining — that diluted sparse).
    even_boost: Dict[str, float] = {did: 0.0 for did in deals}
    if flags.even_effort:
        coverages = [d.total_coverage for d in deals.values()]
        mean_cov = sum(coverages) / max(len(coverages), 1)
        for did, deal in deals.items():
            if deal.total_coverage < mean_cov - 1:
                # Prefer under-covered deals that still have live critical.
                if did in unread_critical_deals or open_by_deal.get(did, 0) > 0:
                    even_boost[did] = 0.22 if did in unread_critical_deals else 0.10

    scored: List[Tuple[float, str, str, str]] = []  # score, analyst, deal, doc

    for a in analysts.values():
        for doc in open_docs:
            deal = deals[doc.deal_id]
            gate = gate_analyst_deal(
                a, deal, day, touched_today, recent_sector, analyst_cfg, flags.analyst_rules
            )
            if not gate.allow:
                blocked[gate.reason] = blocked.get(gate.reason, 0) + 1
                continue
            if remaining.get(deal.id, 0) <= 0:
                blocked["desk_budget"] = blocked.get("desk_budget", 0) + 1
                continue
            fatigue = 0.015 * sum(a.touches_by_day.get(d, 0) for d in range(max(0, day - 6), day + 1))
            fatigue += 0.02 * a.consecutive_ignores
            cong = 0.04 * budgets[deal.id].backlog_days
            last = a.last_deal_touch_day.get(deal.id)
            red = 0.04 if last is not None and day - last < 2 else 0.0
            score = (
                _delta_v(
                    a,
                    deal,
                    doc,
                    flags.use_model_signals,
                    sparse_priority=flags.sparse_priority,
                    open_docs_for_deal=open_by_deal.get(deal.id, 0),
                )
                - fatigue
                - cong
                - red
                + even_boost.get(deal.id, 0.0)
            )
            # Soft critical-first: only in truly dense rooms (≥8 open) so sparse
            # thin streams keep their v9 recall.
            if flags.critical_first and deal.id in unread_critical_deals:
                crowded = raw_open_by_deal.get(deal.id, 0) >= 8
                if crowded:
                    if doc.planted_risk and not doc.redacted:
                        score += 0.35
                    elif not doc.planted_risk:
                        score -= 0.55
                        blocked["critical_first"] = blocked.get("critical_first", 0) + 1
            scored.append((score, a.id, deal.id, doc.id))

    scored.sort(key=lambda x: x[0], reverse=True)
    decisions: List[CoverageDecision] = []

    for score, aid, did, doc_id in scored:
        a = analysts[aid]
        deal = deals[did]
        doc = documents[doc_id]
        if doc.reviewed or remaining.get(did, 0) <= 0:
            continue
        gate = gate_analyst_deal(
            a, deal, day, touched_today, recent_sector, analyst_cfg, flags.analyst_rules
        )
        if not gate.allow:
            blocked[gate.reason] = blocked.get(gate.reason, 0) + 1
            continue
        if score <= 0 and flags.analyst_rules and flags.desk_rules:
            blocked["nonpositive_score"] = blocked.get("nonpositive_score", 0) + 1
            continue

        doc.reviewed = True
        remaining[did] -= 1
        note_coverage(deal, 1)
        decisions.append(CoverageDecision(aid, did, doc_id, day, score))
        a.work_items_by_day[day] = a.work_items_by_day.get(day, 0) + 1
        a.last_deal_touch_day[did] = day
        recent_sector = list(recent_sector) + [(aid, deal.sector, day)]
        if aid not in touched_today:
            a.touches_by_day[day] = a.touches_by_day.get(day, 0) + 1
            touched_today.add(aid)

    return AllocationResult(decisions, budgets, stage_decisions, blocked)
