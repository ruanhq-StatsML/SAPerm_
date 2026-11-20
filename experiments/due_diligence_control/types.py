"""Due-diligence resource ledger.

Maps 1:1 onto the bilateral marketplace control triad:

| Recruiting | Due diligence |
| --- | --- |
| Candidate attention | Analyst attention / alert fatigue |
| Recruiter review + interview capacity | Partner / associate review + IC slots |
| Scarce marketing traffic | Scarce research / data-room coverage |

Three counters stay separate:

1. analyst *touches* (emails / Slack / digests)
2. *work items* inside a touch (docs / questions)
3. per-deal *coverage impressions* (same filing reviewed again)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set


class AnalystLoad(str, Enum):
    FRESH = "fresh"  # high bandwidth
    STEADY = "steady"
    HEAVY = "heavy"  # several live deals
    SATURATED = "saturated"
    OFF = "off"  # PTO / conflicted out


class DealStage(str, Enum):
    SCREEN = "screen"
    RED_FLAG = "red_flag"
    DEEP_DIVE = "deep_dive"
    IC = "ic"
    PASSED = "passed"
    CLOSED = "closed"


@dataclass
class Analyst:
    id: str
    load: AnalystLoad = AnalystLoad.STEADY
    desk: str = "general"
    touches_by_day: Dict[int, int] = field(default_factory=dict)
    work_items_by_day: Dict[int, int] = field(default_factory=dict)
    last_deal_touch_day: Dict[str, int] = field(default_factory=dict)
    muted_deals: Set[str] = field(default_factory=set)
    open_reviews: int = 0
    consecutive_ignores: int = 0
    expertise: float = 0.5  # [0,1] domain fit


@dataclass
class Deal:
    id: str
    sector: str
    openings: int = 1  # IC slots / mandates still needed (usually 1)
    filled: int = 0
    pending_review: int = 0  # docs / memos waiting on humans
    daily_review_capacity: int = 8
    expected_organic_docs_per_day: float = 3.0
    days_to_ic: int = 20
    stage: DealStage = DealStage.SCREEN
    stage_coverage: int = 0
    total_coverage: int = 0
    total_flags: int = 0
    total_confirmed_risks: int = 0
    total_ic_ready: int = 0
    paused: bool = False
    risk_bar: float = 0.45  # model score above this is a "flag"
    value: float = 1.0
    outcome: str = "pending"  # DealOutcome
    open_docs_recent: int = 0  # set by allocator for sparse priority
    last_escalation_day: int = -99
    escalation_pulses: int = 0  # v7: how many times we have escalated this deal
    integrity_strikes: int = 0  # v9: failed IC attempts before hard abstain


@dataclass
class Document:
    id: str
    deal_id: str
    kind: str  # 10-K, contract, transcript, ...
    text: str
    day_arrived: int = 0
    risk_score: float = 0.0  # filled by model hook
    reviewed: bool = False
    # Ground truth for substitution / missing-data studies.
    planted_risk: bool = False
    available: bool = True  # False => never arrives (missing document)
    redacted: bool = False  # True => fields stripped before the model sees them
    found: bool = False  # set when a review confirms the planted risk
    second_looks: int = 0  # v5: requeues after a failed critical confirm
    fresh_recovery: bool = False  # v11: recovered/unredacted this pulse


@dataclass
class ControlFlags:
    analyst_rules: bool = True
    desk_rules: bool = True
    use_model_signals: bool = True
    # v2: refuse IC when completeness / unread critical evidence fails gates.
    integrity_gate: bool = False
    # v3: boost high-risk / critical docs when the room is thin.
    sparse_priority: bool = False
    # v3: escalate missing/redacted evidence instead of only abstaining.
    auto_escalate: bool = False
    # v4: absorb thin streams (raise desk remaining to cover live open docs).
    thin_stream_absorb: bool = False
    # v5: requeue critical docs once after a failed confirm (second analyst look).
    second_pass_confirm: bool = False
    # v6: after escalation recovers evidence, reopen abstained deals for IC retry.
    reopen_after_recovery: bool = False
    # v7: raise alt-data / unredact odds on repeated escalation pulses.
    adaptive_escalation: bool = False
    # v8: same-day micro-pass on docs recovered / unredacted this pulse.
    same_day_recovery_pass: bool = False
    # v9: require multiple integrity failures before hard abstain (keep hunting).
    delayed_abstain: bool = False
    delayed_abstain_strikes: int = 2
    # v10: while unread non-redacted critical sits live, suppress filler for that deal.
    critical_first: bool = False
    # v10: mild boost for under-covered deals (desk even-effort).
    even_effort: bool = False
    # v11: higher confirm odds on freshly recovered/unredacted docs.
    recovery_confirm_boost: bool = False
    # v11: move unused budget from IC-ready deals to deals with unread critical.
    budget_transfer: bool = False
    # v12: shorten escalation cooldown when availability/redaction gap is active.
    fast_escalate_on_gap: bool = False
    # v14: end-of-day last-mile confirm for reviewed-but-unfound critical.
    last_mile_confirm: bool = False


class DealOutcome:
    PENDING = "pending"
    IC_READY = "ic_ready"
    ABSTAIN_INCOMPLETE = "abstain_incomplete"
    ESCALATED = "escalated"
    REOPENED = "reopened"  # v6: abstain lifted after recovery


@dataclass
class CoverageDecision:
    analyst_id: str
    deal_id: str
    document_id: str
    day: int
    score: float
    reason: str = "ok"
