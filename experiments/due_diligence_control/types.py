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


@dataclass
class Document:
    id: str
    deal_id: str
    kind: str  # 10-K, contract, transcript, ...
    text: str
    day_arrived: int = 0
    risk_score: float = 0.0  # filled by model hook
    reviewed: bool = False


@dataclass
class ControlFlags:
    analyst_rules: bool = True
    desk_rules: bool = True
    use_model_signals: bool = True


@dataclass
class CoverageDecision:
    analyst_id: str
    deal_id: str
    document_id: str
    day: int
    score: float
    reason: str = "ok"
