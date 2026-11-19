"""Shared state for the Phase-1 marketplace control loop.

Three resource budgets are tracked separately (do not collapse them):

1. candidate marketing *touches* (emails / pushes)
2. candidate job *slots* inside those touches
3. per-job *impressions* of the same posting
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


class CandidateStatus(str, Enum):
    ACTIVE_HIGH = "active_high"  # actively looking, recent high response
    ACTIVE_COLD = "active_cold"  # actively looking, no recent response
    MULTI_INTERVIEW = "multi_interview"
    LOW_RESPONSE = "low_response"
    PAUSED = "paused"  # paused search or unsubscribed


@dataclass
class Candidate:
    id: str
    status: CandidateStatus
    cluster: str = "default"
    # Rolling windows keyed by day index.
    marketing_touches_by_day: Dict[int, int] = field(default_factory=dict)
    job_slots_by_day: Dict[int, int] = field(default_factory=dict)
    last_job_touch_day: Dict[str, int] = field(default_factory=dict)
    ignored_jobs: Set[str] = field(default_factory=set)
    unsubscribed: bool = False
    open_apps: int = 0
    open_interviews: int = 0
    consecutive_no_response: int = 0
    # Soft match quality proxy in [0, 1] vs job skill keys.
    skill: float = 0.5


@dataclass
class Job:
    id: str
    company_id: str
    family: str  # similar jobs share a family for company-level caps
    openings: int
    filled: int = 0
    pending_review: int = 0
    interview_slots_remaining: int = 0
    daily_review_capacity: int = 30
    expected_organic_apps_per_day: float = 20.0
    hiring_days_left: int = 30
    stage: int = 0  # staged-release phase index
    stage_impressions: int = 0
    total_impressions: int = 0
    total_apps_from_marketing: int = 0
    total_qualified: int = 0
    total_interviews: int = 0
    paused: bool = False
    # Match quality target in [0, 1].
    bar: float = 0.55


@dataclass
class Company:
    id: str
    jobs: List[str] = field(default_factory=list)


@dataclass
class ExposureDecision:
    candidate_id: str
    job_id: str
    channel: str
    day: int
    score: float
    reason: str = "ok"


@dataclass
class DayBundle:
    """One marketing touch may carry several job slots."""

    candidate_id: str
    channel: str
    day: int
    job_ids: List[str]


@dataclass
class ControlFlags:
    candidate_rules: bool = True
    company_rules: bool = True


Channel = str  # "email" | "push" | "in_app"
TouchKey = Tuple[str, int, Channel]  # (candidate_id, day, channel)
