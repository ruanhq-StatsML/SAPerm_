"""Evaluation metrics for the 2x2 factorial."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RunMetrics:
    impressions: int = 0
    marketing_apps: int = 0
    qualified: int = 0
    interviews: int = 0
    fills: int = 0
    unsubscribes: int = 0
    review_wait_days_sum: float = 0.0
    review_wait_obs: int = 0
    opening_days: int = 0  # sum of open seats at day starts
    net_value: float = 0.0
    blocked: Dict[str, int] = field(default_factory=dict)

    def add_blocked(self, blocked: Dict[str, int]) -> None:
        for k, v in blocked.items():
            self.blocked[k] = self.blocked.get(k, 0) + v

    @property
    def interviews_per_1k_impr(self) -> float:
        if self.impressions <= 0:
            return 0.0
        return 1000.0 * self.interviews / self.impressions

    @property
    def fills_per_100_opening_days(self) -> float:
        if self.opening_days <= 0:
            return 0.0
        return 100.0 * self.fills / self.opening_days

    @property
    def unsubscribe_rate(self) -> float:
        # Per impression proxy.
        if self.impressions <= 0:
            return 0.0
        return self.unsubscribes / self.impressions

    @property
    def mean_review_wait(self) -> float:
        if self.review_wait_obs <= 0:
            return 0.0
        return self.review_wait_days_sum / self.review_wait_obs

    def as_dict(self) -> Dict[str, float]:
        return {
            "impressions": float(self.impressions),
            "marketing_apps": float(self.marketing_apps),
            "qualified": float(self.qualified),
            "interviews": float(self.interviews),
            "fills": float(self.fills),
            "unsubscribes": float(self.unsubscribes),
            "interviews_per_1k_impr": self.interviews_per_1k_impr,
            "fills_per_100_opening_days": self.fills_per_100_opening_days,
            "unsubscribe_rate": self.unsubscribe_rate,
            "mean_review_wait": self.mean_review_wait,
            "net_value": self.net_value,
        }


def merge_metrics(rows: List[RunMetrics]) -> RunMetrics:
    out = RunMetrics()
    for m in rows:
        out.impressions += m.impressions
        out.marketing_apps += m.marketing_apps
        out.qualified += m.qualified
        out.interviews += m.interviews
        out.fills += m.fills
        out.unsubscribes += m.unsubscribes
        out.review_wait_days_sum += m.review_wait_days_sum
        out.review_wait_obs += m.review_wait_obs
        out.opening_days += m.opening_days
        out.net_value += m.net_value
        out.add_blocked(m.blocked)
    return out
