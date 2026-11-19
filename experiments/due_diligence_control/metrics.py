"""DD control evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RunMetrics:
    coverage: int = 0
    flags: int = 0
    confirmed: int = 0
    ic_ready: int = 0
    passes: int = 0
    mutes: int = 0
    review_wait_sum: float = 0.0
    review_wait_obs: int = 0
    mandate_days: int = 0
    net_value: float = 0.0
    blocked: Dict[str, int] = field(default_factory=dict)

    def add_blocked(self, blocked: Dict[str, int]) -> None:
        for k, v in blocked.items():
            self.blocked[k] = self.blocked.get(k, 0) + v

    @property
    def confirmed_per_1k_coverage(self) -> float:
        if self.coverage <= 0:
            return 0.0
        return 1000.0 * self.confirmed / self.coverage

    @property
    def ic_per_100_mandate_days(self) -> float:
        if self.mandate_days <= 0:
            return 0.0
        return 100.0 * self.ic_ready / self.mandate_days

    @property
    def mute_rate(self) -> float:
        if self.coverage <= 0:
            return 0.0
        return self.mutes / self.coverage

    @property
    def mean_review_wait(self) -> float:
        if self.review_wait_obs <= 0:
            return 0.0
        return self.review_wait_sum / self.review_wait_obs

    def as_dict(self) -> Dict[str, float]:
        return {
            "coverage": float(self.coverage),
            "flags": float(self.flags),
            "confirmed": float(self.confirmed),
            "ic_ready": float(self.ic_ready),
            "confirmed_per_1k_coverage": self.confirmed_per_1k_coverage,
            "ic_per_100_mandate_days": self.ic_per_100_mandate_days,
            "mute_rate": self.mute_rate,
            "mean_review_wait": self.mean_review_wait,
            "net_value": self.net_value,
        }


def merge_metrics(rows: List[RunMetrics]) -> RunMetrics:
    out = RunMetrics()
    for m in rows:
        out.coverage += m.coverage
        out.flags += m.flags
        out.confirmed += m.confirmed
        out.ic_ready += m.ic_ready
        out.passes += m.passes
        out.mutes += m.mutes
        out.review_wait_sum += m.review_wait_sum
        out.review_wait_obs += m.review_wait_obs
        out.mandate_days += m.mandate_days
        out.net_value += m.net_value
        out.add_blocked(m.blocked)
    return out
