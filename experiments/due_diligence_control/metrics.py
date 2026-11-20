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
    # Substitution study (planted risks / missingness).
    planted_total: int = 0
    planted_available: int = 0
    planted_found: int = 0
    false_ic: int = 0  # IC-ready while an available planted risk is still unread/unfound
    supply_shortage_events: int = 0
    matching_failure_events: int = 0
    ic_blocked_integrity: int = 0
    integrity_refusals: int = 0
    # v3 escalation / abstain
    abstain_incomplete: int = 0
    escalations: int = 0
    docs_recovered: int = 0
    fields_unredacted: int = 0
    halt_coverage_events: int = 0
    # v5 second-pass confirms
    second_pass_confirms: int = 0
    second_pass_requeues: int = 0
    # v6 reopen after recovery
    abstain_reopened: int = 0
    ic_after_reopen: int = 0

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

    @property
    def recall_available(self) -> float:
        if self.planted_available <= 0:
            return 0.0
        return self.planted_found / self.planted_available

    @property
    def recall_all(self) -> float:
        """Includes never-arrived planted docs — hard ceiling under missingness."""
        if self.planted_total <= 0:
            return 0.0
        return self.planted_found / self.planted_total

    @property
    def false_ic_rate(self) -> float:
        if self.ic_ready <= 0:
            return 0.0
        return self.false_ic / self.ic_ready

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
            "planted_total": float(self.planted_total),
            "planted_available": float(self.planted_available),
            "planted_found": float(self.planted_found),
            "recall_available": self.recall_available,
            "recall_all": self.recall_all,
            "false_ic": float(self.false_ic),
            "false_ic_rate": self.false_ic_rate,
            "supply_shortage_events": float(self.supply_shortage_events),
            "matching_failure_events": float(self.matching_failure_events),
            "ic_blocked_integrity": float(self.ic_blocked_integrity),
            "integrity_refusals": float(self.integrity_refusals),
            "abstain_incomplete": float(self.abstain_incomplete),
            "escalations": float(self.escalations),
            "docs_recovered": float(self.docs_recovered),
            "fields_unredacted": float(self.fields_unredacted),
            "halt_coverage_events": float(self.halt_coverage_events),
            "second_pass_confirms": float(self.second_pass_confirms),
            "second_pass_requeues": float(self.second_pass_requeues),
            "abstain_reopened": float(self.abstain_reopened),
            "ic_after_reopen": float(self.ic_after_reopen),
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
        out.planted_total += m.planted_total
        out.planted_available += m.planted_available
        out.planted_found += m.planted_found
        out.false_ic += m.false_ic
        out.supply_shortage_events += m.supply_shortage_events
        out.matching_failure_events += m.matching_failure_events
        out.ic_blocked_integrity += m.ic_blocked_integrity
        out.integrity_refusals += m.integrity_refusals
        out.abstain_incomplete += m.abstain_incomplete
        out.escalations += m.escalations
        out.docs_recovered += m.docs_recovered
        out.fields_unredacted += m.fields_unredacted
        out.halt_coverage_events += m.halt_coverage_events
        out.second_pass_confirms += m.second_pass_confirms
        out.second_pass_requeues += m.second_pass_requeues
        out.abstain_reopened += m.abstain_reopened
        out.ic_after_reopen += m.ic_after_reopen
    return out
