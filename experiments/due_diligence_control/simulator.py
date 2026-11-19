"""Synthetic DD desk: organic filings + controlled analyst coverage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .allocate import allocate_day
from .analyst_rules import AnalystPolicyConfig
from .desk_rules import DeskPolicyConfig
from .metrics import RunMetrics
from .model_hooks import HeuristicRiskProvider, RiskSignalProvider, score_documents
from .types import Analyst, AnalystLoad, ControlFlags, Deal, DealStage, Document


TEMPLATES_CLEAN = [
    "Revenue grew 8% year over year with stable margins and normal working capital.",
    "Management reiterated guidance. Customer retention remained in line with peers.",
    "The company refinanced on schedule with standard covenants and no waiver requests.",
]
TEMPLATES_RISKY = [
    "Auditor highlighted a material weakness and possible going concern uncertainty.",
    "Pending litigation and a related party investigation may trigger covenant breach.",
    "Liquidity risk rose after the rating downgrade and inventory impairment charge.",
]


@dataclass
class SimConfig:
    n_days: int = 25
    n_analysts: int = 24
    n_deals: int = 8
    seed: int = 2026
    n_desks: int = 3
    cost_per_coverage: float = 0.01
    cost_per_mute: float = 0.4
    value_per_confirmed: float = 0.35
    value_per_ic: float = 1.2
    cost_per_review_day: float = 0.02


def _init(cfg: SimConfig, rng: np.random.Generator):
    sectors = [f"sec{i}" for i in range(cfg.n_desks)]
    loads = [AnalystLoad.FRESH, AnalystLoad.STEADY, AnalystLoad.HEAVY, AnalystLoad.SATURATED]
    analysts: Dict[str, Analyst] = {}
    for i in range(cfg.n_analysts):
        analysts[f"a{i}"] = Analyst(
            id=f"a{i}",
            load=loads[int(rng.integers(0, len(loads)))],
            desk=sectors[int(rng.integers(0, len(sectors)))],
            expertise=float(np.clip(rng.normal(0.55, 0.15), 0.1, 0.9)),
            open_reviews=int(rng.integers(0, 3)),
        )
    deals: Dict[str, Deal] = {}
    for j in range(cfg.n_deals):
        pending = int(rng.choice([2, 6, 12, 24, 40], p=[0.25, 0.25, 0.2, 0.2, 0.1]))
        deals[f"d{j}"] = Deal(
            id=f"d{j}",
            sector=sectors[j % len(sectors)],
            pending_review=pending,
            daily_review_capacity=int(rng.choice([6, 8, 10])),
            expected_organic_docs_per_day=float(rng.choice([2.0, 3.0, 5.0])),
            days_to_ic=int(rng.integers(10, 30)),
            risk_bar=float(np.clip(rng.normal(0.45, 0.05), 0.3, 0.6)),
            value=float(np.clip(rng.normal(1.0, 0.2), 0.5, 1.5)),
        )
    return analysts, deals


def _spawn_docs(
    day: int,
    deals: Dict[str, Deal],
    documents: Dict[str, Document],
    rng: np.random.Generator,
    provider: RiskSignalProvider,
) -> None:
    for deal in deals.values():
        n = int(rng.poisson(deal.expected_organic_docs_per_day))
        for k in range(n):
            risky = rng.random() < 0.28
            text = rng.choice(TEMPLATES_RISKY if risky else TEMPLATES_CLEAN)
            # Light paraphrase noise.
            text = f"{text} Ref {deal.id}-{day}-{k}."
            did = f"{deal.id}_doc{day}_{k}_{len(documents)}"
            doc = Document(id=did, deal_id=deal.id, kind="filing", text=text, day_arrived=day)
            score_documents([doc], provider)
            documents[did] = doc
            deal.pending_review += 1


@dataclass
class DueDiligenceSim:
    cfg: SimConfig = field(default_factory=SimConfig)
    flags: ControlFlags = field(default_factory=ControlFlags)
    analyst_cfg: AnalystPolicyConfig = field(default_factory=AnalystPolicyConfig)
    desk_cfg: DeskPolicyConfig = field(default_factory=DeskPolicyConfig)
    provider: RiskSignalProvider = field(default_factory=HeuristicRiskProvider)

    def run(self, seed: Optional[int] = None) -> RunMetrics:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed if seed is None else seed)
        analysts, deals = _init(cfg, rng)
        documents: Dict[str, Document] = {}
        metrics = RunMetrics()
        recent_sector: List[Tuple[str, str, int]] = []

        for day in range(cfg.n_days):
            metrics.mandate_days += sum(
                1 for d in deals.values() if d.stage not in (DealStage.PASSED, DealStage.CLOSED)
            )

            for deal in deals.values():
                cleared = min(deal.pending_review, deal.daily_review_capacity)
                if cleared:
                    metrics.review_wait_sum += (deal.pending_review) / max(deal.daily_review_capacity, 1)
                    metrics.review_wait_obs += 1
                deal.pending_review -= cleared
                deal.days_to_ic = max(deal.days_to_ic - 1, 0)

            _spawn_docs(day, deals, documents, rng, self.provider)

            alloc = allocate_day(
                day,
                analysts,
                deals,
                documents,
                self.flags,
                self.analyst_cfg,
                self.desk_cfg,
                recent_sector=recent_sector,
            )
            metrics.add_blocked(alloc.blocked)
            metrics.coverage += len(alloc.decisions)

            for dec in alloc.decisions:
                a = analysts[dec.analyst_id]
                deal = deals[dec.deal_id]
                doc = documents[dec.document_id]
                metrics.net_value -= cfg.cost_per_coverage
                deal.pending_review = max(deal.pending_review - 1, 0)

                is_flag = doc.risk_score >= deal.risk_bar
                if is_flag:
                    deal.total_flags += 1
                    metrics.flags += 1
                    # Confirmation = senior agrees (expertise helps).
                    if rng.random() < 0.25 + 0.5 * a.expertise * doc.risk_score:
                        deal.total_confirmed_risks += 1
                        metrics.confirmed += 1
                        metrics.net_value += cfg.value_per_confirmed
                        a.consecutive_ignores = 0
                        a.open_reviews += 1
                    else:
                        a.consecutive_ignores += 1
                else:
                    a.consecutive_ignores += 1

                if deal.total_confirmed_risks >= 3 and deal.total_ic_ready == 0 and rng.random() < 0.35:
                    deal.total_ic_ready = 1
                    metrics.ic_ready += 1
                    metrics.net_value += cfg.value_per_ic * deal.value

                # Mute / fatigue.
                week_touches = sum(a.touches_by_day.get(d, 0) for d in range(max(0, day - 6), day + 1))
                if rng.random() < 0.0015 * (1 + week_touches):
                    a.muted_deals.add(deal.id)
                    metrics.mutes += 1
                    metrics.net_value -= cfg.cost_per_mute

                recent_sector.append((a.id, deal.sector, day))

            metrics.net_value -= cfg.cost_per_review_day * sum(
                d.pending_review / max(d.daily_review_capacity, 1) for d in deals.values()
            )
            recent_sector = [r for r in recent_sector if day - r[2] < 7]

            for a in analysts.values():
                if a.open_reviews >= 5:
                    a.load = AnalystLoad.SATURATED
                elif a.open_reviews >= 3:
                    a.load = AnalystLoad.HEAVY

        return metrics
