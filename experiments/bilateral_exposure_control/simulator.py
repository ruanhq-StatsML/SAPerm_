"""Synthetic bilateral marketplace with backlog + organic inflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .allocate import allocate_day
from .candidate_rules import CandidatePolicyConfig
from .company_rules import CompanyPolicyConfig
from .metrics import RunMetrics
from .types import Candidate, CandidateStatus, Company, ControlFlags, Job


@dataclass
class SimConfig:
    n_days: int = 30
    n_candidates: int = 120
    n_companies: int = 6
    jobs_per_company: int = 2
    seed: int = 2026
    # Cluster ids for cluster-randomized experiments.
    n_clusters: int = 4
    apply_base_rate: float = 0.08
    qualify_base_rate: float = 0.35
    interview_from_qualify: float = 0.5
    fill_from_interview: float = 0.35
    fatigue_unsub_base: float = 0.002
    value_per_fill: float = 1.0
    cost_per_impression: float = 0.002
    cost_per_unsub: float = 0.5
    cost_per_review_day: float = 0.01


def _init_world(cfg: SimConfig, rng: np.random.Generator):
    candidates: Dict[str, Candidate] = {}
    statuses = list(CandidateStatus)
    for i in range(cfg.n_candidates):
        st = statuses[int(rng.integers(0, len(statuses) - 1))]  # rarely start paused
        if rng.random() < 0.05:
            st = CandidateStatus.PAUSED
        candidates[f"c{i}"] = Candidate(
            id=f"c{i}",
            status=st,
            cluster=f"m{int(rng.integers(0, cfg.n_clusters))}",
            skill=float(np.clip(rng.normal(0.55, 0.15), 0.05, 0.95)),
            open_apps=int(rng.integers(0, 3)),
            open_interviews=int(rng.integers(0, 2)),
        )

    jobs: Dict[str, Job] = {}
    companies: Dict[str, Company] = {}
    for ci in range(cfg.n_companies):
        cid = f"co{ci}"
        companies[cid] = Company(id=cid, jobs=[])
        for ji in range(cfg.jobs_per_company):
            jid = f"{cid}_j{ji}"
            openings = int(rng.integers(2, 8))
            # The motivating backlog example: some jobs start overloaded.
            pending = int(rng.choice([5, 20, 60, 120, 200], p=[0.25, 0.25, 0.2, 0.2, 0.1]))
            review_cap = int(rng.choice([20, 30, 40]))
            organic = float(rng.choice([8.0, 15.0, 20.0, 25.0]))
            jobs[jid] = Job(
                id=jid,
                company_id=cid,
                family=f"fam{ji % 2}",
                openings=openings,
                pending_review=pending,
                interview_slots_remaining=int(openings * 3),
                daily_review_capacity=review_cap,
                expected_organic_apps_per_day=organic,
                hiring_days_left=int(rng.integers(14, 45)),
                bar=float(np.clip(rng.normal(0.55, 0.08), 0.35, 0.75)),
            )
            companies[cid].jobs.append(jid)
    return candidates, jobs, companies


@dataclass
class MarketplaceSim:
    cfg: SimConfig = field(default_factory=SimConfig)
    flags: ControlFlags = field(default_factory=ControlFlags)
    candidate_cfg: CandidatePolicyConfig = field(default_factory=CandidatePolicyConfig)
    company_cfg: CompanyPolicyConfig = field(default_factory=CompanyPolicyConfig)

    def run(self, seed: Optional[int] = None) -> RunMetrics:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed if seed is None else seed)
        candidates, jobs, _companies = _init_world(cfg, rng)
        metrics = RunMetrics()
        recent_family: List[Tuple[str, str, int]] = []

        for day in range(cfg.n_days):
            metrics.opening_days += sum(max(j.openings - j.filled, 0) for j in jobs.values())

            # Reviewers clear backlog.
            for job in jobs.values():
                cleared = min(job.pending_review, job.daily_review_capacity)
                job.pending_review -= cleared
                # Rough wait proxy: backlog_days before clear.
                if cleared:
                    metrics.review_wait_days_sum += (job.pending_review + cleared) / max(
                        job.daily_review_capacity, 1
                    )
                    metrics.review_wait_obs += 1
                # Organic applications arrive regardless of marketing.
                organic = int(rng.poisson(job.expected_organic_apps_per_day))
                job.pending_review += organic
                job.hiring_days_left = max(job.hiring_days_left - 1, 0)

            alloc = allocate_day(
                day,
                candidates,
                jobs,
                self.flags,
                self.candidate_cfg,
                self.company_cfg,
                recent_company_family=recent_family,
                expected_apply_rate=cfg.apply_base_rate,
            )
            metrics.add_blocked(alloc.blocked)
            metrics.impressions += len(alloc.decisions)

            # Realize outcomes for marketed pairs.
            for dec in alloc.decisions:
                c = candidates[dec.candidate_id]
                job = jobs[dec.job_id]
                metrics.net_value -= cfg.cost_per_impression

                # Apply probability rises with match, falls with fatigue.
                p_apply = cfg.apply_base_rate * (0.6 + 0.8 * max(0.0, c.skill - job.bar + 0.5))
                p_apply *= max(0.2, 1.0 - 0.08 * c.consecutive_no_response)
                if rng.random() < p_apply:
                    job.total_apps_from_marketing += 1
                    job.pending_review += 1
                    c.open_apps += 1
                    c.consecutive_no_response = 0
                    metrics.marketing_apps += 1
                    if rng.random() < cfg.qualify_base_rate * (0.5 + c.skill):
                        job.total_qualified += 1
                        metrics.qualified += 1
                        if (
                            job.interview_slots_remaining > 0
                            and rng.random() < cfg.interview_from_qualify
                        ):
                            job.total_interviews += 1
                            job.interview_slots_remaining -= 1
                            c.open_interviews += 1
                            metrics.interviews += 1
                            if rng.random() < cfg.fill_from_interview and job.filled < job.openings:
                                job.filled += 1
                                metrics.fills += 1
                                metrics.net_value += cfg.value_per_fill
                else:
                    c.consecutive_no_response += 1
                    if c.consecutive_no_response >= 3 and c.status == CandidateStatus.ACTIVE_HIGH:
                        c.status = CandidateStatus.LOW_RESPONSE
                    # Negative feedback / ignore with small probability.
                    if rng.random() < 0.03:
                        c.ignored_jobs.add(job.id)

                # Unsubscribe risk grows with touches.
                week_touches = sum(
                    c.marketing_touches_by_day.get(d, 0) for d in range(max(0, day - 6), day + 1)
                )
                p_unsub = cfg.fatigue_unsub_base * (1 + week_touches)
                if (not c.unsubscribed) and rng.random() < p_unsub:
                    c.unsubscribed = True
                    c.status = CandidateStatus.PAUSED
                    metrics.unsubscribes += 1
                    metrics.net_value -= cfg.cost_per_unsub

                fam = f"{job.company_id}:{job.family}"
                recent_family.append((c.id, fam, day))

            metrics.net_value -= cfg.cost_per_review_day * sum(
                j.pending_review / max(j.daily_review_capacity, 1) for j in jobs.values()
            )

            # Refresh candidate status from pipeline.
            for c in candidates.values():
                if c.unsubscribed:
                    c.status = CandidateStatus.PAUSED
                elif c.open_interviews >= 2:
                    c.status = CandidateStatus.MULTI_INTERVIEW

            # Bound recent_family memory.
            recent_family = [r for r in recent_family if day - r[2] < 7]

        return metrics
