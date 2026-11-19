"""Smoke checks for Phase-1 rule-based bilateral control."""

from __future__ import annotations

from .allocate import allocate_day
from .candidate_rules import CandidatePolicyConfig, dynamic_daily_job_slots, pack_jobs_into_touches
from .company_rules import CompanyPolicyConfig, company_day_budget
from .factorial import run_factorial
from .simulator import MarketplaceSim, SimConfig
from .staged_release import StageDiagnosis, decide_next_batch
from .types import Candidate, CandidateStatus, ControlFlags, Job


def test_email_slots_are_not_email_count():
    cfg = CandidatePolicyConfig(max_job_slots_per_touch=3)
    bundles = pack_jobs_into_touches("c0", ["j1", "j2", "j3", "j4"], day=0, channel="email", cfg=cfg)
    assert len(bundles) == 2
    assert bundles[0].job_ids == ["j1", "j2", "j3"]
    assert bundles[1].job_ids == ["j4"]


def test_dynamic_frequency_table():
    c = Candidate(id="c", status=CandidateStatus.ACTIVE_HIGH)
    assert dynamic_daily_job_slots(c, CandidatePolicyConfig()) == 5
    c.status = CandidateStatus.PAUSED
    assert dynamic_daily_job_slots(c, CandidatePolicyConfig()) == 0
    c.status = CandidateStatus.ACTIVE_COLD
    c.consecutive_no_response = 5
    assert dynamic_daily_job_slots(c, CandidatePolicyConfig()) == 0


def test_backlog_example_throttles():
    """Company needs 10 hires, 200 apps, 120 pending, review 30/day, organic 20/day."""
    job = Job(
        id="j",
        company_id="co",
        family="eng",
        openings=10,
        filled=0,
        pending_review=120,
        interview_slots_remaining=20,
        daily_review_capacity=30,
        expected_organic_apps_per_day=20.0,
        hiring_days_left=20,
    )
    cfg = CompanyPolicyConfig()
    b = company_day_budget(job, cfg, enabled=True)
    assert b.impression_budget == 0
    assert b.reason == "backlog_hard"

    # After backlog clears below soft threshold, pacing can release some traffic.
    job.pending_review = 40  # ~1.33 backlog days < soft 2.0
    b2 = company_day_budget(job, cfg, enabled=True)
    assert b2.marketing_app_room >= 0
    # Still paced; not the uncontrolled flood.
    b_off = company_day_budget(job, cfg, enabled=False)
    assert b_off.impression_budget >= b2.impression_budget


def test_staged_release_separates_supply_and_matching():
    job = Job(
        id="j",
        company_id="co",
        family="eng",
        openings=3,
        stage=0,
        stage_impressions=200,
        total_impressions=200,
        total_apps_from_marketing=30,
        total_qualified=2,
    )
    d = decide_next_batch(job)
    assert d.diagnosis == StageDiagnosis.MATCHING_FAILURE
    assert d.release_impressions == 0

    job2 = Job(
        id="j2",
        company_id="co",
        family="eng",
        openings=3,
        stage=0,
        stage_impressions=200,
        total_impressions=200,
        total_apps_from_marketing=2,
        total_qualified=0,
    )
    d2 = decide_next_batch(job2)
    assert d2.diagnosis == StageDiagnosis.SUPPLY_SHORTAGE


def test_allocate_respects_unsubscribe():
    c = Candidate(id="c0", status=CandidateStatus.ACTIVE_HIGH, unsubscribed=True)
    job = Job(
        id="j0",
        company_id="co",
        family="eng",
        openings=5,
        pending_review=0,
        interview_slots_remaining=10,
        daily_review_capacity=50,
        expected_organic_apps_per_day=5.0,
    )
    out = allocate_day(
        0,
        {"c0": c},
        {"j0": job},
        ControlFlags(True, True),
    )
    assert out.decisions == []
    assert out.blocked.get("negative_feedback", 0) >= 1


def test_factorial_smoke_runs():
    cfg = SimConfig(n_days=5, n_candidates=40, n_companies=3, jobs_per_company=1, n_clusters=2)
    result = run_factorial(seeds=[1, 2], base=cfg, results_dir=None)
    assert set(result.by_arm) == {"control", "t1_candidate", "t2_company", "t3_both"}
    # Controlled arms should not spend more impressions than the uncontrolled flood.
    assert result.by_arm["t3_both"].impressions <= result.by_arm["control"].impressions


def test_sim_both_beats_control_on_waste_proxy():
    cfg = SimConfig(n_days=12, n_candidates=60, n_companies=4, jobs_per_company=1, seed=7)
    control = MarketplaceSim(cfg=cfg, flags=ControlFlags(False, False)).run(seed=7)
    both = MarketplaceSim(cfg=cfg, flags=ControlFlags(True, True)).run(seed=7)
    # Primary Phase-1 claim: less fatigue and/or less backlog pressure per unit value.
    assert both.unsubscribe_rate <= control.unsubscribe_rate + 1e-9
    assert both.mean_review_wait <= control.mean_review_wait + 1e-6
