"""Smoke tests for due-diligence control + model hooks."""

from __future__ import annotations

from .analyst_rules import AnalystPolicyConfig, pack_docs
from .desk_rules import DeskPolicyConfig, desk_day_budget
from .factorial import run_factorial
from .model_hooks import HeuristicRiskProvider, PromptedRiskProvider, score_documents
from .simulator import DueDiligenceSim, SimConfig
from .staged_dd import StageDiagnosis, decide_next_batch
from .types import Analyst, AnalystLoad, ControlFlags, Deal, DealStage, Document


def test_digest_slots_are_not_touch_count():
    bundles = pack_docs("a0", ["d1", "d2", "d3", "d4"], AnalystPolicyConfig(max_work_items_per_touch=3))
    assert len(bundles) == 2
    assert bundles[0] == ["d1", "d2", "d3"]


def test_backlog_throttles_desk():
    deal = Deal(
        id="d",
        sector="sec0",
        pending_review=40,
        daily_review_capacity=8,
        expected_organic_docs_per_day=5.0,
        days_to_ic=15,
    )
    b = desk_day_budget(deal, DeskPolicyConfig(), enabled=True)
    assert b.reason == "backlog_hard"
    assert b.coverage_budget == 0


def test_staged_matching_failure():
    deal = Deal(
        id="d",
        sector="sec0",
        stage=DealStage.SCREEN,
        stage_coverage=40,
        total_coverage=40,
        total_flags=10,
        total_confirmed_risks=1,
    )
    d = decide_next_batch(deal)
    assert d.diagnosis == StageDiagnosis.MATCHING_FAILURE
    assert d.release_coverage == 0


def test_heuristic_flags_risk_lexicon():
    p = HeuristicRiskProvider()
    clean = Document(id="1", deal_id="d", kind="x", text="Revenue grew with stable margins.")
    risky = Document(
        id="2",
        deal_id="d",
        kind="x",
        text="Material weakness and going concern plus covenant breach litigation.",
    )
    score_documents([clean, risky], p)
    assert risky.risk_score > clean.risk_score
    assert risky.risk_score >= 0.35


def test_prompted_financegpt_hook():
    provider = PromptedRiskProvider(fn=lambda t: 0.9 if "fraud" in t.lower() else 0.1, name="fake_financegpt")
    doc = Document(id="1", deal_id="d", kind="x", text="Possible fraud investigation.")
    score_documents([doc], provider)
    assert doc.risk_score == 0.9


def test_mute_blocks_allocation():
    from .allocate import allocate_day

    a = Analyst(id="a0", load=AnalystLoad.FRESH, muted_deals={"d0"})
    deal = Deal(id="d0", sector="sec0", pending_review=0, daily_review_capacity=20, expected_organic_docs_per_day=1)
    doc = Document(id="doc", deal_id="d0", kind="x", text="litigation risk", risk_score=0.8)
    out = allocate_day(0, {"a0": a}, {"d0": deal}, {"doc": doc}, ControlFlags(True, True, True))
    assert out.decisions == []
    assert out.blocked.get("muted", 0) >= 1


def test_factorial_smoke():
    cfg = SimConfig(n_days=6, n_analysts=12, n_deals=4, n_desks=2)
    result = run_factorial(seeds=[1, 2], base=cfg, results_dir=None)
    assert set(result.by_arm) == {"control", "t1_analyst", "t2_desk", "t3_both"}
    assert result.by_arm["t3_both"].coverage <= result.by_arm["control"].coverage


def test_missing_docs_cap_recall_all():
    from .sparsity import REGIMES
    from .substitution_study import StudyConfig, run_one

    cfg = StudyConfig(n_days=8, n_analysts=10, n_deals=4, planted_per_deal=3)
    # v2 abstain-only: availability stays below total when escalations are off.
    m = run_one("controlled_v2", REGIMES["missing_docs"], cfg, seed=3)
    assert m.planted_available < m.planted_total
    assert m.recall_all <= m.recall_available + 1e-9
    assert m.planted_found <= m.planted_available


def test_redaction_blinds_model_score():
    from .model_hooks import HeuristicRiskProvider, score_documents
    from .types import Document

    p = HeuristicRiskProvider()
    full = Document(id="1", deal_id="d", kind="x", text="material weakness going concern covenant breach")
    red = Document(id="2", deal_id="d", kind="x", text="[REDACTED — source field unavailable]", redacted=True)
    score_documents([full, red], p)
    assert full.risk_score > red.risk_score


def test_substitution_board_smoke():
    from .substitution_study import StudyConfig, run_substitution_study

    board = run_substitution_study(
        seeds=[5, 6],
        regimes=["dense", "missing_docs"],
        methods=["flood", "controlled"],
        cfg=StudyConfig(n_days=8, n_analysts=8, n_deals=3, planted_per_deal=2),
        results_dir=None,
    )
    assert len(board.rows) == 4
    dense_c = next(r for r in board.rows if r.regime == "dense" and r.method == "controlled")
    miss_c = next(r for r in board.rows if r.regime == "missing_docs" and r.method == "controlled")
    assert dense_c.metrics.coverage > 0
    assert miss_c.metrics.false_ic_rate <= 0.05 + 1e-9


def test_integrity_gate_blocks_incomplete_ic():
    from .integrity import IntegrityConfig, snapshot_deal
    from .types import Deal, Document

    deal = Deal(id="d0", sector="s", total_confirmed_risks=3)
    corpus = {
        "a": Document(id="a", deal_id="d0", kind="c", text="x", planted_risk=True, available=False),
        "b": Document(id="b", deal_id="d0", kind="c", text="x", planted_risk=True, available=True, found=False),
        "c": Document(id="c", deal_id="d0", kind="c", text="x", planted_risk=True, available=True, found=True),
    }
    snap = snapshot_deal(deal, corpus, IntegrityConfig(min_availability=0.9, max_unfound_available_share=0.1))
    assert snap.allow_ic is False
    assert "low_availability" in snap.reasons or "unfound_available_risks" in snap.reasons


def test_v2_cuts_false_ic_vs_v1_under_missing_docs():
    from .sparsity import REGIMES
    from .substitution_study import StudyConfig, run_one

    cfg = StudyConfig(n_days=10, n_analysts=10, n_deals=4, planted_per_deal=3)
    v1 = run_one("controlled_v1", REGIMES["missing_docs"], cfg, seed=9)
    v2 = run_one("controlled_v2", REGIMES["missing_docs"], cfg, seed=9)
    assert v2.false_ic_rate <= v1.false_ic_rate + 1e-9
    assert v2.ic_blocked_integrity >= 0


def test_v3_escalation_can_recover_missing_docs():
    from .sparsity import REGIMES
    from .substitution_study import StudyConfig, run_one

    cfg = StudyConfig(n_days=14, n_analysts=10, n_deals=4, planted_per_deal=3)
    v3 = run_one("controlled", REGIMES["missing_docs"], cfg, seed=21)
    v2 = run_one("controlled_v2", REGIMES["missing_docs"], cfg, seed=21)
    assert v3.docs_recovered >= 1
    assert v3.false_ic_rate <= v2.false_ic_rate + 1e-9
    assert v3.escalations >= 1


def test_escalation_planner_requests_missing():
    from .escalation import plan_escalations
    from .integrity import snapshot_deal
    from .types import Deal, Document

    deal = Deal(id="d0", sector="s", total_confirmed_risks=2)
    corpus = {
        "m": Document(id="m", deal_id="d0", kind="c", text="x", planted_risk=True, available=False),
        "a": Document(id="a", deal_id="d0", kind="c", text="x", planted_risk=True, available=True, found=True),
    }
    snap = snapshot_deal(deal, corpus)
    actions = plan_escalations(deal, snap, corpus, "supply_shortage")
    kinds = {a.kind for a in actions}
    assert "request_missing_docs" in kinds
    assert "halt_broad_coverage" in kinds


def test_paused_keeps_critical_lane_budget():
    from .desk_rules import DeskPolicyConfig, desk_day_budget

    deal = Deal(
        id="d",
        sector="sec0",
        pending_review=2,
        daily_review_capacity=8,
        expected_organic_docs_per_day=1.0,
        days_to_ic=15,
        paused=True,
    )
    b = desk_day_budget(deal, DeskPolicyConfig(), enabled=True)
    assert b.reason == "critical_lane"
    assert b.coverage_budget >= 2
    assert b.review_room >= 1


def test_unfound_halt_requires_live_pressure():
    from .escalation import plan_escalations
    from .integrity import snapshot_deal
    from .types import Deal, Document

    deal = Deal(id="d0", sector="s", total_confirmed_risks=2, total_coverage=12)
    corpus = {
        "a": Document(
            id="a", deal_id="d0", kind="c", text="x",
            planted_risk=True, available=True, found=False,
        ),
        "b": Document(
            id="b", deal_id="d0", kind="c", text="x",
            planted_risk=True, available=True, found=True,
        ),
    }
    snap = snapshot_deal(deal, corpus)
    # Latent unfound only — no live docs → no soft-halt.
    actions = plan_escalations(deal, snap, corpus, "ok", live={})
    assert not any(a.kind == "halt_broad_coverage" for a in actions)
    # Live unread critical → soft-halt.
    live = {"a": corpus["a"]}
    actions2 = plan_escalations(deal, snap, corpus, "ok", live=live)
    assert any(a.kind == "halt_broad_coverage" for a in actions2)


def test_v4_sparse_recall_not_frozen_vs_v2():
    from .sparsity import REGIMES
    from .substitution_study import StudyConfig, run_one

    cfg = StudyConfig(n_days=14, n_analysts=10, n_deals=4, planted_per_deal=3)
    controlled = run_one("controlled", REGIMES["sparse"], cfg, seed=17)
    v2 = run_one("controlled_v2", REGIMES["sparse"], cfg, seed=17)
    # Critical lane + thin-stream absorb should not collapse far below v2.
    assert controlled.recall_available + 1e-9 >= v2.recall_available - 0.12
    assert controlled.false_ic_rate <= v2.false_ic_rate + 1e-9


def test_v5_second_pass_can_confirm_critical():
    from .sparsity import REGIMES
    from .substitution_study import StudyConfig, run_one

    cfg = StudyConfig(n_days=18, n_analysts=12, n_deals=6, planted_per_deal=3)
    # Dense filler raises miss rate on first look → dual-look fires.
    v5 = run_one("controlled", REGIMES["dense"], cfg, seed=11)
    assert v5.second_pass_requeues >= 1
    assert v5.false_ic_rate <= 1e-9


def test_v6_can_reopen_abstain_under_missing_docs():
    from .sparsity import REGIMES
    from .substitution_study import StudyConfig, run_one

    cfg = StudyConfig(n_days=18, n_analysts=12, n_deals=6, planted_per_deal=3)
    v6 = run_one("controlled", REGIMES["missing_docs"], cfg, seed=11)
    # Escalation + reopen should keep false-IC at 0; reopen or recover should fire.
    assert v6.false_ic_rate <= 1e-9
    assert v6.docs_recovered >= 1
    assert v6.abstain_reopened + v6.ic_after_reopen + v6.abstain_incomplete >= 1


def test_adaptive_escalation_bumps_rates():
    from .escalation import EscalationConfig, adaptive_config

    base = EscalationConfig()
    bumped = adaptive_config(base, pulses_so_far=2)
    assert bumped.alt_data_success_rate > base.alt_data_success_rate
    assert bumped.unredact_success_rate > base.unredact_success_rate
    assert bumped.max_doc_requests >= base.max_doc_requests
    assert bumped.alt_data_success_rate <= base.adaptive_rate_cap


def test_v8_same_day_recovery_reads_under_missing():
    from .sparsity import REGIMES
    from .substitution_study import StudyConfig, run_one

    cfg = StudyConfig(n_days=18, n_analysts=12, n_deals=6, planted_per_deal=3)
    v8 = run_one("controlled", REGIMES["sparse_missing"], cfg, seed=11)
    assert v8.false_ic_rate <= 1e-9
    assert v8.docs_recovered + v8.fields_unredacted >= 1
    assert v8.same_day_recovery_reads >= 1


def test_v9_delayed_abstain_keeps_false_ic_zero():
    from .sparsity import REGIMES
    from .substitution_study import StudyConfig, run_one

    cfg = StudyConfig(n_days=18, n_analysts=12, n_deals=6, planted_per_deal=3)
    v9 = run_one("controlled", REGIMES["dense"], cfg, seed=11)
    assert v9.false_ic_rate <= 1e-9
    assert v9.recall_available >= 0.9


def test_recovery_confirm_boost_flag_roundtrip():
    from .types import ControlFlags, Document

    assert ControlFlags(recovery_confirm_boost=True).recovery_confirm_boost is True
    d = Document(id="x", deal_id="d", kind="c", text="t", fresh_recovery=True)
    assert d.fresh_recovery is True


def test_matching_failure_extra_unredacts_only_when_configured_thin():
    from .escalation import EscalationConfig, plan_escalations
    from .integrity import snapshot_deal
    from .types import Deal, Document

    thin = Deal(
        id="d0", sector="s", total_confirmed_risks=2, total_coverage=12, open_docs_recent=1
    )
    corpus = {
        f"r{i}": Document(
            id=f"r{i}",
            deal_id="d0",
            kind="c",
            text="[REDACTED]",
            planted_risk=True,
            available=True,
            redacted=True,
        )
        for i in range(4)
    }
    snap = snapshot_deal(thin, corpus)
    base_n = len(
        [
            a
            for a in plan_escalations(thin, snap, corpus, "matching_failure", EscalationConfig())
            if a.kind == "request_unredact"
        ]
    )
    boosted_n = len(
        [
            a
            for a in plan_escalations(
                thin,
                snap,
                corpus,
                "matching_failure",
                EscalationConfig(max_doc_requests=2, matching_failure_extra_unredacts=1),
            )
            if a.kind == "request_unredact"
        ]
    )
    assert base_n == 2
    assert boosted_n >= 3


def test_critical_first_prefers_critical_in_crowded_room():
    from .allocate import allocate_day
    from .types import Analyst, AnalystLoad, ControlFlags, Deal, Document

    a = Analyst(id="a0", load=AnalystLoad.FRESH, desk="sec0", expertise=0.8)
    deal = Deal(
        id="d0",
        sector="sec0",
        pending_review=0,
        daily_review_capacity=20,
        expected_organic_docs_per_day=1,
    )
    docs = {
        "c": Document(
            id="c",
            deal_id="d0",
            kind="c",
            text="material weakness going concern",
            risk_score=0.9,
            planted_risk=True,
            available=True,
        )
    }
    # Crowded room: ≥8 open docs so soft critical-first engages.
    for i in range(8):
        docs[f"f{i}"] = Document(
            id=f"f{i}",
            deal_id="d0",
            kind="f",
            text=f"ordinary update {i}",
            risk_score=0.2,
            planted_risk=False,
        )
    flags = ControlFlags(
        analyst_rules=True,
        desk_rules=True,
        use_model_signals=True,
        critical_first=True,
        thin_stream_absorb=True,
    )
    out = allocate_day(0, {"a0": a}, {"d0": deal}, docs, flags)
    assert out.decisions
    assert out.decisions[0].document_id == "c"
    assert out.blocked.get("critical_first", 0) >= 1
