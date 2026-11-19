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
    m = run_one("controlled", REGIMES["missing_docs"], cfg, seed=3)
    assert m.planted_available < m.planted_total
    assert m.recall_all <= m.recall_available + 1e-9
    # Hard ceiling: cannot find more than arrived.
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
        cfg=StudyConfig(n_days=6, n_analysts=8, n_deals=3, planted_per_deal=2),
        results_dir=None,
    )
    assert len(board.rows) == 4
    miss_c = next(r for r in board.rows if r.regime == "missing_docs" and r.method == "controlled")
    dense_c = next(r for r in board.rows if r.regime == "dense" and r.method == "controlled")
    assert miss_c.metrics.recall_all <= dense_c.metrics.recall_all + 0.05


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
    v2 = run_one("controlled", REGIMES["missing_docs"], cfg, seed=9)
    # Same seed → same discovery path; gate should not increase false IC.
    assert v2.false_ic_rate <= v1.false_ic_rate + 1e-9
    assert v2.ic_blocked_integrity >= 0
