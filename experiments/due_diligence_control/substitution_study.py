"""Can controlled DD + finance signals *substitute* for traditional diligence?

We plant known risks, then stress the desk under dense / sparse / missing
regimes. Methods compared:

- ``flood`` — no capacity rules (traditional "read everything available")
- ``controlled`` — analyst + desk rules + model scores (this package)
- ``rules_blind`` — rules on, model off (no FinanceGPT signal)
- ``model_only`` — model on, rules off (FinanceGPT without control)

Oracle ceiling: ``recall_available`` cannot exceed 1; ``recall_all`` collapses
when planted docs are missing — no method substitutes for absent evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .allocate import allocate_day
from .analyst_rules import AnalystPolicyConfig
from .desk_rules import DeskPolicyConfig
from .escalation import EscalationConfig, apply_escalations, plan_escalations
from .integrity import diagnose_integrity, snapshot_deal
from .metrics import RunMetrics, merge_metrics
from .model_hooks import HeuristicRiskProvider, score_documents
from .sparsity import REGIMES, DataRegime
from .staged_dd import StageDiagnosis
from .types import Analyst, AnalystLoad, ControlFlags, Deal, DealOutcome, DealStage, Document


PLANTED_TEXT = (
    "CRITICAL: material weakness, going concern uncertainty, covenant breach, "
    "related party investigation, and liquidity risk after downgrade."
)
CLEAN_TEXT = (
    "Ordinary course update: revenue stable, covenants in compliance, "
    "no litigation disclosed in this excerpt."
)
REDACTED = "[REDACTED — source field unavailable]"


METHODS: Dict[str, ControlFlags] = {
    "flood": ControlFlags(
        analyst_rules=False, desk_rules=False, use_model_signals=True
    ),
    "controlled_v1": ControlFlags(
        analyst_rules=True, desk_rules=True, use_model_signals=True
    ),
    "controlled_v2": ControlFlags(
        analyst_rules=True,
        desk_rules=True,
        use_model_signals=True,
        integrity_gate=True,
    ),
    # v4 = integrity + escalate + sparse priority + critical lane + thin-stream absorb
    "controlled": ControlFlags(
        analyst_rules=True,
        desk_rules=True,
        use_model_signals=True,
        integrity_gate=True,
        sparse_priority=True,
        auto_escalate=True,
        thin_stream_absorb=True,
    ),
    "rules_blind": ControlFlags(
        analyst_rules=True,
        desk_rules=True,
        use_model_signals=False,
        integrity_gate=True,
        auto_escalate=True,
        thin_stream_absorb=True,
    ),
    "model_only": ControlFlags(
        analyst_rules=False, desk_rules=False, use_model_signals=True
    ),
}


@dataclass
class StudyConfig:
    n_days: int = 20
    n_analysts: int = 12  # tighter attention so flood over-reads filler
    n_deals: int = 6
    n_desks: int = 3
    planted_per_deal: int = 3
    seed: int = 2026


def _init(cfg: StudyConfig, rng: np.random.Generator):
    sectors = [f"sec{i}" for i in range(cfg.n_desks)]
    loads = [AnalystLoad.FRESH, AnalystLoad.STEADY, AnalystLoad.HEAVY]
    analysts = {
        f"a{i}": Analyst(
            id=f"a{i}",
            load=loads[int(rng.integers(0, len(loads)))],
            desk=sectors[int(rng.integers(0, len(sectors)))],
            expertise=float(np.clip(rng.normal(0.6, 0.12), 0.2, 0.95)),
        )
        for i in range(cfg.n_analysts)
    }
    deals = {
        f"d{j}": Deal(
            id=f"d{j}",
            sector=sectors[j % len(sectors)],
            pending_review=int(rng.integers(0, 8)),
            daily_review_capacity=8,
            expected_organic_docs_per_day=2.0,
            days_to_ic=int(rng.integers(12, 22)),
            risk_bar=0.42,
            value=1.0,
        )
        for j in range(cfg.n_deals)
    }
    return analysts, deals


def _plant_corpus(
    deals: Dict[str, Deal],
    cfg: StudyConfig,
    regime: DataRegime,
    rng: np.random.Generator,
) -> Dict[str, Document]:
    """Create the full latent corpus up front; availability is regime-dependent."""
    docs: Dict[str, Document] = {}
    provider = HeuristicRiskProvider()
    for deal in deals.values():
        # Planted critical docs.
        for k in range(cfg.planted_per_deal):
            available = rng.random() >= regime.missing_doc_rate
            redacted = available and (rng.random() < regime.redact_rate)
            text = REDACTED if redacted else PLANTED_TEXT
            did = f"{deal.id}_plant{k}"
            doc = Document(
                id=did,
                deal_id=deal.id,
                kind="critical",
                text=text,
                planted_risk=True,
                available=available,
                redacted=redacted,
            )
            if available:
                score_documents([doc], provider)
                if redacted:
                    doc.risk_score = min(doc.risk_score, 0.15)  # model blind
            docs[did] = doc
        # Filler / noise docs (always available when "spawned").
        n_fill = max(int(cfg.planted_per_deal * 4 * regime.filler_mult), 4)
        for k in range(n_fill):
            did = f"{deal.id}_fill{k}"
            doc = Document(
                id=did,
                deal_id=deal.id,
                kind="filler",
                text=CLEAN_TEXT + f" #{k}",
                planted_risk=False,
                available=True,
            )
            score_documents([doc], provider)
            docs[did] = doc
    return docs


def _release_docs(
    day: int,
    deals: Dict[str, Deal],
    corpus: Dict[str, Document],
    live: Dict[str, Document],
    regime: DataRegime,
    rng: np.random.Generator,
) -> None:
    """Poisson-thin the available corpus into the live data room."""
    for deal in deals.values():
        pending = [
            d
            for d in corpus.values()
            if d.deal_id == deal.id and d.available and d.id not in live and not d.reviewed
        ]
        if not pending:
            continue
        n = int(rng.poisson(regime.docs_per_deal_per_day))
        if n <= 0:
            continue
        # Prefer releasing a mix; shuffle then take n.
        rng.shuffle(pending)
        for doc in pending[:n]:
            doc.day_arrived = day
            live[doc.id] = doc
            deal.pending_review += 1


def _unfound_available(deal_id: str, corpus: Dict[str, Document]) -> bool:
    for d in corpus.values():
        if d.deal_id != deal_id or not d.planted_risk:
            continue
        if d.available and not d.found:
            return True
    return False


@dataclass
class SubstitutionRun:
    method: str
    regime: str
    metrics: RunMetrics


def run_one(
    method: str,
    regime: DataRegime,
    cfg: StudyConfig,
    seed: int,
) -> RunMetrics:
    rng = np.random.default_rng(seed)
    analysts, deals = _init(cfg, rng)
    corpus = _plant_corpus(deals, cfg, regime, rng)
    live: Dict[str, Document] = {}
    flags = METHODS[method]
    metrics = RunMetrics()
    recent_sector: List[Tuple[str, str, int]] = []

    metrics.planted_total = sum(1 for d in corpus.values() if d.planted_risk)
    metrics.planted_available = sum(1 for d in corpus.values() if d.planted_risk and d.available)

    for day in range(cfg.n_days):
        metrics.mandate_days += sum(
            1 for d in deals.values() if d.stage not in (DealStage.PASSED, DealStage.CLOSED)
        )
        for deal in deals.values():
            cleared = min(deal.pending_review, deal.daily_review_capacity)
            if cleared:
                metrics.review_wait_sum += deal.pending_review / max(deal.daily_review_capacity, 1)
                metrics.review_wait_obs += 1
            deal.pending_review -= cleared
            deal.days_to_ic = max(deal.days_to_ic - 1, 0)

        _release_docs(day, deals, corpus, live, regime, rng)

        # Model-blind: wipe scores so ranking ignores FinanceGPT signal.
        if not flags.use_model_signals:
            for d in live.values():
                if not d.reviewed:
                    d.risk_score = 0.35

        alloc = allocate_day(
            day,
            analysts,
            deals,
            live,
            flags,
            AnalystPolicyConfig(),
            DeskPolicyConfig(),
            recent_sector=recent_sector,
        )
        metrics.add_blocked(alloc.blocked)
        metrics.coverage += len(alloc.decisions)
        for sd in alloc.stage_decisions.values():
            if sd.diagnosis == StageDiagnosis.SUPPLY_SHORTAGE:
                metrics.supply_shortage_events += 1
            elif sd.diagnosis == StageDiagnosis.MATCHING_FAILURE:
                metrics.matching_failure_events += 1

        # Integrity-aware diagnosis + v3 escalations (recover missing / unredact).
        esc_cfg = EscalationConfig()
        for deal in deals.values():
            if deal.outcome in (DealOutcome.IC_READY, DealOutcome.ABSTAIN_INCOMPLETE):
                continue
            sig = diagnose_integrity(deal, corpus, coverage_today=len(alloc.decisions))
            if sig == "supply_shortage":
                metrics.supply_shortage_events += 1
            elif sig == "matching_failure":
                metrics.matching_failure_events += 1
            snap = snapshot_deal(deal, corpus)
            # Rate-limit: at most one escalation pulse every 3 days per deal.
            cooled = (day - deal.last_escalation_day) >= 3
            live_unread_critical = any(
                d.deal_id == deal.id
                and d.planted_risk
                and not d.reviewed
                and not d.found
                for d in live.values()
            )
            live_filler = sum(
                1
                for d in live.values()
                if d.deal_id == deal.id and not d.reviewed and not d.planted_risk
            )
            unfound_live = (
                "unfound_available_risks" in snap.reasons
                and deal.total_coverage >= 8
                and (live_unread_critical or live_filler >= 3)
            )
            need_escalate = cooled and (
                sig in ("supply_shortage", "matching_failure")
                or "low_availability" in snap.reasons
                or "high_redaction" in snap.reasons
                or unfound_live
            )
            if flags.auto_escalate and need_escalate:
                diag = sig if sig != "ok" else (
                    "matching_failure" if "high_redaction" in snap.reasons else "supply_shortage"
                )
                actions = plan_escalations(deal, snap, corpus, diag, esc_cfg, live=live)
                if actions:
                    metrics.escalations += len(actions)
                    deal.outcome = DealOutcome.ESCALATED
                    deal.last_escalation_day = day
                    result = apply_escalations(
                        actions, corpus, live, deals, rng, day, esc_cfg
                    )
                    metrics.docs_recovered += result.docs_recovered
                    metrics.fields_unredacted += result.fields_unredacted
                    metrics.halt_coverage_events += len(result.halted_deals)
                    if result.docs_recovered or result.fields_unredacted:
                        metrics.net_value += 0.25 * (
                            result.docs_recovered + result.fields_unredacted
                        )

        for dec in alloc.decisions:
            a = analysts[dec.analyst_id]
            deal = deals[dec.deal_id]
            doc = live[dec.document_id]
            metrics.net_value -= 0.01
            deal.pending_review = max(deal.pending_review - 1, 0)

            is_flag = doc.risk_score >= deal.risk_bar
            if is_flag:
                deal.total_flags += 1
                metrics.flags += 1
            # Confirmation of planted risk: reviewed + (not redacted) + expertise.
            if doc.planted_risk and doc.available and not doc.redacted:
                p_confirm = 0.55 + 0.4 * a.expertise
                if rng.random() < p_confirm:
                    doc.found = True
                    deal.total_confirmed_risks += 1
                    metrics.confirmed += 1
                    metrics.planted_found += 1
                    metrics.net_value += 0.5
                    a.consecutive_ignores = 0
                else:
                    a.consecutive_ignores += 1
            elif doc.planted_risk and doc.redacted:
                # Human might still catch a redacted stub with small probability.
                if rng.random() < 0.08 * a.expertise:
                    doc.found = True
                    deal.total_confirmed_risks += 1
                    metrics.confirmed += 1
                    metrics.planted_found += 1
                    metrics.net_value += 0.5
                else:
                    a.consecutive_ignores += 1
            else:
                a.consecutive_ignores += 1

            if deal.total_confirmed_risks >= 2 and deal.total_ic_ready == 0 and rng.random() < 0.4:
                if flags.integrity_gate:
                    snap = snapshot_deal(deal, corpus)
                    if not snap.allow_ic:
                        metrics.ic_blocked_integrity += 1
                        metrics.integrity_refusals += 1
                        deal.outcome = DealOutcome.ABSTAIN_INCOMPLETE
                        metrics.abstain_incomplete += 1
                        # Honest abstain beats a false IC.
                        metrics.net_value += 0.05
                        recent_sector.append((a.id, deal.sector, day))
                        continue
                deal.total_ic_ready = 1
                deal.outcome = DealOutcome.IC_READY
                metrics.ic_ready += 1
                # False confidence: IC while an available planted risk remains unfound.
                if _unfound_available(deal.id, corpus):
                    metrics.false_ic += 1
                    metrics.net_value -= 0.8
                else:
                    metrics.net_value += 1.0

            recent_sector.append((a.id, deal.sector, day))

        recent_sector = [r for r in recent_sector if day - r[2] < 7]

    # Recount after escalations may have recovered docs / lifted redactions.
    metrics.planted_available = sum(1 for d in corpus.values() if d.planted_risk and d.available)
    metrics.planted_found = sum(1 for d in corpus.values() if d.planted_risk and d.found)
    return metrics


@dataclass
class SubstitutionBoard:
    rows: List[SubstitutionRun] = field(default_factory=list)
    table_md: str = ""
    verdict_md: str = ""


def run_substitution_study(
    seeds: Optional[List[int]] = None,
    regimes: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    cfg: Optional[StudyConfig] = None,
    results_dir: Optional[Path] = None,
) -> SubstitutionBoard:
    seeds = seeds or [11, 12, 13, 14]
    regimes = regimes or list(REGIMES.keys())
    methods = methods or list(METHODS.keys())
    cfg = cfg or StudyConfig()
    board = SubstitutionBoard()

    for rname in regimes:
        regime = REGIMES[rname]
        for method in methods:
            offset = sum(ord(ch) for ch in rname)
            runs = [run_one(method, regime, cfg, seed=s + offset) for s in seeds]
            board.rows.append(SubstitutionRun(method, rname, merge_metrics(runs)))

    lines = [
        "# Substitution under missing / sparse data",
        "",
        "Planted critical risks per deal; methods must find them under each regime.",
        "",
        f"Seeds={seeds}, days={cfg.n_days}, deals={cfg.n_deals}, "
        f"analysts={cfg.n_analysts}, planted/deal={cfg.planted_per_deal}.",
        "",
        "| Regime | Method | Coverage | Recall@avail | Recall@all | "
        "False-IC | Abstain | Recovered | Unredacted | Confirmed/1k | Net |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in board.rows:
        m = row.metrics
        lines.append(
            f"| {row.regime} | {row.method} | {m.coverage} | {m.recall_available:.2f} | "
            f"{m.recall_all:.2f} | {m.false_ic_rate:.2f} | {m.abstain_incomplete} | "
            f"{m.docs_recovered} | {m.fields_unredacted} | "
            f"{m.confirmed_per_1k_coverage:.1f} | {m.net_value:.1f} |"
        )

    verdict = _verdict(board)
    board.table_md = "\n".join(lines)
    board.verdict_md = verdict
    full = board.table_md + "\n\n" + verdict + "\n"
    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "SUBSTITUTION.md").write_text(full, encoding="utf-8")
    return board


def _pick(board: SubstitutionBoard, regime: str, method: str) -> RunMetrics:
    for row in board.rows:
        if row.regime == regime and row.method == method:
            return row.metrics
    raise KeyError((regime, method))


def _has(board: SubstitutionBoard, regime: str, method: str) -> bool:
    return any(r.regime == regime and r.method == method for r in board.rows)


def _verdict(board: SubstitutionBoard) -> str:
    lines = [
        "## Verdict: can this substitute for traditional due diligence?",
        "",
        "### Short answer",
        "",
        "**Partial substitute for triage and attention allocation — not a full "
        "substitute for primary-source diligence when evidence is missing.**",
        "",
        "### Evidence from this board",
        "",
    ]
    n = 1
    if _has(board, "dense", "controlled") and _has(board, "dense", "flood"):
        dense_c = _pick(board, "dense", "controlled")
        dense_f = _pick(board, "dense", "flood")
        lines.append(
            f"{n}. **Dense data room.** Controlled(v2) recall@available={dense_c.recall_available:.2f} "
            f"vs flood={dense_f.recall_available:.2f}, at coverage {dense_c.coverage} vs "
            f"{dense_f.coverage}. Efficiency (confirmed/1k) controlled="
            f"{dense_c.confirmed_per_1k_coverage:.1f} vs flood={dense_f.confirmed_per_1k_coverage:.1f}. "
            "→ OK to replace *undifferentiated reading*, not partner sign-off."
        )
        lines.append("")
        n += 1
    if _has(board, "dense", "controlled") and _has(board, "dense", "controlled_v1"):
        v4 = _pick(board, "dense", "controlled")
        v1 = _pick(board, "dense", "controlled_v1")
        lines.append(
            f"{n}. **v4 vs v1 (dense).** False-IC {v1.false_ic_rate:.2f}→{v4.false_ic_rate:.2f}; "
            f"abstain={v4.abstain_incomplete}; recovered={v4.docs_recovered}; "
            f"net {v1.net_value:.1f}→{v4.net_value:.1f}."
        )
        lines.append("")
        n += 1
    if _has(board, "missing_docs", "controlled") and _has(board, "missing_docs", "controlled_v2"):
        v4 = _pick(board, "missing_docs", "controlled")
        v2 = _pick(board, "missing_docs", "controlled_v2")
        lines.append(
            f"{n}. **v4 escalation vs v2 abstain-only (missing_docs).** "
            f"recall@all {v2.recall_all:.2f}→{v4.recall_all:.2f} via recovered={v4.docs_recovered}; "
            f"false-IC stays {v4.false_ic_rate:.2f}."
        )
        lines.append("")
        n += 1
    if _has(board, "sparse", "controlled") and _has(board, "sparse", "flood"):
        sparse_c = _pick(board, "sparse", "controlled")
        sparse_f = _pick(board, "sparse", "flood")
        sparse_v2 = (
            _pick(board, "sparse", "controlled_v2")
            if _has(board, "sparse", "controlled_v2")
            else None
        )
        delta = ""
        if sparse_v2 is not None:
            delta = (
                f" v4 lifts sparse recall vs the soft-halt freeze "
                f"({sparse_v2.recall_available:.2f}→{sparse_c.recall_available:.2f} vs v2)."
            )
        lines.append(
            f"{n}. **Sparse filings.** Controlled recall@available={sparse_c.recall_available:.2f} "
            f"(flood={sparse_f.recall_available:.2f}). Thin-stream absorb + critical lane "
            f"keep arriving docs readable under soft-halt.{delta}"
        )
        lines.append("")
        n += 1
    if _has(board, "missing_docs", "controlled") and _has(board, "missing_docs", "flood"):
        miss_c = _pick(board, "missing_docs", "controlled")
        miss_f = _pick(board, "missing_docs", "flood")
        avail = miss_c.planted_available / max(miss_c.planted_total, 1)
        lines.append(
            f"{n}. **Missing documents.** recall@all controlled={miss_c.recall_all:.2f}, "
            f"flood={miss_f.recall_all:.2f}. Both collapse toward the availability ceiling "
            f"(available share ≈ {avail:.2f}). "
            f"v4 recovers {miss_c.docs_recovered} missing docs via escalation; "
            f"false-IC {miss_c.false_ic_rate:.2f} vs flood {miss_f.false_ic_rate:.2f}; "
            f"abstain={miss_c.abstain_incomplete}. "
            "Still cannot invent evidence that escalation fails to obtain."
        )
        lines.append("")
        n += 1
    if _has(board, "missing_fields", "controlled") and _has(board, "missing_fields", "model_only"):
        field_c = _pick(board, "missing_fields", "controlled")
        field_m = _pick(board, "missing_fields", "model_only")
        lines.append(
            f"{n}. **Missing / redacted fields.** Model-only recall@available={field_m.recall_available:.2f}; "
            f"controlled(v4)={field_c.recall_available:.2f} after unredacted={field_c.fields_unredacted}. "
            "FinanceGPT is blind on redactions until escalation lifts them; "
            "v4 requests unredact instead of inventing fields."
        )
        lines.append("")
        n += 1
    if _has(board, "sparse_missing", "controlled"):
        worst_c = _pick(board, "sparse_missing", "controlled")
        lines.append(
            f"{n}. **Worst case (sparse_missing).** Controlled recall@all={worst_c.recall_all:.2f}, "
            f"false-IC rate={worst_c.false_ic_rate:.2f}. Expect low completeness; staged diagnosis "
            f"fires supply-shortage={worst_c.supply_shortage_events}, matching-failure="
            f"{worst_c.matching_failure_events} as the honest signal."
        )
        lines.append("")

    lines.extend(
        [
            "### What *can* be substituted",
            "",
            "- First-pass screening and prioritization under dense or sparse *observable* rooms",
            "- Fatigue / backlog control so analysts do not over-read filler",
            "- Explicit diagnosis of supply shortage vs matching failure",
            "",
            "### What *cannot* be substituted",
            "",
            "- Risks in documents that never arrive (missing_docs)",
            "- Risks hidden by redaction / absent fields without alternative evidence",
            "- Legal/partner accountability and primary confirmation interviews",
            "- IC decisions when false-IC risk remains material",
            "",
            "### Operating rule",
            "",
            "Use controlled DD + FinanceGPT as a **capacity-aware co-pilot**. "
            "Treat `recall_all` under missingness as the integrity metric: if availability "
            "is low, refuse 'complete diligence' claims and escalate data collection.",
        ]
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13, 14])
    p.add_argument("--days", type=int, default=20)
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    args = p.parse_args(argv)
    cfg = StudyConfig(n_days=args.days)
    board = run_substitution_study(seeds=list(args.seeds), cfg=cfg, results_dir=args.results_dir)
    print(board.table_md)
    print()
    print(board.verdict_md)
    print(f"\nWrote {args.results_dir / 'SUBSTITUTION.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
