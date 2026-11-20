"""v3: turn integrity refusals into actionable escalations.

Refusing IC is necessary but not sufficient. When evidence is missing or
redacted, the desk should request alternative / unredacted sources and
halt undifferentiated coverage — not keep reading filler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .integrity import IntegritySnapshot
from .types import Deal, Document


@dataclass(frozen=True)
class EscalationConfig:
    # Max missing planted docs to surface per escalation pulse.
    max_doc_requests: int = 2
    # Probability that a requested missing doc arrives via alternative channel.
    alt_data_success_rate: float = 0.35
    # Probability that a redaction is lifted after an unredact request.
    unredact_success_rate: float = 0.40


@dataclass
class EscalationAction:
    deal_id: str
    kind: str  # request_missing_docs | request_unredact | halt_broad_coverage | alt_data
    reason: str
    target_doc_id: Optional[str] = None


@dataclass
class EscalationResult:
    actions: List[EscalationAction] = field(default_factory=list)
    docs_recovered: int = 0
    fields_unredacted: int = 0
    halted_deals: List[str] = field(default_factory=list)


def plan_escalations(
    deal: Deal,
    snap: IntegritySnapshot,
    corpus: Dict[str, Document],
    diagnosis: str,
    cfg: Optional[EscalationConfig] = None,
) -> List[EscalationAction]:
    cfg = cfg or EscalationConfig()
    actions: List[EscalationAction] = []

    if diagnosis == "supply_shortage" or "low_availability" in snap.reasons:
        missing = [
            d
            for d in corpus.values()
            if d.deal_id == deal.id and d.planted_risk and not d.available
        ]
        for doc in missing[: cfg.max_doc_requests]:
            actions.append(
                EscalationAction(deal.id, "request_missing_docs", "low_availability", doc.id)
            )
            actions.append(
                EscalationAction(deal.id, "alt_data", "seek_alternative_channel", doc.id)
            )
        actions.append(
            EscalationAction(deal.id, "halt_broad_coverage", "supply_shortage_no_blind_spend")
        )

    if diagnosis == "matching_failure" or "high_redaction" in snap.reasons:
        redacted = [
            d
            for d in corpus.values()
            if d.deal_id == deal.id and d.planted_risk and d.available and d.redacted and not d.found
        ]
        for doc in redacted[: cfg.max_doc_requests]:
            actions.append(
                EscalationAction(deal.id, "request_unredact", "high_redaction", doc.id)
            )
        actions.append(
            EscalationAction(deal.id, "halt_broad_coverage", "matching_failure_retarget")
        )

    # Only after some reading has happened — don't freeze a fresh deal.
    if "unfound_available_risks" in snap.reasons and deal.total_coverage >= 8:
        actions.append(
            EscalationAction(deal.id, "halt_broad_coverage", "finish_unread_critical_first")
        )

    # Deduplicate by (kind, target).
    seen = set()
    uniq: List[EscalationAction] = []
    for a in actions:
        key = (a.kind, a.target_doc_id, a.reason)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(a)
    return uniq


def apply_escalations(
    actions: List[EscalationAction],
    corpus: Dict[str, Document],
    live: Dict[str, Document],
    deals: Dict[str, Deal],
    rng,
    day: int,
    cfg: Optional[EscalationConfig] = None,
    provider=None,
) -> EscalationResult:
    """Resolve escalations: recover missing docs / lift redactions with stochastic success."""
    from .model_hooks import HeuristicRiskProvider, score_documents

    cfg = cfg or EscalationConfig()
    provider = provider or HeuristicRiskProvider()
    out = EscalationResult(actions=list(actions))
    halted = set()
    recovered_deals = set()

    # 1) Resolve evidence requests first.
    for act in actions:
        deal = deals.get(act.deal_id)
        if deal is None or act.target_doc_id is None:
            continue
        doc = corpus.get(act.target_doc_id)
        if doc is None:
            continue

        if act.kind in ("request_missing_docs", "alt_data") and not doc.available:
            if rng.random() < cfg.alt_data_success_rate:
                doc.available = True
                if doc.redacted:
                    doc.text = (
                        "CRITICAL: material weakness, going concern uncertainty, covenant breach, "
                        "related party investigation, and liquidity risk after downgrade."
                    )
                    doc.redacted = False
                score_documents([doc], provider)
                doc.day_arrived = day
                live[doc.id] = doc
                deal.pending_review += 1
                recovered_deals.add(deal.id)
                out.docs_recovered += 1
        elif act.kind == "request_unredact" and doc.available and doc.redacted:
            if rng.random() < cfg.unredact_success_rate:
                doc.redacted = False
                doc.text = (
                    "CRITICAL: material weakness, going concern uncertainty, covenant breach, "
                    "related party investigation, and liquidity risk after downgrade."
                )
                score_documents([doc], provider)
                recovered_deals.add(deal.id)
                out.fields_unredacted += 1

    # 2) Soft-halt only deals that did *not* obtain new evidence this pulse.
    for act in actions:
        if act.kind != "halt_broad_coverage":
            continue
        deal = deals.get(act.deal_id)
        if deal is None:
            continue
        if act.deal_id in recovered_deals:
            deal.paused = False
            continue
        deal.paused = True
        halted.add(deal.id)

    for did in recovered_deals:
        deals[did].paused = False

    out.halted_deals = sorted(halted)
    return out
