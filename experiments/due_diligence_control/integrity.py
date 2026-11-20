"""Completeness / integrity gates for due-diligence IC claims.

v2 lesson from the substitution board: high false-IC under missingness means
the system must *refuse* completeness claims when evidence is thin, redacted,
or still unread — not just rank documents harder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from .types import Deal, Document


@dataclass(frozen=True)
class IntegrityConfig:
    # Refuse IC if fewer than this share of planted/critical docs are available.
    min_availability: float = 0.75
    # Refuse IC if too many available critical docs are still unread/unfound.
    max_unfound_available_share: float = 0.25
    # Refuse IC if redaction share among available critical docs is high.
    max_redaction_share: float = 0.30
    # Minimum confirmed risks before IC is even considered.
    min_confirmed: int = 2


@dataclass
class IntegritySnapshot:
    deal_id: str
    planted_total: int
    planted_available: int
    planted_found: int
    planted_redacted: int
    availability: float
    unfound_available_share: float
    redaction_share: float
    allow_ic: bool
    reasons: List[str]


def _critical(docs: Iterable[Document], deal_id: str) -> List[Document]:
    return [d for d in docs if d.deal_id == deal_id and d.planted_risk]


def snapshot_deal(
    deal: Deal,
    corpus: Dict[str, Document],
    cfg: Optional[IntegrityConfig] = None,
) -> IntegritySnapshot:
    cfg = cfg or IntegrityConfig()
    crit = _critical(corpus.values(), deal.id)
    total = len(crit)
    available = [d for d in crit if d.available]
    found = [d for d in available if d.found]
    redacted = [d for d in available if d.redacted]
    n_avail = len(available)
    n_found = len(found)
    n_red = len(redacted)
    availability = n_avail / max(total, 1)
    unfound_share = (n_avail - n_found) / max(n_avail, 1)
    red_share = n_red / max(n_avail, 1)

    reasons: List[str] = []
    allow = True
    if deal.total_confirmed_risks < cfg.min_confirmed:
        allow = False
        reasons.append("insufficient_confirmed")
    if availability < cfg.min_availability:
        allow = False
        reasons.append("low_availability")
    if unfound_share > cfg.max_unfound_available_share:
        allow = False
        reasons.append("unfound_available_risks")
    if red_share > cfg.max_redaction_share:
        allow = False
        reasons.append("high_redaction")

    return IntegritySnapshot(
        deal_id=deal.id,
        planted_total=total,
        planted_available=n_avail,
        planted_found=n_found,
        planted_redacted=n_red,
        availability=availability,
        unfound_available_share=unfound_share,
        redaction_share=red_share,
        allow_ic=allow,
        reasons=reasons,
    )


def diagnose_integrity(
    deal: Deal,
    corpus: Dict[str, Document],
    coverage_today: int,
) -> str:
    """Extra stage-like signal when corpus completeness is the bottleneck."""
    snap = snapshot_deal(deal, corpus)
    if snap.availability < 0.7:
        return "supply_shortage"
    if snap.redaction_share > 0.4 and coverage_today > 0:
        return "matching_failure"
    if snap.unfound_available_share > 0.5 and deal.total_coverage >= 20:
        return "supply_shortage"
    return "ok"
