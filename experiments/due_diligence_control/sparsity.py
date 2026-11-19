"""Missing / sparse data regimes for the substitution argument.

Three failure modes are deliberately separated:

1. **Sparse arrival** — few documents per day (thin data room).
2. **Missing documents** — planted-risk files never arrive (MNAR/MCAR).
3. **Missing fields** — file arrives but critical text is redacted before scoring.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataRegime:
    name: str
    docs_per_deal_per_day: float
    # Probability a planted-risk document is withheld entirely.
    missing_doc_rate: float = 0.0
    # Probability available planted text is redacted before the model sees it.
    redact_rate: float = 0.0
    # Extra clean filler multiplier (noise volume).
    filler_mult: float = 1.0
    note: str = ""


REGIMES: dict[str, DataRegime] = {
    "dense": DataRegime(
        "dense",
        docs_per_deal_per_day=6.0,
        missing_doc_rate=0.0,
        redact_rate=0.0,
        filler_mult=3.5,
        note="Full noisy data room; capacity binds so triage can save waste.",
    ),
    "sparse": DataRegime(
        "sparse",
        docs_per_deal_per_day=0.55,
        missing_doc_rate=0.0,
        redact_rate=0.0,
        filler_mult=1.0,
        note="Thin filings stream; attention allocation matters most.",
    ),
    "missing_docs": DataRegime(
        "missing_docs",
        docs_per_deal_per_day=3.5,
        missing_doc_rate=0.45,
        redact_rate=0.0,
        filler_mult=2.0,
        note="Critical evidence never arrives — hard ceiling on recall_all.",
    ),
    "missing_fields": DataRegime(
        "missing_fields",
        docs_per_deal_per_day=3.5,
        missing_doc_rate=0.05,
        redact_rate=0.55,
        filler_mult=2.0,
        note="Docs arrive but risk language is stripped — model blind spot.",
    ),
    "sparse_missing": DataRegime(
        "sparse_missing",
        docs_per_deal_per_day=0.7,
        missing_doc_rate=0.35,
        redact_rate=0.25,
        filler_mult=1.5,
        note="Worst case: sparse + missing docs + redactions.",
    ),
}
