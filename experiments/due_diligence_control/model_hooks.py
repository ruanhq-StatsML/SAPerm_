"""Model hooks for due-diligence risk signals.

Phase 1 does **not** require downloading FinanceGPT. A heuristic scorer
keeps the control loop testable. Optional backends:

- ``finbert`` — ProsusAI/finbert (sentiment / risk tone proxy)
- ``prompted`` — placeholder for FinanceGPT / FinGPT / AdaptLLM finance-LLM
  via a callable that maps text → risk in [0, 1]

FinanceGPT-class models are *signal providers*, not traffic controllers.
Hard caps (mute, backlog_hard, frequency) always win.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol, Sequence

from .types import Document


class RiskSignalProvider(Protocol):
    name: str

    def score(self, text: str) -> float:
        """Return risk / materiality in [0, 1]."""


RISK_LEXICON = (
    "going concern",
    "material weakness",
    "restatement",
    "litigation",
    "default",
    "covenant breach",
    "impairment",
    "fraud",
    "investigation",
    "related party",
    "going-concern",
    "liquidity risk",
    "downgrade",
)


@dataclass
class HeuristicRiskProvider:
    """Keyword + length proxy. Deterministic, no network."""

    name: str = "heuristic"

    def score(self, text: str) -> float:
        low = text.lower()
        hits = sum(1 for k in RISK_LEXICON if k in low)
        density = hits / max(len(re.findall(r"\w+", low)), 1)
        # Mild length prior: longer excerpts slightly more chance of a flag.
        length_prior = min(len(low) / 4000.0, 0.25)
        return float(min(1.0, 0.12 * hits + 4.0 * density + length_prior))


@dataclass
class PromptedRiskProvider:
    """Wrap any FinanceGPT / FinGPT callable: text → risk."""

    fn: Callable[[str], float]
    name: str = "prompted_finance_llm"

    def score(self, text: str) -> float:
        return float(min(1.0, max(0.0, self.fn(text))))


@dataclass
class FinBertRiskProvider:
    """Optional FinBERT tone → risk. Imports transformers lazily."""

    model_id: str = "ProsusAI/finbert"
    name: str = "finbert"

    def __post_init__(self) -> None:
        self._pipe = None

    def _ensure(self):
        if self._pipe is not None:
            return
        from transformers import pipeline  # type: ignore

        self._pipe = pipeline("sentiment-analysis", model=self.model_id, truncation=True)

    def score(self, text: str) -> float:
        self._ensure()
        out = self._pipe(text[:512])[0]
        label = str(out.get("label", "")).lower()
        conf = float(out.get("score", 0.0))
        if "neg" in label:
            return min(1.0, 0.55 + 0.45 * conf)
        if "pos" in label:
            return max(0.0, 0.35 * (1.0 - conf))
        return 0.4 * conf


def get_provider(kind: str = "heuristic") -> RiskSignalProvider:
    if kind == "heuristic":
        return HeuristicRiskProvider()
    if kind == "finbert":
        return FinBertRiskProvider()
    raise ValueError(f"unknown provider {kind!r}; use heuristic|finbert or PromptedRiskProvider")


def score_documents(docs: Sequence[Document], provider: RiskSignalProvider) -> List[Document]:
    for d in docs:
        d.risk_score = provider.score(d.text)
    return list(docs)


# Suggested fine-tune / serve targets for Phase 2 (not downloaded here):
FINANCE_MODEL_CANDIDATES = [
    {
        "id": "ProsusAI/finbert",
        "role": "fast risk/tone classifier on excerpts",
        "why": "mature, tiny, good as first-pass filter before LLM",
    },
    {
        "id": "AdaptLLM/finance-LLM",
        "role": "domain-adapted generative finance LLM",
        "why": "stronger than random 'FinanceGPT' forks for memo drafting",
    },
    {
        "id": "FinGPT/fingpt-mt_llama3",
        "role": "multi-task FinGPT LoRA family",
        "why": "sentiment / filing QA; keep as signal, not allocator",
    },
    {
        "id": "custom-FinanceGPT-lora",
        "role": "your DD corpus SFT/DPO",
        "why": "fine-tune on red-flag labels + IC memos; plug via PromptedRiskProvider",
    },
]
