"""Phase-1 rule-based due-diligence coverage control + finance model hooks."""

from .escalation import plan_escalations
from .factorial import run_factorial
from .integrity import snapshot_deal
from .model_hooks import FINANCE_MODEL_CANDIDATES, get_provider
from .simulator import DueDiligenceSim, SimConfig
from .substitution_study import run_substitution_study

__all__ = [
    "run_factorial",
    "run_substitution_study",
    "DueDiligenceSim",
    "SimConfig",
    "get_provider",
    "FINANCE_MODEL_CANDIDATES",
    "snapshot_deal",
    "plan_escalations",
]
