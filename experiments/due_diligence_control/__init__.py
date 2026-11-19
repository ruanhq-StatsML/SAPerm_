"""Phase-1 rule-based due-diligence coverage control + finance model hooks."""

from .factorial import run_factorial
from .model_hooks import FINANCE_MODEL_CANDIDATES, get_provider
from .simulator import DueDiligenceSim, SimConfig

__all__ = [
    "run_factorial",
    "DueDiligenceSim",
    "SimConfig",
    "get_provider",
    "FINANCE_MODEL_CANDIDATES",
]
