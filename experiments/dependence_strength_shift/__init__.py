"""Dependence strength shift: signed η² change for X_j | X_{-j} across batches."""

from experiments.dependence_strength_shift.dgp import (
    inject_residual_strength,
    load_uci_matrix,
    make_synthetic_strength_batches,
    make_uci_factor_batches,
    make_uci_strength_batches,
)
from experiments.dependence_strength_shift.metrics import ranking_recovery

__all__ = [
    "inject_residual_strength",
    "load_uci_matrix",
    "make_synthetic_strength_batches",
    "make_uci_factor_batches",
    "make_uci_strength_batches",
    "ranking_recovery",
]
