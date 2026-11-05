"""Out-of-distribution variable importance, importable from the repo root.

The procedures live in ``experiments.deep_active_sampling.ood_vimp``. This
module is the import path from the sampling sketch:

``from OOD_vimp_wrapper import permuCATE_fit_nuisance, ...``
"""

from experiments.deep_active_sampling.ood_vimp import (
    MMD_LOCO,
    NAMMD_LOCO,
    RF_domain_classifier,
    brier_L2,
    cauchy_combine_pval,
    ood_vimp_wrapper,
    permuCATE_calPO,
    permuCATE_fit_nuisance,
    permuCATE_vimp,
    permuCATE_vimp_withPOModel,
    po_risk,
    po_risk_index,
    po_risk_inference,
    pool_feature_weights,
    pred_vimp_wrapper,
    pred_vimp_wrapper_loco,
)

__all__ = [
    "MMD_LOCO",
    "NAMMD_LOCO",
    "RF_domain_classifier",
    "brier_L2",
    "cauchy_combine_pval",
    "ood_vimp_wrapper",
    "permuCATE_calPO",
    "permuCATE_fit_nuisance",
    "permuCATE_vimp",
    "permuCATE_vimp_withPOModel",
    "po_risk",
    "po_risk_index",
    "po_risk_inference",
    "pool_feature_weights",
    "pred_vimp_wrapper",
    "pred_vimp_wrapper_loco",
]
