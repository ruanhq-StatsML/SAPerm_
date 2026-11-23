"""Online PO-risk, with the query window chosen by active sampling."""

from experiments.online_po_risk.score import (
    MMD_LOCO,
    RF_domain_classifier,
    fit_dr_tau,
    ood_vimp,
    ood_vimp_wrapper,
    online_po_shapley,
    tau_shapley_importance,
)


def run_comparison(*args, **kwargs):
    from experiments.online_po_risk.run import run_comparison as _run

    return _run(*args, **kwargs)


__all__ = [
    "MMD_LOCO",
    "RF_domain_classifier",
    "fit_dr_tau",
    "ood_vimp",
    "ood_vimp_wrapper",
    "online_po_shapley",
    "run_comparison",
    "tau_shapley_importance",
]
