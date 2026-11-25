"""LLM eval prototype: PO-risk (FSDS) attribution on judge scores.

Primary endpoint: recover the planted concept-drift feature block via
PO(S) = mean(tau_S^2), with random-forest nuisance models.
Does not modify existing CFPerm / RFPerm / ood_vimp libraries.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
