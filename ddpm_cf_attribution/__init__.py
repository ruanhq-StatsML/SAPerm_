"""Conditional DDPM counterfactual attribution on a 10-site linear panel.

The product call is ``ddpm_cf_attribution.run()``.
"""

__version__ = "0.1.0"


def run(family: str = "linear", smoke: bool = False, out_dir=None):
    """Train, attribute, and write the five-layer report. Returns the result dict."""
    from ddpm_cf_attribution.run_experiment import run as _run

    return _run(family=family, smoke=smoke, out_dir=out_dir)


__all__ = ["run"]
