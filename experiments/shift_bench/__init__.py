"""Clean testing surface: digest → (optional period embed) → interactions → benchmark."""

from experiments.shift_bench.digest import Digested, digest
from experiments.shift_bench.embed_periods import discretize_periods, embed_concat
from experiments.shift_bench.interactions import ood_vimp, window_bundle

__all__ = [
    "Digested",
    "digest",
    "discretize_periods",
    "embed_concat",
    "ood_vimp",
    "run_benchmark_all",
    "window_bundle",
]


def run_benchmark_all(*args, **kwargs):
    from experiments.shift_bench.run_benchmark_all import run

    return run(*args, **kwargs)
