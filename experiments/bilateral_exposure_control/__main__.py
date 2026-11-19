"""Run the Phase-1 bilateral exposure control factorial board."""

from __future__ import annotations

import argparse
from pathlib import Path

from .factorial import run_factorial
from .simulator import SimConfig


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--candidates", type=int, default=120)
    p.add_argument("--companies", type=int, default=6)
    p.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028, 2029])
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    args = p.parse_args(argv)

    cfg = SimConfig(
        n_days=args.days,
        n_candidates=args.candidates,
        n_companies=args.companies,
    )
    result = run_factorial(seeds=list(args.seeds), base=cfg, results_dir=args.results_dir)
    print(result.table_md)
    print(f"Wrote {args.results_dir / 'FACTORIAL.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
