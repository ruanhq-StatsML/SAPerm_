"""Run the due-diligence control factorial board."""

from __future__ import annotations

import argparse
from pathlib import Path

from .factorial import run_factorial
from .simulator import SimConfig


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=int, default=25)
    p.add_argument("--analysts", type=int, default=24)
    p.add_argument("--deals", type=int, default=8)
    p.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    args = p.parse_args(argv)
    cfg = SimConfig(n_days=args.days, n_analysts=args.analysts, n_deals=args.deals)
    result = run_factorial(seeds=list(args.seeds), base=cfg, results_dir=args.results_dir)
    print(result.table_md)
    print(f"Wrote {args.results_dir / 'FACTORIAL.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
