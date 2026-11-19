"""Run due-diligence boards: factorial and/or missing-data substitution study."""

from __future__ import annotations

import argparse
from pathlib import Path

from .factorial import run_factorial
from .simulator import SimConfig
from .substitution_study import StudyConfig, run_substitution_study


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=("factorial", "substitution", "both"),
        default="both",
    )
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
    results = args.results_dir

    if args.mode in ("factorial", "both"):
        cfg = SimConfig(n_days=args.days, n_analysts=args.analysts, n_deals=args.deals)
        result = run_factorial(seeds=list(args.seeds), base=cfg, results_dir=results)
        print(result.table_md)
        print(f"Wrote {results / 'FACTORIAL.md'}")

    if args.mode in ("substitution", "both"):
        study = run_substitution_study(
            seeds=list(args.seeds),
            cfg=StudyConfig(n_days=min(args.days, 20), n_analysts=16, n_deals=6),
            results_dir=results,
        )
        print(study.table_md)
        print()
        print(study.verdict_md)
        print(f"Wrote {results / 'SUBSTITUTION.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
