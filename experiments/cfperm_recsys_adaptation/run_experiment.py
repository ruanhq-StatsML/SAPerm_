"""Entry point for the CFPerm update-selection benchmark.

Smoke test:

    python experiments/cfperm_recsys_adaptation/run_experiment.py --dataset synthetic --seeds 3 --permutations 99 --quick

From this directory the same command is ``python run_experiment.py ...``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cfperm_recsys_adaptation.dgp import SCENARIOS  # noqa: E402
from experiments.cfperm_recsys_adaptation.experiment import run_experiment  # noqa: E402


def _seeds(text: str) -> list[int]:
    text = text.strip()
    if "," in text:
        return [int(part) for part in text.split(",") if part]
    count = int(text)
    return [202601 + i for i in range(count)]


def _budgets(text: str | None, quick: bool) -> list[int]:
    if text:
        return [int(part) for part in text.split(",") if part]
    return [100] if quick else [50, 100, 250, 500]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CFPerm-guided update selection benchmark")
    parser.add_argument("--dataset", default="synthetic", choices=("synthetic", "modasense"))
    parser.add_argument("--seeds", default="3")
    parser.add_argument("--permutations", type=int, default=99)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--label-budgets", default=None)
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    scenarios = [part.strip() for part in args.scenarios.split(",") if part.strip()]
    unknown = [name for name in scenarios if name not in SCENARIOS]
    if unknown:
        parser.error(f"unknown scenarios: {unknown}")
    output = Path(args.output) if args.output else Path(__file__).resolve().parent / "results"
    if args.dataset == "modasense":
        location = Path(args.input) if args.input else None
        found = location is not None and location.exists()
        output.mkdir(parents=True, exist_ok=True)
        message = (
            "# ModaSense pilot not run\n\n"
            "No verified ModaSense or CMU-MOSEI embedding table was available in this checkout. "
            "The runner does not substitute a proxy label or a synthetic table for that pilot.\n\n"
            f"- requested input: {location}\n"
            f"- input exists: {found}\n"
        )
        (output / "summary.md").write_text(message)
        print(message)
        return 2
    import torch

    torch.set_num_threads(1)
    config = {
        "dataset": args.dataset,
        "quick": bool(args.quick),
        "seeds": _seeds(args.seeds),
        "permutations": int(args.permutations),
        "label_budgets": _budgets(args.label_budgets, args.quick),
        "scenarios": scenarios,
        "output": output,
    }
    run_experiment(config)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
