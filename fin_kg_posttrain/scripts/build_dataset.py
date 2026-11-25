"""python -m fin_kg_posttrain.scripts.build_dataset --edges ... --nodes ... --out-dir ..."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fin_kg_posttrain.pipeline import build


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--nodes", type=Path, required=True)
    p.add_argument("--edges", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-sft", type=int, default=200)
    p.add_argument("--n-dpo", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    print(
        json.dumps(
            build(args.nodes, args.edges, args.out_dir, args.n_sft, args.n_dpo, args.seed),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
