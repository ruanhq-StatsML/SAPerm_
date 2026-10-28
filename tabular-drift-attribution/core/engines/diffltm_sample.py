"""Draw one DiffLTM row per predicate.

The predicate is built by DiffLTMEngine: today's other features and yesterday's
full features. `method` is smc, harpoon, rejection, or cfg. One condition is
not reused across rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def _condition(spec):
    from diffltm.generation.conditioning import Condition

    equality = spec.get("equality") or {}
    ranges = {
        name: (float(lo), float(hi))
        for name, (lo, hi) in (spec.get("ranges") or {}).items()
    }
    return Condition(equality=equality, ranges=ranges)


def _draw(model, method, condition, seed, guidance, pin, harpoon_eta):
    if method == "cfg":
        from diffltm.generation.sampling import euler

        frame, _ = euler(model, 1, seed, condition, guidance, pin)
    elif method == "smc":
        from diffltm.generation.smc import sample

        frame, _ = sample(model, 1, seed, condition, pin)
    elif method == "harpoon":
        from diffltm.generation.harpoon import sample

        frame, _ = sample(model, 1, seed, condition, harpoon_eta)
    elif method == "rejection":
        from diffltm.generation.rejection import RejectionPool
        from diffltm.generation.sampling import euler

        pool = RejectionPool(lambda size, chunk_seed: euler(model, size, chunk_seed), seed)
        try:
            frame, diagnostics = pool.get(condition, 1)
        finally:
            pool.close()
        if diagnostics.get("budget_exhausted") or len(frame) != 1:
            raise RuntimeError("rejection did not return a row inside the condition")
    else:
        raise ValueError(f"unknown method {method}")
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--method", required=True, choices=("smc", "harpoon", "rejection", "cfg"))
    parser.add_argument("--request", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--harpoon-eta", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pin", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    repo = Path.cwd()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from diffltm.trainer import load_model

    model, _checkpoint = load_model(Path(args.run_dir) / "selected.pt", args.device)
    payload = json.loads(Path(args.request).read_text())
    target = list(payload["target"])
    frames = []
    for index, spec in enumerate(payload["rows"]):
        condition = _condition(spec)
        condition.validate(model.preprocessor.schema)
        row_seed = args.seed + index * 1_000_003
        frame = _draw(
            model, args.method, condition, row_seed, args.guidance, args.pin, args.harpoon_eta,
        )
        if len(frame) != 1:
            raise RuntimeError(f"{args.method} returned {len(frame)} rows for one condition")
        frames.append(frame.iloc[[0]])
    out = pd.concat(frames, ignore_index=True)
    missing = [name for name in target if name not in out.columns]
    if missing:
        raise KeyError(f"sample is missing {missing[0]}")
    out[target].to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
