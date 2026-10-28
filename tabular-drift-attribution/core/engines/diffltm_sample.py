"""Sample one DiffLTM row from a conditional-sampling schema.

Each request row is a ``Condition.dumps`` string: bounds, equality, ranges,
weights. ``free`` is the target block and must stay out of those fields.
cfg encodes the schema with ``channels()``. SMC, Harpoon, and rejection use
the same ``Condition`` as their predicate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def _load(text):
    from diffltm.generation.conditioning import Condition

    condition = Condition.loads(text)
    if condition.predicate is not None:
        raise ValueError("this update does not use a numerical predicate")
    if condition.weights or condition.bounds is not None:
        raise ValueError("this update does not set a linear constraint")
    return condition


def _check(condition, free, columns, schema):
    used = set(condition.equality) | set(condition.ranges) | set(condition.weights)
    overlap = used & set(free)
    if overlap:
        raise ValueError(f"free columns are conditioned: {sorted(overlap)}")
    missing = [name for name in list(free) + list(used) if name not in schema.columns]
    if missing:
        raise ValueError(f"checkpoint schema has no column {missing[0]}")
    if columns is not None:
        unknown = [name for name in columns["columns"] if name not in schema.columns]
        if unknown:
            raise ValueError(f"checkpoint schema has no column {unknown[0]}")
        bad_ranges = [name for name in condition.ranges if name not in schema.numerical]
        if bad_ranges:
            raise ValueError(f"range column {bad_ranges[0]} is not numerical")
    condition.validate(schema)


def _draw(model, method, condition, seed, guidance, pin, harpoon_eta):
    if method == "cfg":
        from diffltm.generation.conditioning import channels
        from diffltm.generation.sampling import euler

        encoded, mask = channels(model, condition, 1)
        if encoded.shape[-1] != model.condition_dim or mask.shape != encoded.shape:
            raise RuntimeError("conditional channel schema does not match the checkpoint")
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
    free = list(payload["free"])
    columns = payload.get("columns")
    frames = []
    for index, text in enumerate(payload["conditions"]):
        condition = _load(text)
        _check(condition, free, columns, model.preprocessor.schema)
        row_seed = args.seed + index * 1_000_003
        frame = _draw(
            model, args.method, condition, row_seed, args.guidance, args.pin, args.harpoon_eta,
        )
        if len(frame) != 1:
            raise RuntimeError(f"{args.method} returned {len(frame)} rows for one condition")
        frames.append(frame.iloc[[0]])
    out = pd.concat(frames, ignore_index=True)
    missing = [name for name in free if name not in out.columns]
    if missing:
        raise KeyError(f"sample is missing {missing[0]}")
    out[free].to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
