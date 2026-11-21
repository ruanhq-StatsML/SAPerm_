"""Run multimodal higher-order PO-risk interaction on MSR-VTT-style batches.

Example
-------
python -m experiments.multimodal_attribution.run_interaction --band top30 --n 500
python -m experiments.multimodal_attribution.run_interaction --band pc30_100 --n 500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.deep_active_sampling.model_registry import default_model_registry
from experiments.deep_active_sampling.ood_vimp import multimodal_block_attribution
from experiments.multimodal_attribution.msrvtt import (
    DEFAULT_BATCH_SIZE,
    build_msrvtt_style_batches,
    load_precomputed_npz_modalities,
)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--band", choices=("top30", "pc30_100"), default="top30")
    p.add_argument("--n", type=int, default=DEFAULT_BATCH_SIZE, help="samples per domain batch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--interaction-shift", type=float, default=1.25)
    p.add_argument(
        "--model-m",
        default="ridge_regressor",
        help="nuisance outcome / tau model key in MODEL_REGISTRY",
    )
    p.add_argument("--out", type=Path, default=None, help="optional JSON dump path")
    p.add_argument("--text-ref", type=Path, default=None)
    p.add_argument("--audio-ref", type=Path, default=None)
    p.add_argument("--video-ref", type=Path, default=None)
    p.add_argument("--text-new", type=Path, default=None)
    p.add_argument("--audio-new", type=Path, default=None)
    p.add_argument("--video-new", type=Path, default=None)
    p.add_argument("--y-ref", type=Path, default=None, help=".npy outcome for ref batch")
    p.add_argument("--y-new", type=Path, default=None, help=".npy outcome for new batch")
    return p.parse_args()


def _maybe_precomputed(args):
    paths = [args.text_ref, args.audio_ref, args.video_ref, args.text_new, args.audio_new, args.video_new]
    if all(p is None for p in paths):
        return None, None
    if any(p is None for p in paths) or args.y_ref is None or args.y_new is None:
        raise SystemExit("provide all six embedding paths plus --y-ref and --y-new")
    precomputed = {
        "ref": load_precomputed_npz_modalities(
            str(args.text_ref), str(args.audio_ref), str(args.video_ref)
        ),
        "new": load_precomputed_npz_modalities(
            str(args.text_new), str(args.audio_new), str(args.video_new)
        ),
    }
    outcomes = {
        "ref": np.load(args.y_ref),
        "new": np.load(args.y_new),
    }
    return precomputed, outcomes


def main(argv=None):
    if argv is None:
        args = _parse_args()
    else:
        import sys

        old = sys.argv
        sys.argv = [old[0], *argv]
        try:
            args = _parse_args()
        finally:
            sys.argv = old
    precomputed, outcomes = _maybe_precomputed(args)
    # For pc30_100 synthetic dims must exceed 100 features per modality.
    dims = (128, 128, 128) if args.band == "top30" else (160, 160, 160)
    batch = build_msrvtt_style_batches(
        n_per_batch=args.n,
        band=args.band,
        seed=args.seed,
        interaction_shift=args.interaction_shift,
        dims=dims,
        precomputed=precomputed,
        outcomes=outcomes,
    )
    registry = default_model_registry(ntree=40, ridge_alpha=0.5)
    result = multimodal_block_attribution(
        batch["X"],
        batch["Y"],
        batch["T"],
        batch["modality_indices"],
        model_registry=registry,
        model_m=args.model_m,
        model_tau=args.model_m,
        model_nu=args.model_m,
        seed=args.seed,
        test_size=0.5,
    )
    summary = {
        "band": batch["band"],
        "n_per_batch": batch["n_per_batch"],
        "p": int(batch["X"].shape[1]),
        "modality_indices": batch["modality_indices"],
        "pairwise": {
            k: {kk: float(vv) for kk, vv in v.items() if isinstance(vv, (float, int, np.floating))}
            for k, v in result["pairwise"].items()
        },
        "third_order": None
        if result["third_order"] is None
        else {
            k: {kk: float(vv) for kk, vv in v.items() if isinstance(vv, (float, int, np.floating))}
            for k, v in result["third_order"].items()
        },
    }
    print(json.dumps(summary, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
