#!/usr/bin/env python3
"""CLI entry — see ``pipeline.py`` for stages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from experiments.meta_vs_embed_power import config as C
from experiments.meta_vs_embed_power.pipeline import OUT, run_anisotropy, run_power


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--B", type=int, default=C.B)
    p.add_argument("--n-perm", type=int, default=C.N_PERM)
    p.add_argument("--n-doc", type=int, default=C.N_DOC)
    p.add_argument("--pca-k", type=int, default=C.PCA_K)
    p.add_argument("--n-estimators", type=int, default=C.RF_N_ESTIMATORS)
    p.add_argument("--n-max", type=int, default=C.ROW_CAP)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--datasets", nargs="*", default=list(C.DATASETS))
    p.add_argument("--scenarios", nargs="*", default=list(C.SCENARIOS))
    p.add_argument("--embed-model", default="MiniLM_L6")
    p.add_argument(
        "--feature-sets",
        nargs="*",
        default=list(C.FEATURE_SETS),
        help="without_meta | with_meta | meta_only",
    )
    p.add_argument("--out-suffix", default="", help="checkpoint tag + output suffix")
    p.add_argument("--skip-anisotropy", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--anisotropy-only", action="store_true")
    args = p.parse_args(argv)

    if args.smoke:
        args.B = 4
        args.n_perm = 20
        args.n_doc = 60
        args.n_estimators = 30
        args.n_max = 400
        args.pca_k = 12
        args.n_jobs = 1
        args.datasets = ["train_bias_finder", "gender"]
        args.scenarios = ["null_shift", "sentiment_meta"]

    OUT.mkdir(parents=True, exist_ok=True)
    aniso: dict = {}
    if not args.skip_anisotropy:
        aniso = run_anisotropy(args.n_max, args.seed, OUT, args.embed_model)
    elif (OUT / "anisotropy_effective_rank.json").is_file():
        aniso = json.loads((OUT / "anisotropy_effective_rank.json").read_text(encoding="utf-8"))

    if args.anisotropy_only:
        print(f"Wrote {OUT / 'anisotropy_effective_rank.json'}", file=sys.stderr)
        return 0

    result = run_power(
        B=args.B,
        n_perm=args.n_perm,
        n_doc=args.n_doc,
        pca_k=args.pca_k,
        n_estimators=args.n_estimators,
        n_max=args.n_max,
        seed=args.seed,
        datasets=args.datasets,
        scenarios=args.scenarios,
        feature_sets=args.feature_sets,
        n_jobs=args.n_jobs,
        out_dir=OUT,
        embed_model=args.embed_model,
        checkpoint_tag=args.out_suffix or "default",
    )
    result["anisotropy_headline"] = {
        k: {
            "anisotropy_at_max_pc": v[-1]["anisotropy"] if v else None,
            "effective_rank_at_max_pc": v[-1]["effective_rank"] if v else None,
        }
        for k, v in aniso.items()
    }

    tag = "smoke" if args.smoke else "full"
    suf = f"_{args.out_suffix}" if args.out_suffix else ""
    out_json = OUT / f"power_meta_embed_{tag}{suf}.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Wrote {out_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())