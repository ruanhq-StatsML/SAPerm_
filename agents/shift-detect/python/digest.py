#!/usr/bin/env python3
"""Data digestion entrypoint (thin dispatch → datasets/*).

  python digest.py --dataset modasense_emotion --audio --write
  python digest.py --dataset biasfinder_gen --n-max 2000
"""

from __future__ import annotations

import argparse

from datasets import digest as digest_named


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        required=True,
        choices=["modasense_emotion", "biasfinder_gen"],
    )
    p.add_argument("--n-max", type=int, default=None)
    p.add_argument("--audio", action="store_true", help="ModaSense: include audio meta")
    p.add_argument("--pca-k", type=int, default=24, help="BiasFinder PCA dims")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--write", action="store_true")
    args = p.parse_args(argv)

    if args.dataset == "modasense_emotion":
        from datasets.modasense_emotion import digest, write_digest_report

        d = digest(args.n_max or 500, with_audio=args.audio)
        print(f"[digest] {d.dataset} n={d.n} M={d.M.shape} {d.schema.get('layout')}")
        if args.write:
            print(f"[digest] wrote {write_digest_report(d)}")
    else:
        from datasets.biasfinder_gen import digest

        d = digest(args.n_max or 2000, pca_k=args.pca_k, seed=args.seed)
        print(f"[digest] {d.dataset} n={d.n} Z={None if d.Z is None else d.Z.shape} M={d.M.shape}")
        if args.write:
            from utils import OUT

            path = OUT / f"digest_{d.dataset}_n{d.n}.npz"
            d.save(path)
            print(f"[digest] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
