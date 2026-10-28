"""Forward one dataset through one generative model, one batch at a time."""

from __future__ import annotations

import argparse

from core.generator import GENERATIVE_MODELS
from run import DATASETS, print_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="synthetic", choices=sorted(DATASETS))
    parser.add_argument("--model", default="stub", choices=sorted(GENERATIVE_MODELS))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    results = DATASETS[args.dataset].forward(args.model, seed=args.seed)
    print_summary(results, {})


if __name__ == "__main__":
    main()
