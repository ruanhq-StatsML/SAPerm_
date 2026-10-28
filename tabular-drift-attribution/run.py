"""CLI. Fit the generator on batch 0, then attribute each later batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from core.attribution import rank_blocks, rank_features
from core.digestion import digest
from core.generator import CounterfactualGenerator
from core.iterative import block_update, lag_column_names
from core.quality_gate import QualityGate
from data.loaders import (
    load_home_credit,
    load_ieee_cis,
    load_movielens,
    load_synthetic_panel,
)


LOADERS = {
    "movielens": load_movielens,
    "synthetic": load_synthetic_panel,
    "ieee_cis": load_ieee_cis,
    "home_credit": load_home_credit,
}


def condition_columns(n_features: int, target_cols: list[int]) -> list[int]:
    """Columns that stay observed while `target_cols` is regenerated."""
    target = set(target_cols)
    return [c for c in range(n_features) if c not in target]


def run(cfg):
    loader = LOADERS[cfg["dataset"]]
    data = loader(**cfg.get("loader_args", {}))
    batches = data["X_batched"]
    blocks = data["blocks"]
    gt = data["gt"]
    col_names = data["col_names"]
    n_features = data["F"]

    gate = QualityGate().fit(batches[0])
    digested = digest(
        batches[0], col_names, max_cardinality=int(cfg.get("max_cardinality", 32)),
    )
    generator = CounterfactualGenerator(
        backend=cfg["generator"],
        seed=cfg["seed"],
        **cfg.get("generator_args", {}),
    ).fit(batches[0], col_names, discrete_cols=digested["discrete"])
    n_sweeps = int(cfg.get("n_sweeps", 1))
    methods = list(generator.methods) if generator.methods else [None]
    yest_names = lag_column_names(col_names)

    outputs = []
    for t in range(1, len(batches)):
        yesterday = batches[t - 1]
        current = batches[t]
        for method in methods:
            counterfactual = _counterfactual(
                generator, current, yesterday, blocks, col_names,
                yest_names, n_features, n_sweeps, method,
            )
            is_valid, diag = gate.check(counterfactual)
            row = {"t": t}
            if method is not None:
                row["method"] = method
            if int(is_valid.sum()) < 20:
                outputs.append({**row, "skip": "too few valid", **diag})
                continue
            block_ranked, block_prop = rank_blocks(
                current[is_valid], counterfactual[is_valid], blocks, metric=cfg["metric"],
            )
            feature_ranked = rank_features(
                current[is_valid], counterfactual[is_valid], col_names,
            )
            feature_total = sum(feature_ranked.values()) + 1e-8
            feature_contribution = {
                name: value / feature_total
                for name, value in list(feature_ranked.items())[:10]
            }
            gt_blocks = gt.get(t, set())
            top_block = next(iter(block_ranked))
            hit = top_block in gt_blocks if gt_blocks else None
            contribution = {k: round(float(v), 6) for k, v in block_prop.items()}
            row.update({
                "metric": cfg["metric"],
                "block_ranked": {k: round(float(v), 6) for k, v in block_ranked.items()},
                "block_prop": contribution,
                "contribution": contribution,
                "feature_ranked": {
                    k: round(float(v), 6) for k, v in list(feature_ranked.items())[:10]
                },
                "feature_contribution": {
                    k: round(float(v), 6) for k, v in feature_contribution.items()
                },
                "top": top_block,
                "gt_blocks": sorted(gt_blocks),
                "top_hit": hit,
                "n_valid": int(is_valid.sum()),
                "valid_ratio": round(diag["valid_ratio"], 3),
            })
            outputs.append(row)
    return outputs


def _counterfactual(
    generator, current, yesterday, blocks, col_names, yest_names,
    n_features, n_sweeps, method,
):
    def draw(state, yest, cols):
        others = condition_columns(n_features, cols)
        return generator.generate(
            X_cond=state[:, others],
            X_target_ref=state[:, cols],
            n=len(state),
            col_names_cond=[col_names[c] for c in others],
            col_names_target=[col_names[c] for c in cols],
            X_yesterday=yest,
            col_names_yesterday=yest_names,
            method=method,
        )

    return block_update(draw, current, yesterday, blocks, n_sweeps)


def print_summary(results, blocks):
    names = list(blocks) if blocks else []
    if not names:
        for row in results:
            if "block_prop" in row:
                names = list(row["block_prop"])
                break
    methods = []
    for row in results:
        label = row.get("method")
        if label not in methods:
            methods.append(label)
    for label in methods:
        subset = [row for row in results if row.get("method") == label]
        title = f" method={label}" if label is not None else ""
        print("\n" + "=" * 80)
        print(f"{'t':>3} | " + " | ".join(f"{b:>12s}" for b in names) + f" | top         hit{title}")
        print("-" * 80)
        hits = 0
        scored = 0
        for row in subset:
            if row.get("skip"):
                print(f"{row['t']:>3} | SKIP: {row['skip']}")
                continue
            props = [f"{row['block_prop'][b]:>11.1%}" for b in names]
            hit = row.get("top_hit")
            flag = "Y" if hit else ("N" if hit is False else "-")
            if hit is not None:
                scored += 1
                hits += int(hit)
            print(f"{row['t']:>3} | " + " | ".join(props) + f" | {row['top']:<12s}{flag}")
        print("-" * 80)
        if scored:
            print(f"Top-1 accuracy: {hits}/{scored} = {hits / scored:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synthetic.yaml")
    parser.add_argument("--out", default="attribution.json")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    results = run(cfg)
    print_summary(results, cfg.get("_blocks", {}))
    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved -> {args.out}")
