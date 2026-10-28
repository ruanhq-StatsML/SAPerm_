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


class DatasetForward:
    """One dataset. Each later batch is one forward pass of one generative model."""

    def __init__(self, loader):
        self.loader = loader

    def forward(
        self,
        model,
        seed=0,
        metric="mean_diff",
        loader_args=None,
        generator_args=None,
        n_sweeps=1,
        max_cardinality=32,
    ):
        data = self.loader(**(loader_args or {}))
        batches = data["X_batched"]
        blocks = data["blocks"]
        gt = data["gt"]
        col_names = data["col_names"]
        n_features = data["F"]
        gate = QualityGate().fit(batches[0])
        generator = CounterfactualGenerator(
            backend=model, seed=seed, **(generator_args or {}),
        )
        digested = digest(batches[0], col_names, max_cardinality=max_cardinality)
        generator.fit(batches[0], col_names, discrete_cols=digested["discrete"])
        yest_names = lag_column_names(col_names)
        methods = list(generator.methods) if generator.methods else [None]
        outputs = []
        for t in range(1, len(batches)):
            for method in methods:
                counterfactual = forward_batch(
                    generator, batches[t], batches[t - 1], blocks, col_names,
                    yest_names, n_features, n_sweeps, method,
                )
                outputs.append(_score(
                    t, method, batches[t], counterfactual, blocks, col_names,
                    gt, metric, gate,
                ))
        return outputs


DATASETS = {
    "movielens": DatasetForward(load_movielens),
    "synthetic": DatasetForward(load_synthetic_panel),
    "ieee_cis": DatasetForward(load_ieee_cis),
    "home_credit": DatasetForward(load_home_credit),
}


def condition_columns(n_features: int, target_cols: list[int]) -> list[int]:
    """Columns that stay observed while `target_cols` is regenerated."""
    target = set(target_cols)
    return [c for c in range(n_features) if c not in target]


def run(cfg):
    return DATASETS[cfg["dataset"]].forward(
        cfg["generator"],
        seed=cfg["seed"],
        metric=cfg["metric"],
        loader_args=cfg.get("loader_args"),
        generator_args=cfg.get("generator_args"),
        n_sweeps=int(cfg.get("n_sweeps", 1)),
        max_cardinality=int(cfg.get("max_cardinality", 32)),
    )


def forward_batch(
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


def _score(t, method, current, counterfactual, blocks, col_names, gt, metric, gate):
    is_valid, diag = gate.check(counterfactual)
    row = {"t": t}
    if method is not None:
        row["method"] = method
    if int(is_valid.sum()) < 20:
        return {**row, "skip": "too few valid", **diag}
    block_ranked, contribution = rank_blocks(
        current[is_valid], counterfactual[is_valid], blocks, metric=metric,
    )
    feature_ranked = rank_features(
        current[is_valid], counterfactual[is_valid], col_names,
    )
    feature_total = sum(feature_ranked.values()) + 1e-8
    gt_blocks = gt.get(t, set())
    top = next(iter(block_ranked))
    contribution = {k: round(float(v), 6) for k, v in contribution.items()}
    row.update({
        "metric": metric,
        "block_ranked": {k: round(float(v), 6) for k, v in block_ranked.items()},
        "block_prop": contribution,
        "contribution": contribution,
        "feature_ranked": {
            k: round(float(v), 6) for k, v in list(feature_ranked.items())[:10]
        },
        "feature_contribution": {
            name: value / feature_total
            for name, value in list(feature_ranked.items())[:10]
        },
        "top": top,
        "gt_blocks": sorted(gt_blocks),
        "top_hit": top in gt_blocks if gt_blocks else None,
        "n_valid": int(is_valid.sum()),
        "valid_ratio": round(diag["valid_ratio"], 3),
    })
    return row


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
