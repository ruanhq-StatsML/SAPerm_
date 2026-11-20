"""
Demo/solver CSV → ShareGPT（单测用）。

正式训练数据请走 real-world 桥：
  python fine_tuning_kg_representation.py --corpus equity_evaluation
  → bridge_to_ft.build_representation
  → ft_text_*.csv → fine_tuning_data_wrapper_* → embedding CSV

本模块保留给：已有 nodes/edges CSV（含 realworld 导出的图）再灌 ShareGPT。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from fin_kg_posttrain.builders.sharegpt import DATASET_INFO_SNIPPET, dump_jsonl
from fin_kg_posttrain.graph.models import FinGraph
from fin_kg_posttrain.tasks.sample import iter_task_samples


def build(
    nodes_csv: Path,
    edges_csv: Path,
    out_dir: Path,
    n_sft: int = 200,
    n_dpo: int = 40,
    seed: int = 42,
) -> dict:
    g = FinGraph.from_csv(nodes_csv, edges_csv)
    sft_rows: List[dict] = []
    dpo_rows: List[dict] = []
    kind_counts: dict[str, int] = {}

    for sample in iter_task_samples(g, n_sft=n_sft, n_dpo=n_dpo, seed=seed):
        kind_counts[sample.kind] = kind_counts.get(sample.kind, 0) + 1
        if sample.sft_row is not None:
            sft_rows.append(sample.sft_row)
        if sample.dpo_row is not None:
            dpo_rows.append(sample.dpo_row)

    out_dir.mkdir(parents=True, exist_ok=True)
    sft_path = out_dir / "fin_kg_sft.jsonl"
    dpo_path = out_dir / "fin_kg_dpo.jsonl"
    dump_jsonl(sft_rows, sft_path)
    dump_jsonl(dpo_rows, dpo_path)

    snippet_path = out_dir / "dataset_info.snippet.json"
    snippet_path.write_text(
        json.dumps(DATASET_INFO_SNIPPET, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "nodes": len(g.nodes),
        "edges": len(g.edges),
        "n_sft": len(sft_rows),
        "n_dpo": len(dpo_rows),
        "kind_counts": kind_counts,
        "sft_path": str(sft_path),
        "dpo_path": str(dpo_path),
        "dataset_info_snippet": str(snippet_path),
        "mix_note": "图谱侧样本；另需自备 ~30% general_fin / math_cot 防遗忘混入 LLaMA-Factory dataset 字段",
    }
    (out_dir / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="FinKG → ShareGPT JSONL (skeleton)")
    p.add_argument("--nodes", type=Path, required=True)
    p.add_argument("--edges", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-sft", type=int, default=200)
    p.add_argument("--n-dpo", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    summary = build(args.nodes, args.edges, args.out_dir, args.n_sft, args.n_dpo, args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
