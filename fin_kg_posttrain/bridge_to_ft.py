"""
Bridge: real-world KG representation → SAPerm fine-tuning / embedding pipeline.

Plugs into:
  fine_tuning_data_wrapper_bert / fine_tuning_data_wrapper_gpt  (FT_data_extraction_utils)
  → Resulted_data/X_emb_*.csv
  → pipeline_example.py / benchmark_all_method  (distribution-shift tests)

Does NOT invent synthetic ownership graphs. Knowledge comes from dataset.zip labels.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fin_kg_posttrain.builders.sharegpt import dump_jsonl
from fin_kg_posttrain.realworld.extract_kg import export_graph_csv, extract_bundle
from fin_kg_posttrain.realworld.loaders import DEFAULT_ZIP, list_corpora, load_corpus
from fin_kg_posttrain.realworld.pack import pack_for_ft, pack_sharegpt_rows


REPO_ROOT = Path(__file__).resolve().parents[1]


def rows_to_csv(rows: List[Dict[str, str]], path: Path, fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        keys = []
        seen = set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fieldnames = keys
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_representation(
    corpus_name: str,
    out_dir: Path,
    zip_path: Path = DEFAULT_ZIP,
    max_fact_rows: Optional[int] = 5000,
    n_sft: int = 200,
    n_dpo: int = 40,
    include_raw_mix: float = 0.25,
    seed: int = 42,
) -> Dict[str, Any]:
    corpus = load_corpus(corpus_name, zpath=zip_path)
    bundle = extract_bundle(corpus, max_fact_rows=max_fact_rows)
    ft_rows = pack_for_ft(bundle, include_raw_mix=include_raw_mix, seed=seed)
    sft_rows, dpo_rows = pack_sharegpt_rows(bundle, n_sft=n_sft, n_dpo=n_dpo, seed=seed)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ft_csv = out_dir / f"ft_text_{corpus_name}.csv"
    rows_to_csv(ft_rows, ft_csv)

    nodes_csv = out_dir / f"nodes_{corpus_name}.csv"
    edges_csv = out_dir / f"edges_{corpus_name}.csv"
    export_graph_csv(bundle, nodes_csv, edges_csv)

    sft_path = out_dir / f"sharegpt_sft_{corpus_name}.jsonl"
    dpo_path = out_dir / f"sharegpt_dpo_{corpus_name}.jsonl"
    dump_jsonl(sft_rows, sft_path)
    dump_jsonl(dpo_rows, dpo_path)

    summary = {
        "corpus": corpus_name,
        "n_source_rows": len(corpus.rows),
        "n_ft_rows": len(ft_rows),
        "n_graph_nodes": len(bundle.graph.nodes),
        "n_graph_edges": len(bundle.graph.edges),
        "n_sharegpt_sft": len(sft_rows),
        "n_sharegpt_dpo": len(dpo_rows),
        "ft_csv": str(ft_csv),
        "nodes_csv": str(nodes_csv),
        "edges_csv": str(edges_csv),
        "sharegpt_sft": str(sft_path),
        "sharegpt_dpo": str(dpo_path),
        "label_cols": corpus.label_cols,
        "meta": bundle.meta,
        "next_step": (
            "Pass ft_csv into fine_tuning_data_wrapper_bert/gpt "
            "(text_col='text', column_to_join=label_cols), "
            "then feed Resulted_data embeddings into pipeline_example.py"
        ),
    }
    (out_dir / f"summary_{corpus_name}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def run_ft_wrappers_optional(
    ft_csv: Path,
    label_cols: List[str],
    backend: str = "bert",
    output_prefix: str = "Resulted_data/X_emb_kgrep",
) -> Dict[str, str]:
    """Optional: call existing FT wrappers if transformers/peft/pandas are installed."""
    try:
        import pandas as pd
        from FT_data_extraction_utils import (
            fine_tuning_data_wrapper_bert,
            fine_tuning_data_wrapper_gpt,
        )
    except Exception as exc:  # noqa: BLE001
        return {"skipped": f"FT wrappers unavailable ({exc})"}

    df = pd.read_csv(ft_csv)
    if "text" not in df.columns:
        raise ValueError(f"{ft_csv} missing text column")
    join = [c for c in label_cols if c in df.columns]
    out: Dict[str, str] = {}
    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)
    if backend in {"bert", "both"}:
        emb = f"{output_prefix}_bertFineTuning.csv"
        lora = f"{output_prefix}_bertFineTuningLora.csv"
        fine_tuning_data_wrapper_bert(
            df,
            text_col="text",
            output_csv_emb=emb,
            output_csv_lora=lora,
            column_to_join=join or None,
        )
        out["bert_emb"] = emb
        out["bert_lora"] = lora
    if backend in {"gpt", "both"}:
        emb = f"{output_prefix}_gptFineTuning.csv"
        lora = f"{output_prefix}_gptFineTuningLora.csv"
        fine_tuning_data_wrapper_gpt(
            df,
            text_col="text",
            output_csv_emb=emb,
            output_csv_lora=lora,
            column_to_join=join or None,
        )
        out["gpt_emb"] = emb
        out["gpt_lora"] = lora
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Real-world KG representation → FT bridge")
    p.add_argument("--corpus", default="equity_evaluation", choices=list_corpora())
    p.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "fin_kg_posttrain" / "data" / "realworld")
    p.add_argument("--max-fact-rows", type=int, default=5000)
    p.add_argument("--n-sft", type=int, default=200)
    p.add_argument("--n-dpo", type=int, default=40)
    p.add_argument("--raw-mix", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--run-ft",
        action="store_true",
        help="Also call fine_tuning_data_wrapper_* (needs GPU + deps)",
    )
    p.add_argument("--ft-backend", default="bert", choices=["bert", "gpt", "both"])
    args = p.parse_args()

    summary = build_representation(
        corpus_name=args.corpus,
        out_dir=args.out_dir,
        zip_path=args.zip,
        max_fact_rows=args.max_fact_rows,
        n_sft=args.n_sft,
        n_dpo=args.n_dpo,
        include_raw_mix=args.raw_mix,
        seed=args.seed,
    )
    if args.run_ft:
        summary["ft_run"] = run_ft_wrappers_optional(
            Path(summary["ft_csv"]),
            summary["label_cols"],
            backend=args.ft_backend,
            output_prefix=str(REPO_ROOT / "Resulted_data" / f"X_emb_kgrep_{args.corpus}"),
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
