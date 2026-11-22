#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
TEX = R / "interaction_writeup_v2.tex"
CROSS = R / "power_cross_modality_full.json"

def main() -> int:
    if not CROSS.exists():
        print("cross-modality json missing")
        return 0
    data = json.loads(CROSS.read_text())
    blocks = ["embed_pca", "meta", "vader_sent", "spacy_syntax", "spacy_style"]
    body = []
    for t in data["summary"]["table"]:
        bp = t["cf_block_power"]
        sc = t["scenario"].replace("_", r"\_")
        cells = " & ".join(f"{bp.get(b, float('nan')):.2f}" for b in blocks)
        body.append(f"{sc} & {cells} \\\\")
    table = "\n".join([
        r"\begin{center}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Scenario & embed\_pca & meta & vader\_sent & spacy\_syntax & spacy\_style \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        "",
    ])
    tex = TEX.read_text(encoding="utf-8")
    marker = r"\section{One-line takeaway}"
    if "Cross-modality CFPerm powers (filled)" not in tex:
        insert = r"\subsection{Cross-modality CFPerm powers (filled)}" + "\n" + table + "\n" + marker
        TEX.write_text(tex.replace(marker, insert, 1), encoding="utf-8")
        print("updated", TEX)
    else:
        print("already filled")
    md = ["# Cross-modality CFPerm block powers", "",
          "| scenario | " + " | ".join(blocks) + " |",
          "| --- | " + " | ".join(["---:"] * len(blocks)) + " |"]
    for t in data["summary"]["table"]:
        bp = t["cf_block_power"]
        md.append("| " + t["scenario"] + " | " + " | ".join(f"{bp.get(b, float('nan')):.2f}" for b in blocks) + " |")
    (R / "power_cross_modality_blocks.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
