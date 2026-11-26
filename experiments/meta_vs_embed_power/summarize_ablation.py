"""Per-feature ablation summary for the gender dimension scan (MiniLM, RFPerm upper tail).

For each column c of [M, S] (spaCy columns prefixed "spacy:"):
* add:  power([Z, c]) - power(without_meta)        (paired over the same windows)
* loo:  power([Z, M, S] \\ c) - power([Z, M, S])     (negative = c contributes)
Paired SE = sd(reject_a - reject_b) / sqrt(B). Also: spearman(c, y) on the 1000-row frame, single-column
AUC for W (max(AUC, 1-AUC), mean over windows b=0..9), feature group.

    python3 -m experiments.meta_vs_embed_power.summarize_ablation            # writes ablation_gender.{md,json}
    python3 -m experiments.meta_vs_embed_power.summarize_ablation --select   # prints top/bottom-5 add: kinds
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from experiments.meta_vs_embed_power import run_dim_scan as D
from experiments.meta_vs_embed_power.power import make_windows

SCEN = D.SCENARIOS
KS = ("24", "full")
SPLITS = ("RF_random", "RF_ordered")
SELECT_SCEN = ("sentiment_meta", "syntax_meta")

GROUPS = {
    "VADER": {"vader_neg", "vader_neu", "vader_pos", "vader_compound", "polar_gap", "sentiment_std"},
    "长度": {"char_len", "word_count", "token_count", "sent_count", "avg_token_len", "avg_sent_len"},
    "词汇多样性": {"type_token_ratio", "hapax_ratio", "lemma_ttr"},
}


def group(col: str) -> str:
    base = col.removeprefix("spacy:")
    for g, names in GROUPS.items():
        if base in names:
            return g
    if base.startswith("pos_") or base in {"noun_ratio", "verb_ratio", "verb_noun_ratio"}:
        return "POS 比例"
    return "其他"


def load_rows() -> dict:
    by = defaultdict(dict)
    for path in sorted(D.OUT.glob("dim_scan_gender*.jsonl")):
        for line in path.open():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("emb", D.BASE_EMB) != D.BASE_EMB or r["tests"] != "rf" or r["mtry"] is not None:
                continue
            if r["kind"] in ("without", "all") or r["kind"].startswith(D.ABL_PREFIXES):
                by[(r["kind"], r["k"], r["scenario"])][r["b"]] = r
    return by


def paired(by: dict, a: tuple, b: tuple, split: str):
    ra, rb = by.get(a, {}), by.get(b, {})
    common = sorted(set(ra) & set(rb))
    if not common:
        return None
    xa = np.array([ra[i][split]["p_value"] < 0.05 for i in common], dtype=float)
    xb = np.array([rb[i][split]["p_value"] < 0.05 for i in common], dtype=float)
    diff = xa - xb
    return {"power": float(xa.mean()), "delta": float(diff.mean()),
            "se": float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else float("nan"), "B": len(common)}


def column_diag() -> dict:
    d = D.load_data()
    cols = D.ms_columns()
    MS, y = d["MS"], d["y"]
    rho = {c: (0.0 if MS[:, j].std() == 0 else float(spearmanr(MS[:, j], y).correlation)) for j, c in enumerate(cols)}
    aucw = {}
    for sc in SCEN:
        acc = np.zeros(len(cols))
        for b in range(10):
            rng = np.random.default_rng(D.rep_seed(sc, b))
            _, _, _, _, _, _, Ae, An = make_windows(sc, D.z_k("24"), MS, y, d["names"], D.N_DOC, rng)
            W = np.r_[np.zeros(len(Ae)), np.ones(len(An))]
            X = np.vstack([Ae, An])
            for j in range(len(cols)):
                a = 0.5 if X[:, j].std() == 0 else roc_auc_score(W, X[:, j])
                acc[j] += max(a, 1 - a)
        aucw[sc] = dict(zip(cols, (acc / 10).tolist()))
    return {"rho": rho, "auc_w": aucw}


def build_summary() -> dict:
    by = load_rows()
    diag = column_diag()
    out = {"diag": diag, "cells": {}}
    for sc in SCEN:
        for k in KS:
            for split in SPLITS:
                rows = []
                ref_all = paired(by, ("all", k, sc), ("without", k, sc), split)
                for c in D.ms_columns():
                    add = paired(by, (f"add:{c}", k, sc), ("without", k, sc), split)
                    loo = paired(by, (f"loo:{c}", k, sc), ("all", k, sc), split)
                    if add is None and loo is None:
                        continue
                    rows.append({"col": c, "group": group(c), "rho_y": diag["rho"][c],
                                 "auc_w": diag["auc_w"][sc][c], "add": add, "loo": loo})
                out["cells"][f"{sc}|{k}|{split}"] = {"all_vs_without": ref_all, "rows": rows}
    return out


def fmt(x, key="delta"):
    return "—" if not x else f"{x[key]:+.3f}"


def md(summary: dict) -> str:
    parts = []
    for sc in SCEN:
        for k in KS:
            cr, co = summary["cells"].get(f"{sc}|{k}|RF_random"), summary["cells"].get(f"{sc}|{k}|RF_ordered")
            if not cr or not cr["rows"]:
                continue
            ordd = {r["col"]: r for r in co["rows"]} if co else {}
            ref = cr["all_vs_without"]
            rows = sorted(cr["rows"], key=lambda r: -(r["add"]["delta"] if r["add"] else -9))
            head = (f"#### {sc} · k={k}（Z+M+S 相对 without：{fmt(ref)}，B={ref['B'] if ref else '—'}）\n\n"
                    "| 特征 | 组 | ρ(y) | AUC_W | Z+列 功效 | Δadd random (SE) | Δadd ordered | Δloo random (SE) | Δloo ordered |\n"
                    "|---|---|--:|--:|--:|--:|--:|--:|--:|")
            lines = [head]
            for r in rows:
                o = ordd.get(r["col"], {})
                a, l_ = r["add"], r["loo"]
                lines.append(
                    f"| {r['col']} | {r['group']} | {r['rho_y']:+.2f} | {r['auc_w']:.2f} | "
                    f"{a['power']:.3f} | {a['delta']:+.3f} ({a['se']:.3f}) | {fmt(o.get('add'))} | "
                    + (f"{l_['delta']:+.3f} ({l_['se']:.3f})" if l_ else "—") + f" | {fmt(o.get('loo'))} |"
                    if a else f"| {r['col']} | {r['group']} | {r['rho_y']:+.2f} | {r['auc_w']:.2f} | — | — | — | — | — |")
            parts.append("\n".join(lines))
            g = defaultdict(list)
            for r in cr["rows"]:
                if r["add"]:
                    g[r["group"]].append(r)
            gl = ["| 组 | 列数 | 平均 Δadd | 最大 Δadd（列） | Δadd > 2SE 列数 | Δadd < −2SE 列数 | 平均 Δloo |",
                  "|---|--:|--:|---|--:|--:|--:|"]
            for name, rs in sorted(g.items(), key=lambda kv: -np.mean([r["add"]["delta"] for r in kv[1]])):
                best = max(rs, key=lambda r: r["add"]["delta"])
                loo = [r["loo"]["delta"] for r in rs if r["loo"]]
                gl.append(f"| {name} | {len(rs)} | {np.mean([r['add']['delta'] for r in rs]):+.3f} | "
                          f"{best['add']['delta']:+.3f}（{best['col']}） | "
                          f"{sum(r['add']['delta'] > 2 * r['add']['se'] for r in rs)} | "
                          f"{sum(r['add']['delta'] < -2 * r['add']['se'] for r in rs)} | "
                          + (f"{np.mean(loo):+.3f}" if loo else "—") + " |")
            parts.append(f"按组汇总（{sc}, k={k}, random）\n\n" + "\n".join(gl))
    return "\n\n".join(parts) + "\n"


def select(summary: dict, n: int = 5) -> list[str]:
    score = defaultdict(list)
    for sc in SELECT_SCEN:
        for r in summary["cells"].get(f"{sc}|24|RF_random", {}).get("rows", []):
            if r["add"]:
                score[r["col"]].append(r["add"]["delta"])
    ranked = sorted((np.mean(v), c) for c, v in score.items() if len(v) == len(SELECT_SCEN))
    picks = [c for _, c in ranked[-n:][::-1]] + [c for _, c in ranked[:n]]
    return [f"add:{c}" for c in picks]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--select", action="store_true")
    a = p.parse_args()
    s = build_summary()
    if a.select:
        print(" ".join(select(s)))
        return
    (D.OUT / "ablation_gender.json").write_text(json.dumps(s, indent=1))
    text = md(s)
    (D.OUT / "ablation_gender.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
