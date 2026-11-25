#!/usr/bin/env python3
"""ModaSense 三模态专题：CFPerm 模态块 → 驱动子集 → 业务映射.

  python case_modasense_business.py
  python case_modasense_business.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from datasets.modasense_emotion import digest as digest_mm
from extract_mm_features import schema_mm
from perm_tests import cfperm_loco_pvalue
from subset_localize import carrier_rank_from_windows, select_subset

OUT = Path(__file__).resolve().parent / "results"

# feature → (业务语义, 动作) — 只保留会进 subset 的命名维
BUSINESS = {
    "vader_compound": ("文本情感极性", "情绪路由 / 漂移告警"),
    "vader_pos": ("正向质量", "正向占比监控"),
    "vader_neu": ("中性/含糊", "暧昧语气识别"),
    "vader_neg": ("负向质量", "负面升温检测"),
    "vader_polar_gap": ("正负冲突", "矛盾/讽刺候选"),
    "img_brightness": ("画面明暗", "视觉氛围抽检"),
    "img_contrast": ("对比度", "视觉冲击 QA"),
    "img_warm_ratio": ("暖色占比", "色温策略"),
    "img_sat_mean": ("饱和度", "色彩强度"),
    "img_edge_density": ("边缘密度", "画面复杂度"),
    "aud_rms_energy": ("响度", "音量/激动度"),
    "aud_spectral_centroid": ("频谱重心", "音色明暗"),
    "aud_silence_ratio": ("静音比", "停顿/犹豫"),
    "aud_zcr": ("过零率", "嘈杂度"),
    "aud_lowband_ratio": ("低频比", "低沉/压迫感"),
    "aud_peak_to_rms": ("峰值比", "突发尖峰"),
}

ROUTE = {
    "text_sentiment_shift": "vader_compound",
    "image_brightness_shift": "img_brightness",
    "audio_energy_shift": "aud_rms_energy",
}


def _windows(M, names, scenario, n_doc, rng):
    if scenario == "null_shift":
        idx = rng.permutation(len(M))
        return idx[:n_doc], idx[n_doc : 2 * n_doc]
    score = M[:, names.index(ROUTE[scenario])]
    order = np.argsort(score)
    lo, hi = order[: max(2 * n_doc, n_doc)], order[-max(2 * n_doc, n_doc) :]
    return (
        rng.choice(lo, n_doc, replace=len(lo) < n_doc),
        rng.choice(hi, n_doc, replace=len(hi) < n_doc),
    )


def analyze(scenario, M, y, names, *, n_doc, n_perm, seed, n_estimators) -> dict[str, Any]:
    a, b = _windows(M, names, scenario, n_doc, np.random.default_rng(seed))
    Xe, Xn = M[a], M[b]
    X = np.vstack([Xe, Xn])
    Y = np.concatenate([y[a], y[b]])
    W = np.concatenate([np.zeros(len(a), dtype=int), np.ones(len(b), dtype=int)])

    blocks = []
    for bname, spec in schema_mm(with_audio=True)["blocks"].items():
        r = cfperm_loco_pvalue(
            X, Y, W,
            test_indices=np.array(spec["indices"], dtype=int),
            n_perm=n_perm, seed=seed + abs(hash(bname)) % 997,
            n_estimators=n_estimators,
        )
        blocks.append({
            "block": bname,
            "importance": float(r["statistic"]),
            "p_value": float(r["p_value"]),
            "reject": bool(r["reject"]),
            "indices": spec["indices"],
        })
    blocks.sort(key=lambda d: (-d["importance"], d["p_value"]))
    top = blocks[0]

    carriers = carrier_rank_from_windows(
        Xe, Xn, np.array(top["indices"], dtype=int), names
    )
    subset = select_subset(
        [{**r, "p_value": 1.0, "reject": False} for r in carriers],
        top_k=5, cum_share=0.85,
    )
    biz = [
        {
            "feature": n,
            "meaning": BUSINESS.get(n, ("其它 meta", "人工释义"))[0],
            "action": BUSINESS.get(n, ("其它 meta", "人工释义"))[1],
        }
        for n in subset["names"]
    ]
    value = {
        "text_meta": f"文本通道告警；驱动维 {subset['names']} → 情绪 KPI，不牵动视听产线。",
        "image_meta": f"视觉通道抽检；驱动维 {subset['names']} → 明暗/色温 QA。",
        "audio_meta": f"音频通道质检；驱动维 {subset['names']} → 响度/音色/停顿工单。",
    }.get(top["block"], "")

    return {
        "scenario": scenario,
        "top_block": top["block"],
        "blocks": blocks,
        "drivers": subset["names"],
        "business_map": biz,
        "value": value,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-max", type=int, default=500)
    p.add_argument("--n-doc", type=int, default=80)
    p.add_argument("--n-perm", type=int, default=60)
    p.add_argument("--n-estimators", type=int, default=35)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    if args.smoke:
        args.n_max, args.n_doc, args.n_perm, args.n_estimators = 120, 40, 30, 25

    d = digest_mm(args.n_max, with_audio=True)
    print(f"[case] M={d.M.shape}", file=sys.stderr)

    rows = []
    for i, sc in enumerate(ROUTE):
        row = analyze(
            sc, d.M, d.y, d.meta_names,
            n_doc=args.n_doc, n_perm=args.n_perm,
            seed=args.seed + 17 * i, n_estimators=args.n_estimators,
        )
        rows.append(row)
        print(f"  {sc}: {row['top_block']} → {row['drivers']}", file=sys.stderr)

    case = {"dataset": "ModaSense/emotion", "scenarios": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "case_modasense_business.json").write_text(
        json.dumps(case, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md = ["# ModaSense 三模态：子集 × 业务逻辑", ""]
    for r in rows:
        md += [
            f"## {r['scenario']}",
            f"- Top block: `{r['top_block']}`",
            f"- Drivers: {r['drivers']}",
            f"- Value: {r['value']}",
            "",
            "| feature | 语义 | 动作 |",
            "| --- | --- | --- |",
        ]
        for m in r["business_map"]:
            md.append(f"| `{m['feature']}` | {m['meaning']} | {m['action']} |")
        md.append("")
    text = "\n".join(md)
    (OUT / "case_modasense_business.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
