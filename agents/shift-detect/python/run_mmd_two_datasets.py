#!/usr/bin/env python3
"""MMD × ρ_PO × power on two datasets (BiasFinder + ModaSense).

Uses ready ``mmd_by_modality``; joins published power / ρ_PO tables.
Writes ``results/mmd_two_datasets.{json,md}``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from datasets.biasfinder_gen import digest as digest_bf
from datasets.modasense_emotion import digest as digest_mm
from extract_mm_features import schema_mm
from mmd_modalities import mmd_by_modality
from power_embed_meta import make_windows

OUT = Path(__file__).resolve().parent / "results"
B_MMD = 50
N_DOC = 80


def _mean_mmd(mods: dict[str, np.ndarray], split_fn, B: int = B_MMD) -> dict[str, float]:
    acc = {k: [] for k in mods}
    for b in range(B):
        a, c = split_fn(np.random.default_rng(1000 + b))
        m = mmd_by_modality(mods, a, c, seed=b)
        for k, v in m.items():
            acc[k].append(v)
    return {k: float(np.mean(vs)) for k, vs in acc.items()}


def biasfinder_block() -> dict:
    d = digest_bf(n_max=2000, pca_k=24, seed=0)
    Z, M, y, texts = d.Z, d.M, d.y, d.texts
    mods = {
        "embed": StandardScaler().fit_transform(Z),
        "meta": StandardScaler().fit_transform(M),
    }
    # also meta sub-blocks as "modalities" of the text tower
    from extract_features import FEATURE_NAMES

    vader_i = [FEATURE_NAMES.index(n) for n in FEATURE_NAMES if n.startswith("vader_")]
    syntax_i = [
        FEATURE_NAMES.index(n)
        for n in (
            "pos_noun", "pos_verb", "pos_adj", "pos_adv", "pos_propn", "pos_pron", "verb_noun_ratio"
        )
    ]
    mods_fine = {
        "embed": mods["embed"],
        "vader_sent": StandardScaler().fit_transform(M[:, vader_i]),
        "spacy_syntax": StandardScaler().fit_transform(M[:, syntax_i]),
    }

    scenarios = ["null_shift", "sentiment_meta", "syntax_meta", "train_vs_heldout"]
    rows = []
    for sc in scenarios:
        def split_fn(rng, scenario=sc):
            Xe_e, ye, Xn_e, yn, Xe_b, Xn_b = make_windows(
                scenario, Z, M, y, texts, N_DOC, rng
            )
            # recover indices by matching first PCA cols — fragile; use make_windows internals
            # Instead re-implement index pick like make_windows
            return _bf_indices(scenario, M, y, N_DOC, rng)

        mmd = _mean_mmd(mods, split_fn)
        mmd_fine = _mean_mmd(mods_fine, split_fn)
        top = max(mmd, key=mmd.get)
        rows.append({
            "scenario": sc,
            "mmd": mmd,
            "mmd_fine": mmd_fine,
            "mmd_top": top,
            "mmd_fine_top": max(mmd_fine, key=mmd_fine.get),
        })
        print(f"[bf] {sc}: {mmd} fine_top={rows[-1]['mmd_fine_top']}", flush=True)

    # join published signatures
    sig = json.loads((OUT / "interaction_signatures.json").read_text())
    by = {s["scenario"]: s for s in sig["signatures"]}
    # map names
    alias = {
        "null_shift": "null_shift",
        "sentiment_meta": "sentiment_meta",
        "syntax_meta": "syntax_meta",
        "train_vs_heldout": "train_vs_heldout",
    }
    for r in rows:
        s = by.get(alias[r["scenario"]])
        if not s:
            continue
        r["RF_embed"] = s["power_embed"]["RFPerm"]
        r["RF_embed_meta"] = s["power_meta"]["RFPerm"]
        r["DR_embed"] = s["power_embed"]["DRPerm"]
        r["DR_embed_meta"] = s["power_meta"]["DRPerm"]
        r["CF_delta"] = s["delta_power"]["CFPerm"]
        r["rho_PO"] = s["rho_PO"]
        r["label"] = s["interaction_label"]
    return {"dataset": "biasfinder_gen", "rows": rows}


def _bf_indices(scenario, M, y, n_doc, rng):
    from extract_features import FEATURE_NAMES

    n = len(y)
    compounds = M[:, FEATURE_NAMES.index("vader_compound")]
    if scenario == "null_shift":
        idx = rng.permutation(n)
        return idx[:n_doc], idx[n_doc : 2 * n_doc]
    if scenario == "sentiment_meta":
        order = np.argsort(compounds)
        neg = order[: max(n_doc * 3, n_doc)]
        mid = order[len(order) // 3 : 2 * len(order) // 3]
        return rng.choice(mid, n_doc, replace=len(mid) < n_doc), rng.choice(
            neg, n_doc, replace=len(neg) < n_doc
        )
    if scenario == "syntax_meta":
        score = M[:, FEATURE_NAMES.index("verb_noun_ratio")]
        order = np.argsort(score)
        lo = order[: max(n_doc * 3, n_doc)]
        hi = order[-max(n_doc * 3, n_doc) :]
        return rng.choice(lo, n_doc, replace=len(lo) < n_doc), rng.choice(
            hi, n_doc, replace=len(hi) < n_doc
        )
    if scenario == "train_vs_heldout":
        half = n // 2
        return rng.choice(np.arange(half), n_doc, replace=False), rng.choice(
            np.arange(half, n), n_doc, replace=False
        )
    raise ValueError(scenario)


def modasense_block() -> dict:
    d = digest_mm(500, with_audio=True)
    sch = schema_mm(with_audio=True)
    M, names = d.M, d.meta_names
    mods = {
        "text": StandardScaler().fit_transform(M[:, sch["modality"]["text"]]),
        "image": StandardScaler().fit_transform(M[:, sch["modality"]["image"]]),
        "audio": StandardScaler().fit_transform(M[:, sch["modality"]["audio"]]),
    }
    route = {
        "null_shift": None,
        "text_sentiment_shift": "vader_compound",
        "image_brightness_shift": "img_brightness",
        "audio_energy_shift": "aud_rms_energy",
    }

    def split_fn_factory(key):
        def split_fn(rng):
            n = len(M)
            if key is None:
                idx = rng.permutation(n)
                return idx[:N_DOC], idx[N_DOC : 2 * N_DOC]
            score = M[:, names.index(key)]
            order = np.argsort(score)
            lo = order[: max(2 * N_DOC, N_DOC)]
            hi = order[-max(2 * N_DOC, N_DOC) :]
            return rng.choice(lo, N_DOC, replace=len(lo) < N_DOC), rng.choice(
                hi, N_DOC, replace=len(hi) < N_DOC
            )

        return split_fn

    rows = []
    for sc, key in route.items():
        mmd = _mean_mmd(mods, split_fn_factory(key))
        rows.append({
            "scenario": sc,
            "mmd": mmd,
            "mmd_top": max(mmd, key=mmd.get),
        })
        print(f"[mm] {sc}: {mmd}", flush=True)

    # join full mm power table
    mmj = json.loads((OUT / "power_mm_modasense_full.json").read_text())
    by = {e["scenario"]: e for e in mmj["summary"]["table"]}
    smoke = json.loads((OUT / "mmd_po_power_modasense_smoke.json").read_text())
    smoke_by = {e["scenario"]: e for e in smoke["summary"]["table"]}
    for r in rows:
        if r["scenario"] in by:
            e = by[r["scenario"]]
            r["RF_text"] = e["power"]["text_only"]["RFPerm"]
            r["RF_mm"] = e["power"]["multimodal"]["RFPerm"]
            r["rho_PO"] = e["rho_PO"]
            r["CF_text"] = e["cf_block_power"]["text_meta"]
            r["CF_image"] = e["cf_block_power"]["image_meta"]
        if r["scenario"] in smoke_by:
            r["RF_smoke"] = smoke_by[r["scenario"]]["RF_power"]
            r["rho_smoke"] = smoke_by[r["scenario"]]["rho_PO"]
    return {"dataset": "modasense_emotion", "rows": rows}


def write_md(payload: dict, path: Path) -> None:
    bf = payload["biasfinder"]
    mm = payload["modasense"]
    lines = [
        "# Two-dataset argument: MMD × ρ_PO × power",
        "",
        "General-purpose shell: modality arrays in → MMD / power / ρ_PO out.",
        "Same read on BiasFinder (embed∥meta) and ModaSense (text∥image∥audio).",
        "",
        "## BiasFinder (embed vs meta)",
        "",
        "| scenario | MMD embed | MMD meta | fine-top | RF e→e+m | ρ_PO | label |",
        "| --- | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for r in bf["rows"]:
        lines.append(
            f"| {r['scenario']} | {r['mmd']['embed']:.3g} | {r['mmd']['meta']:.3g} | "
            f"{r.get('mmd_fine_top','')} | "
            f"{r.get('RF_embed', float('nan')):.2f}→{r.get('RF_embed_meta', float('nan')):.2f} | "
            f"{r.get('rho_PO', float('nan')):.2f} | {r.get('label','')} |"
        )
    lines += [
        "",
        "## ModaSense (text / image / audio)",
        "",
        "| scenario | MMD t/i/a | top | RF text→MM | ρ_PO |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for r in mm["rows"]:
        m = r["mmd"]
        rf0 = r.get("RF_text")
        rf1 = r.get("RF_mm")
        if rf0 is None and "RF_smoke" in r:
            rf0, rf1 = r["RF_smoke"].get("text"), r["RF_smoke"].get("all")
        rho = r.get("rho_PO")
        if rho is None and "rho_smoke" in r:
            rho = r["rho_smoke"].get("all_over_text")
        lines.append(
            f"| {r['scenario']} | {m['text']:.3g}/{m['image']:.3g}/{m['audio']:.3g} | "
            f"**{r['mmd_top']}** | "
            f"{float(rf0 if rf0 is not None else float('nan')):.2f}→"
            f"{float(rf1 if rf1 is not None else float('nan')):.2f} | "
            f"{float(rho if rho is not None else float('nan')):.2f} |"
        )
    lines += [
        "",
        "## Insights（两库同一套读法）",
        "",
        "1. **MMD 定通道**：BiasFinder 上 sentiment/syntax 窗 meta（及 vader/syntax 细块）MMD 抬升；"
        "ModaSense 上路由模态 MMD 高一个数量级（text 0.33 / image 0.58 / audio 0.55），null 三模态同阶。",
        "2. **Power 定能不能检**：BiasFinder heldout RF=1（概念残差）；sentiment 小涨。"
        "ModaSense text 窗 RF=1；image/audio 窗 MMD 高但 RF≈0 —— 动了 ≠ 能报警。",
        "3. **ρ_PO 定拼特征有没有吸收**：BiasFinder sentiment/syntax ρ≈0.25/0.19（absorb）；"
        "heldout ρ≈1.03（residual）。ModaSense image ρ=0.62（吸收）；text ρ≈1.95（已检满，拼图不降 PO）。",
        "4. **General-purpose**：两库都是 `{modality: array}` → 同三角对照；换任意多塔模型只需换数组。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    print("=== BiasFinder ===", flush=True)
    bf = biasfinder_block()
    print("=== ModaSense ===", flush=True)
    mm = modasense_block()
    payload = {"biasfinder": bf, "modasense": mm, "B_mmd": B_MMD, "n_doc": N_DOC}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mmd_two_datasets.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_md(payload, OUT / "mmd_two_datasets.md")
    print(OUT / "mmd_two_datasets.md")
    print((OUT / "mmd_two_datasets.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
