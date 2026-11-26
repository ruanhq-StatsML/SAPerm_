#!/usr/bin/env python3
"""Amazon VADER-focused and per-column (32 M + 36 S) RFPerm B=150 ablations.

All outputs are stored under ``results/amazon/fullgrid/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from experiments.meta_vs_embed_power import config as C
from experiments.meta_vs_embed_power.amazon_extract import ENCODERS, OUT_DIR, _verified, load_aligned
from experiments.meta_vs_embed_power.data import load_gender
from experiments.meta_vs_embed_power.features import meta_matrix
from experiments.meta_vs_embed_power.pipeline import run_power
from experiments.meta_vs_embed_power.power import make_windows
from experiments.meta_vs_embed_power.run_amazon_fullgrid import (
    CLIP,
    FULLGRID_DIR,
    PREFIX,
    SCENARIO_ORDER,
    SCENARIOS,
    SEED,
    _crossfit_e,
    _extract_rf_from_ckpt,
    k_to_pca_k,
)

VADER6 = ("vader_neg", "vader_neu", "vader_pos", "vader_compound", "polar_gap", "sentiment_std")
VADER4 = ("vader_neg", "vader_neu", "vader_pos", "vader_compound")
VADER_ARMS = (
    "with_vader_all",
    "with_vader4",
    "with_m_no_vader",
    "with_col_vader_neg",
    "with_col_vader_neu",
    "with_col_vader_pos",
    "with_col_vader_compound",
    "with_col_polar_gap",
    "with_col_sentiment_std",
)

GROUP_MAP = {
    **{c: "VADER" for c in VADER6},
    **{c: "长度" for c in ("char_len", "word_count", "token_count", "sent_count", "avg_token_len", "avg_sent_len")},
    **{c: "词汇多样性" for c in ("type_token_ratio", "hapax_ratio", "lemma_ttr", "stopword_ratio", "alpha_ratio")},
    **{
        c: "POS 比例"
        for c in (
            "noun_ratio", "verb_ratio", "verb_noun_ratio",
            "pos_adj_ratio", "pos_adv_ratio", "pos_pron_ratio", "pos_det_ratio",
            "pos_adp_ratio", "pos_propn_ratio", "pos_aux_ratio", "pos_num_ratio",
            "pos_cconj_ratio", "pos_sconj_ratio", "pos_part_ratio", "pos_intj_ratio",
            "pos_punct_ratio",
        )
    },
}


def feature_group(raw_name: str) -> str:
    base = raw_name.removeprefix("spacy:")
    return GROUP_MAP.get(base, "其他")


def _select_cols(Zfeat: np.ndarray, Mfull: np.ndarray, names: list[str], arm: str) -> np.ndarray:
    if arm == "without_meta":
        return Zfeat
    if arm == "with_meta":
        idx = [i for i, n in enumerate(names) if not n.startswith("spacy:")]
    elif arm == "with_spacy":
        idx = [i for i, n in enumerate(names) if n.startswith("spacy:")]
    elif arm == "with_vader_all":
        idx = [i for i, n in enumerate(names) if n in VADER6]
    elif arm == "with_vader4":
        idx = [i for i, n in enumerate(names) if n in VADER4]
    elif arm == "with_m_no_vader":
        idx = [i for i, n in enumerate(names) if not n.startswith("spacy:") and n not in VADER6]
    elif arm.startswith("with_col_"):
        t = arm[len("with_col_"):]
        idx = [names.index(t)]
    else:
        raise ValueError(arm)
    return np.hstack([Zfeat, Mfull[:, idx]])


def _vader_diag_cell(enc: str, k_label: str, spec: str, R: int) -> dict[str, Any]:
    Zfeat, Mfull, y, names = load_aligned(enc, k_to_pca_k(k_label), seed=SEED, spec=spec)
    yi = y.astype(int)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    arms = ("without_meta",) + VADER_ARMS
    y_auc: dict[str, float] = {}
    for arm in arms:
        X = _select_cols(Zfeat, Mfull, names, arm)
        clf = RandomForestClassifier(n_estimators=C.RF_N_ESTIMATORS, random_state=SEED, n_jobs=1)
        p = cross_val_predict(clf, X, yi, cv=cv, method="predict_proba")[:, 1]
        y_auc[arm] = float(roc_auc_score(yi, p))

    windows: dict[str, dict[str, dict[str, float]]] = {}
    for scen in SCENARIOS[spec]:
        acc = {arm: {"auc": [], "var_resid": []} for arm in arms}
        for b in range(R):
            rep_seed = SEED + 10_007 * (zlib.crc32(scen.encode()) % 997) + 17 * b
            Ze, ye, Zn, yn, _, _, Mfe, Mfn = make_windows(
                scen, Zfeat, Mfull, y, names, C.N_DOC, np.random.default_rng(rep_seed)
            )
            Zwin, Mwin = np.vstack([Ze, Zn]), np.vstack([Mfe, Mfn])
            W = np.r_[np.zeros(len(ye), int), np.ones(len(yn), int)]
            for arm in arms:
                Xw = _select_cols(Zwin, Mwin, names, arm)
                e = _crossfit_e(Xw, W, rep_seed + 3)
                ec = np.clip(e, CLIP, 1.0 - CLIP)
                acc[arm]["auc"].append(roc_auc_score(W, e))
                acc[arm]["var_resid"].append(float(np.var(W - ec)))
        base_v = float(np.mean(acc["without_meta"]["var_resid"]))
        windows[scen] = {
            arm: {
                "W_auc": float(np.mean(v["auc"])),
                "var_resid_ratio_vs_Z": float(np.mean(v["var_resid"]) / base_v),
            }
            for arm, v in acc.items()
        }
    return {"encoder": enc, "k": k_label, "spec": spec, "y_auc": y_auc, "windows": windows}


def run_vader_diag(encoders: list[str], R: int, n_jobs: int, k_vals: tuple[str, ...] = ("full",)) -> dict[str, Any]:
    FULLGRID_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [(enc, k, spec) for enc in encoders for k in k_vals for spec in ("gift", "cross")]
    res = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_vader_diag_cell)(enc, k, spec, R) for enc, k, spec in tasks
    )
    sp_y: dict[str, dict[str, float]] = {}
    for spec in ("gift", "cross"):
        df, meta, _, _ = _verified("MiniLM_L6", spec)
        names = list(meta["meta_names"])
        M, y = meta["M"], df["y"].to_numpy()
        sp_y[spec] = {c: float(spearmanr(M[:, names.index(c)], y).correlation) for c in VADER6}
    _, yg, texts_g = load_gender(n_max=1000, seed=SEED)
    Mg, ng = meta_matrix(texts_g, Path(OUT_DIR).parent / "meta_gender_n1000.npz")
    sp_y["gender"] = {c: float(spearmanr(Mg[:, ng.index(c)], yg).correlation) for c in VADER6}

    payload = {"R_windows": R, "spearman_y": sp_y, "cells": {f"{x['encoder']}|{x['k']}|{x['spec']}": x for x in res}}
    out = FULLGRID_DIR / "vader_diagnostics.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}", file=sys.stderr, flush=True)
    return payload


def _run_rf_arms(
    *,
    encoders: list[str],
    k_vals: tuple[str, ...],
    arms: list[str],
    scenarios: list[str] | None,
    tag_prefix: str,
    n_jobs: int,
    splits: tuple[str, ...] = ("random", "ordered"),
) -> None:
    FULLGRID_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["RF_PERM_BACKEND"] = "python"
    os.environ["RF_ONLY_RUN"] = "1"
    os.environ["POWER_TESTS"] = "RFPerm"
    os.environ.setdefault("POWER_BACKEND", "loky")
    os.environ["POWER_CHUNK_MULT"] = "16"

    for enc in encoders:
        for k_label in k_vals:
            pca_k = k_to_pca_k(k_label)
            for split in splits:
                os.environ["RF_SPLIT"] = split
                suf = "" if split == "random" else "_ordered"
                tag = f"{tag_prefix}_k{k_label}{suf}"
                for spec in ("gift", "cross"):
                    ds = f"{PREFIX[spec]}_{enc}"
                    scens = [s for s in SCENARIOS[spec] if scenarios is None or s in scenarios]
                    if not scens:
                        continue
                    run_power(
                        B=150,
                        n_perm=150,
                        n_doc=C.N_DOC,
                        pca_k=pca_k,
                        n_estimators=C.RF_N_ESTIMATORS,
                        n_max=C.ROW_CAP,
                        seed=SEED,
                        datasets=[ds],
                        scenarios=scens,
                        feature_sets=arms,
                        n_jobs=n_jobs,
                        out_dir=FULLGRID_DIR,
                        embed_model=enc,
                        checkpoint_tag=tag,
                    )


def run_vader_rf(encoders: list[str], n_jobs: int, k_vals: tuple[str, ...] = ("full",)) -> None:
    _run_rf_arms(
        encoders=encoders,
        k_vals=k_vals,
        arms=list(VADER_ARMS),
        scenarios=None,
        tag_prefix="fg_vader",
        n_jobs=n_jobs,
    )


def _all_col_arms() -> tuple[list[str], list[str]]:
    _, meta, _, spacy = _verified("MiniLM_L6", "gift")
    m_names = list(meta["meta_names"])
    s_names = list(spacy["spacy_names"])
    single = [f"with_col_{c}" for c in m_names if c not in VADER6] + [f"with_col_spacy:{c}" for c in s_names]
    loo = [f"with_m_loo_{c}" for c in m_names] + [f"with_s_loo_{c}" for c in s_names]
    return single, loo


def run_col_diag(R: int, n_jobs: int) -> dict[str, Any]:
    FULLGRID_DIR.mkdir(parents=True, exist_ok=True)
    _, meta_g, _, spacy_g = _verified("MiniLM_L6", "gift")
    all_names = list(meta_g["meta_names"]) + [f"spacy:{n}" for n in spacy_g["spacy_names"]]

    def _one_spec(spec: str) -> dict[str, Any]:
        Zfeat, Mfull, y, names = load_aligned("MiniLM_L6", 24, seed=SEED, spec=spec)
        sp_y = {
            n: (0.0 if Mfull[:, j].std() == 0 else float(spearmanr(Mfull[:, j], y).correlation))
            for j, n in enumerate(names)
        }
        # Single-column W AUC (on [col] alone) across R windows
        def _scen_w(scen: str) -> dict[str, float]:
            vals = {n: [] for n in names}
            for b in range(R):
                rep_seed = SEED + 10_007 * (zlib.crc32(scen.encode()) % 997) + 17 * b
                _, ye, _, yn, _, _, Mfe, Mfn = make_windows(
                    scen, Zfeat, Mfull, y, names, C.N_DOC, np.random.default_rng(rep_seed)
                )
                Mwin = np.vstack([Mfe, Mfn])
                W = np.r_[np.zeros(len(ye), int), np.ones(len(yn), int)]
                for j, n in enumerate(names):
                    col = Mwin[:, [j]]
                    if col.std() == 0:
                        vals[n].append(0.5)
                    else:
                        e = _crossfit_e(col, W, rep_seed + 3)
                        vals[n].append(float(roc_auc_score(W, e)))
            return {n: float(np.mean(v)) for n, v in vals.items()}

        scens = [s for s in SCENARIOS[spec] if s != "null_shift"] + ["null_shift"]
        w_res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_scen_w)(s) for s in scens)
        return {
            "spearman_y": sp_y,
            "single_col_W_auc": dict(zip(scens, w_res)),
        }

    payload = {
        "R_windows": R,
        "groups": {n: feature_group(n) for n in all_names},
        "specs": {spec: _one_spec(spec) for spec in ("gift", "cross")},
    }
    out = FULLGRID_DIR / "col_diagnostics.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}", file=sys.stderr, flush=True)
    return payload


def run_col_rf(n_jobs: int, mode: str = "all", batch_size: int = 10) -> None:
    single, loo = _all_col_arms()
    if mode == "single":
        arms = single
    elif mode == "loo":
        arms = loo
    else:
        arms = single + loo

    shift_scens = ["sentiment_meta", "syntax_meta", "train_vs_heldout", "domain_shift"]
    for scen_group, scens in (("shift", shift_scens), ("null", ["null_shift"])):
        for start in range(0, len(arms), batch_size):
            sub = arms[start : start + batch_size]
            tag_prefix = f"fg_col_{scen_group}_b{start // batch_size:02d}"
            _run_rf_arms(
                encoders=["MiniLM_L6"],
                k_vals=("24", "full"),
                arms=sub,
                scenarios=scens,
                tag_prefix=tag_prefix,
                n_jobs=n_jobs,
            )


def _load_all_rf_map() -> dict[str, dict[str, dict]]:
    """Load k=24/128/full base arms + vader arms + col ablation arms."""
    rf_data: dict[str, dict[str, dict]] = {"random": {}, "ordered": {}}
    # k=24 base arms from parent checkpoints
    for split, tag in (("random", "amazon_rf_b150"), ("ordered", "amazon_rf_b150_ordered")):
        for ck in sorted(OUT_DIR.glob(f"checkpoint_{tag}_*.npz")):
            ds = ck.stem[len(f"checkpoint_{tag}_"):]
            if not ds.startswith(("amazon_", "amazon-cross_")):
                continue
            for (d, scen, arm), val in _extract_rf_from_ckpt(ck).items():
                rf_data[split][f"{d}|24|{scen}|{arm}"] = val

    # Fullgrid base k=64/128/full + vader + col batches
    for ck in sorted(FULLGRID_DIR.glob("checkpoint_fg_*.npz")):
        stem = ck.stem[len("checkpoint_"):]
        if "_k" not in stem:
            continue
        split = "ordered" if "_ordered_" in stem else "random"
        # Extract k_label cleanly from `_k{24|64|128|full}`
        k_label = None
        for cand in ("24", "64", "128", "full"):
            if f"_k{cand}_" in stem:
                k_label = cand
                break
        if k_label is None:
            continue
        rows = np.load(ck, allow_pickle=True)["rows"]
        bucket = defaultdict(list)
        for r in rows:
            rf = r["RFPerm"]
            if not rf.get("skipped"):
                bucket[(r["dataset"], r["scenario"], r["feature_set"])].append(rf)
        for (d, scen, arm), vals in bucket.items():
            rf_data[split][f"{d}|{k_label}|{scen}|{arm}"] = {
                "n_reps": len(vals),
                "power": float(np.mean([x["p_value"] < 0.05 for x in vals])),
                "power_abs": float(np.mean([x["p_value_abs"] < 0.05 for x in vals])),
                "mean_stat": float(np.mean([x["statistic"] for x in vals])),
            }
    return rf_data


def summarize_vader() -> None:
    rf_data = _load_all_rf_map()
    fg_diag = json.loads((FULLGRID_DIR / "fullgrid_diagnostics.json").read_text()) if (FULLGRID_DIR / "fullgrid_diagnostics.json").is_file() else {"cells": {}}
    vd_diag = json.loads((FULLGRID_DIR / "vader_diagnostics.json").read_text()) if (FULLGRID_DIR / "vader_diagnostics.json").is_file() else {"cells": {}, "spearman_y": {}}

    arms_order = ("without_meta", "with_meta", "with_spacy") + VADER_ARMS
    rnd, ordr = rf_data["random"], rf_data["ordered"]
    f = lambda v: "—" if v is None else f"{v:.2f}"
    df = lambda a, b: "—" if a is None or b is None else f"{a - b:+.2f}"

    lines = ["# Amazon VADER-Focused RFPerm B=150", ""]
    sp_y = vd_diag.get("spearman_y", {})
    if sp_y:
        lines += [
            "## VADER 列与 y 的 Spearman 相关（Amazon gift / cross vs gender）",
            "",
            "| 特征 | gift ρ(y) | cross ρ(y) | gender ρ(y) |",
            "|---|--:|--:|--:|",
        ]
        for c in VADER6:
            lines.append(
                f"| {c} | {sp_y.get('gift', {}).get(c, float('nan')):+.3f} | "
                f"{sp_y.get('cross', {}).get(c, float('nan')):+.3f} | "
                f"{sp_y.get('gender', {}).get(c, float('nan')):+.3f} |"
            )
        lines.append("")

    lines += [
        "## VADER 各臂 RFPerm Power & Δ(arm − without_meta)",
        "",
        "| 编码器 | k | 场景 | 臂 | power | Δ(w/o) | ord power | ord Δ | y_AUC | AUC_W | VarRatio |",
        "|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    summary_rows = []
    for enc in ENCODERS:
        for k_label in ("full", "24", "128"):
            has_vader = any(
                (rnd.get(f"amazon_{enc}|{k_label}|sentiment_meta|{a}") or ordr.get(f"amazon_{enc}|{k_label}|sentiment_meta|{a}"))
                for a in VADER_ARMS
            )
            if not has_vader:
                continue
            for spec in ("gift", "cross"):
                ds = f"{PREFIX[spec]}_{enc}"
                dc_fg = fg_diag["cells"].get(f"{enc}|{k_label}|{spec}", {})
                dc_vd = vd_diag["cells"].get(f"{enc}|{k_label}|{spec}", {})
                y_map = {**dc_fg.get("y_auc", {}), **dc_vd.get("y_auc", {})}
                for scen in SCENARIOS[spec]:
                    w_map = {**dc_fg.get("windows", {}).get(scen, {}), **dc_vd.get("windows", {}).get(scen, {})}
                    wo = (rnd.get(f"{ds}|{k_label}|{scen}|without_meta") or {}).get("power")
                    owo = (ordr.get(f"{ds}|{k_label}|{scen}|without_meta") or {}).get("power")
                    if wo is None:
                        continue
                    for arm in arms_order:
                        p = (rnd.get(f"{ds}|{k_label}|{scen}|{arm}") or {}).get("power")
                        op = (ordr.get(f"{ds}|{k_label}|{scen}|{arm}") or {}).get("power")
                        if p is None and op is None:
                            continue
                        ya = y_map.get(arm)
                        wa = (w_map.get(arm) or {}).get("W_auc")
                        vr = (w_map.get(arm) or {}).get("var_resid_ratio_vs_Z")
                        lines.append(
                            f"| {enc} | {k_label} | {scen} | {arm} | {f(p)} | {df(p, wo)} | {f(op)} | {df(op, owo)} | {f(ya)} | {f(wa)} | {f(vr)} |"
                        )
                        summary_rows.append({
                            "encoder": enc, "k": k_label, "scenario": scen, "arm": arm,
                            "power": p, "delta": None if p is None else p - wo,
                            "ord_power": op, "ord_delta": None if op is None or owo is None else op - owo,
                            "y_auc": ya, "W_auc": wa, "var_resid_ratio": vr,
                        })
    (FULLGRID_DIR / "vader_rf_summary.json").write_text(json.dumps({"spearman_y": sp_y, "rows": summary_rows}, indent=2))
    (FULLGRID_DIR / "vader_rf_table.md").write_text("\n".join(lines) + "\n")
    print("Wrote vader_rf_table.md and vader_rf_summary.json")


def summarize_col() -> None:
    rf_data = _load_all_rf_map()
    col_diag = json.loads((FULLGRID_DIR / "col_diagnostics.json").read_text()) if (FULLGRID_DIR / "col_diagnostics.json").is_file() else {}
    rnd, ordr = rf_data["random"], rf_data["ordered"]
    _, meta_g, _, spacy_g = _verified("MiniLM_L6", "gift")
    m_names = list(meta_g["meta_names"])
    s_names = list(spacy_g["spacy_names"])
    all_cols = [("M", c, c) for c in m_names] + [("S", c, f"spacy:{c}") for c in s_names]

    f = lambda v: "—" if v is None else f"{v:.3f}"
    df = lambda v: "—" if v is None else f"{v:+.3f}"

    lines = ["# Amazon Per-Column Feature Ablation (MiniLM_L6, RFPerm B=150)", ""]
    out_rows = []
    for k_label in ("24", "full"):
        for spec in ("gift", "cross"):
            ds = f"{PREFIX[spec]}_MiniLM_L6"
            sp_y = col_diag.get("specs", {}).get(spec, {}).get("spearman_y", {})
            w_auc_map = col_diag.get("specs", {}).get(spec, {}).get("single_col_W_auc", {})
            for scen in SCENARIOS[spec]:
                wo = (rnd.get(f"{ds}|{k_label}|{scen}|without_meta") or {}).get("power")
                wm = (rnd.get(f"{ds}|{k_label}|{scen}|with_meta") or {}).get("power")
                ws = (rnd.get(f"{ds}|{k_label}|{scen}|with_spacy") or {}).get("power")
                owo = (ordr.get(f"{ds}|{k_label}|{scen}|without_meta") or {}).get("power")
                owm = (ordr.get(f"{ds}|{k_label}|{scen}|with_meta") or {}).get("power")
                ows = (ordr.get(f"{ds}|{k_label}|{scen}|with_spacy") or {}).get("power")
                if wo is None:
                    continue
                scen_rows = []
                for blk, short, full_name in all_cols:
                    s_arm = f"with_col_{full_name}"
                    l_arm = f"with_{blk.lower()}_loo_{short}"
                    p_s = (rnd.get(f"{ds}|{k_label}|{scen}|{s_arm}") or {}).get("power")
                    op_s = (ordr.get(f"{ds}|{k_label}|{scen}|{s_arm}") or {}).get("power")
                    p_l = (rnd.get(f"{ds}|{k_label}|{scen}|{l_arm}") or {}).get("power")
                    op_l = (ordr.get(f"{ds}|{k_label}|{scen}|{l_arm}") or {}).get("power")
                    if p_s is None and p_l is None:
                        continue
                    blk_full = wm if blk == "M" else ws
                    oblk_full = owm if blk == "M" else ows
                    d_s = None if p_s is None else p_s - wo
                    od_s = None if op_s is None or owo is None else op_s - owo
                    d_l_wo = None if p_l is None else p_l - wo
                    d_l_blk = None if p_l is None or blk_full is None else p_l - blk_full
                    od_l_blk = None if op_l is None or oblk_full is None else op_l - oblk_full
                    row = {
                        "k": k_label, "spec": spec, "scenario": scen, "block": blk,
                        "feature": short, "full_name": full_name, "group": feature_group(full_name),
                        "spearman_y": sp_y.get(full_name),
                        "single_W_auc": w_auc_map.get(scen, {}).get(full_name),
                        "p_single": p_s, "delta_single_vs_wo": d_s,
                        "ord_p_single": op_s, "ord_delta_single_vs_wo": od_s,
                        "p_loo": p_l, "delta_loo_vs_wo": d_l_wo, "delta_loo_vs_block": d_l_blk,
                        "ord_p_loo": op_l, "ord_delta_loo_vs_block": od_l_blk,
                    }
                    scen_rows.append(row)
                    out_rows.append(row)
                if not scen_rows:
                    continue
                scen_rows.sort(key=lambda r: (r["delta_single_vs_wo"] if r["delta_single_vs_wo"] is not None else -9.0), reverse=True)
                lines += [
                    f"## k={k_label} · {scen} (w/o={wo:.3f}, with_meta={f(wm)}, with_spacy={f(ws)})",
                    "",
                    "| 块 | 特征 | 组 | ρ(y) | 单列AUC_W | Z+单列 | Δ(单列−w/o) | ord Δ单列 | LOO | Δ(LOO−块) | ord Δ(LOO−块) |",
                    "|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|",
                ]
                for r in scen_rows:
                    lines.append(
                        f"| {r['block']} | {r['feature']} | {r['group']} | {df(r['spearman_y'])} | {f(r['single_W_auc'])} | "
                        f"{f(r['p_single'])} | {df(r['delta_single_vs_wo'])} | {df(r['ord_delta_single_vs_wo'])} | "
                        f"{f(r['p_loo'])} | {df(r['delta_loo_vs_block'])} | {df(r['ord_delta_loo_vs_block'])} |"
                    )
                # Group summary for this (k, scen)
                lines += ["", f"**组汇总 (k={k_label} · {scen})**", "",
                          "| 块 | 组 | n | mean Δ(单列−w/o) | min/max Δ单列 | mean Δ(LOO−块) |",
                          "|---|---|--:|--:|---|--:|"]
                for blk in ("M", "S"):
                    for grp in ("VADER", "长度", "词汇多样性", "POS 比例", "其他"):
                        sub = [r for r in scen_rows if r["block"] == blk and r["group"] == grp]
                        if not sub:
                            continue
                        ds_vals = [r["delta_single_vs_wo"] for r in sub if r["delta_single_vs_wo"] is not None]
                        dl_vals = [r["delta_loo_vs_block"] for r in sub if r["delta_loo_vs_block"] is not None]
                        m_ds = float(np.mean(ds_vals)) if ds_vals else None
                        rng_ds = f"{min(ds_vals):+.3f}/{max(ds_vals):+.3f}" if ds_vals else "—"
                        m_dl = float(np.mean(dl_vals)) if dl_vals else None
                        lines.append(f"| {blk} | {grp} | {len(sub)} | {df(m_ds)} | {rng_ds} | {df(m_dl)} |")
                lines.append("")

    (FULLGRID_DIR / "col_ablation_summary.json").write_text(json.dumps({"rows": out_rows}, indent=2))
    (FULLGRID_DIR / "col_ablation_tables.md").write_text("\n".join(lines) + "\n")
    print("Wrote col_ablation_tables.md and col_ablation_summary.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vader-diag", action="store_true")
    ap.add_argument("--vader-rf", action="store_true")
    ap.add_argument("--vader-summarize", action="store_true")
    ap.add_argument("--col-diag", action="store_true")
    ap.add_argument("--col-rf", action="store_true")
    ap.add_argument("--col-mode", choices=["single", "loo", "all"], default="all")
    ap.add_argument("--col-summarize", action="store_true")
    ap.add_argument("--encoders", nargs="*", default=list(ENCODERS))
    ap.add_argument("--k-vals", nargs="*", default=["full"])
    ap.add_argument("--diag-r", type=int, default=50)
    args = ap.parse_args()

    n_jobs = int(os.environ.get("POWER_N_JOBS", "4"))
    k_tup = tuple(args.k_vals)
    if args.vader_diag:
        run_vader_diag(args.encoders, args.diag_r, n_jobs, k_vals=k_tup)
    if args.vader_rf:
        run_vader_rf(args.encoders, n_jobs, k_vals=k_tup)
    if args.vader_summarize:
        summarize_vader()
    if args.col_diag:
        run_col_diag(args.diag_r, n_jobs)
    if args.col_rf:
        run_col_rf(n_jobs, mode=args.col_mode)
    if args.col_summarize:
        summarize_col()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
