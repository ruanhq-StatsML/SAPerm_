#!/usr/bin/env python3
"""Amazon VADER-focused RFPerm B=150 arms (diagnostics, runs, summary).

All outputs are stored under ``results/amazon/fullgrid/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
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
    SCENARIOS,
    SEED,
    _crossfit_e,
    load_rf_map,
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
                from joblib.externals.loky import get_reusable_executor

                get_reusable_executor().shutdown(wait=True)
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


def run_vader_rf(
    encoders: list[str],
    n_jobs: int,
    k_vals: tuple[str, ...] = ("full",),
    splits: tuple[str, ...] = ("random", "ordered"),
) -> None:
    _run_rf_arms(
        encoders=encoders,
        k_vals=k_vals,
        arms=list(VADER_ARMS),
        scenarios=None,
        tag_prefix="fg_vader",
        n_jobs=n_jobs,
        splits=splits,
    )


def summarize_vader() -> None:
    rf_data = load_rf_map()
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vader-diag", action="store_true")
    ap.add_argument("--vader-rf", action="store_true")
    ap.add_argument("--vader-summarize", action="store_true")
    ap.add_argument("--vader-splits", nargs="*", default=["random", "ordered"])
    ap.add_argument("--encoders", nargs="*", default=list(ENCODERS))
    ap.add_argument("--k-vals", nargs="*", default=["full"])
    ap.add_argument("--diag-r", type=int, default=50)
    args = ap.parse_args()

    n_jobs = int(os.environ.get("POWER_N_JOBS", "4"))
    k_tup = tuple(args.k_vals)
    if args.vader_diag:
        run_vader_diag(args.encoders, args.diag_r, n_jobs, k_vals=k_tup)
    if args.vader_rf:
        run_vader_rf(args.encoders, n_jobs, k_vals=k_tup, splits=tuple(args.vader_splits))
    if args.vader_summarize:
        summarize_vader()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
