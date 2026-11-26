#!/usr/bin/env python3
"""Amazon full-grid (k = 24, 64, 128, full no-PCA) power & absorption diagnostics.

Writes exclusively under ``results/amazon/fullgrid/`` without modifying any
parent-chain checkpoints or result files in ``results/amazon/``.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from experiments.meta_vs_embed_power import config as C
from experiments.meta_vs_embed_power.amazon_extract import (
    ENCODERS,
    OUT_DIR,
    emb_path,
    load_aligned,
)
from experiments.meta_vs_embed_power.decorr_features import dropped
from experiments.meta_vs_embed_power.pipeline import run_power
from experiments.meta_vs_embed_power.power import make_windows
from experiments.meta_vs_embed_power.stats_utils import make_folds, rf_classifier

FULLGRID_DIR = OUT_DIR / "fullgrid"
SEED = 2026
N_PERM = int(os.environ.get("POWER_N_PERM", "150"))

K_VALUES = ("24", "64", "128", "full")
BASE_ARMS = ("without_meta", "with_meta", "with_spacy")
DECORR_ARMS = ("with_meta_decorr", "with_spacy_decorr")
FULL_B50_ARMS = ("without_meta", "with_meta", "with_spacy", "with_vader4", "with_meta_decorr")
ALL_ARMS = BASE_ARMS + ("with_vader4",) + DECORR_ARMS

SCENARIOS = {
    "gift": ["null_shift", "sentiment_meta", "syntax_meta", "train_vs_heldout"],
    "cross": ["null_shift", "domain_shift"],
}
SCENARIO_ORDER = ["null_shift", "sentiment_meta", "syntax_meta", "train_vs_heldout", "domain_shift"]
PREFIX = {"gift": "amazon", "cross": "amazon-cross"}
CLIP = 0.05


def k_to_pca_k(k_label: str) -> int:
    return 0 if k_label == "full" else int(k_label)


def arms_for_k(k_label: str) -> tuple[str, ...]:
    return ALL_ARMS if k_label in ("24", "full") else BASE_ARMS


def _datasets(spec: str, encoders: list[str]) -> list[tuple[str, str]]:
    specs = list(SCENARIOS) if spec == "all" else [spec]
    return [(s, f"{PREFIX[s]}_{e}") for s in specs for e in encoders if emb_path(e, s).is_file()]


def _arm_matrix(Zfeat: np.ndarray, Mfull: np.ndarray, names: list[str], spec: str, arm: str) -> np.ndarray:
    if arm == "without_meta":
        return Zfeat
    if arm == "with_meta":
        keep = [i for i, n in enumerate(names) if not n.startswith("spacy:")]
        return np.hstack([Zfeat, Mfull[:, keep]])
    if arm == "with_spacy":
        sp = [i for i, n in enumerate(names) if n.startswith("spacy:")]
        return np.hstack([Zfeat, Mfull[:, sp]])
    if arm in ("with_meta_decorr", "with_spacy_decorr"):
        spacy_blk = arm.startswith("with_spacy")
        drop = dropped(spec, "")
        cols = [i for i, n in enumerate(names) if n.startswith("spacy:") == spacy_blk and n not in drop]
        return np.hstack([Zfeat, Mfull[:, cols]])
    raise ValueError(arm)


def _crossfit_e(X: np.ndarray, W: np.ndarray, seed: int) -> np.ndarray:
    n = len(W)
    out = np.zeros(n)
    for k, te in enumerate(make_folds(n, 2, seed)):
        tr = np.setdiff1d(np.arange(n), te)
        e = rf_classifier(seed + 100 + k, n_estimators=max(20, C.RF_N_ESTIMATORS // 2))
        e.fit(X[tr], W[tr])
        classes = list(e.classes_)
        proba = e.predict_proba(X[te])
        out[te] = proba[:, classes.index(1)] if 1 in classes else 1.0 - proba[:, 0]
    return out


def _diag_cell(enc: str, k_label: str, spec: str, R: int) -> dict[str, Any]:
    pca_k = k_to_pca_k(k_label)
    Zfeat, Mfull, y, names = load_aligned(enc, pca_k, seed=SEED, spec=spec)
    yi = y.astype(int)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    arms = arms_for_k(k_label)

    y_auc: dict[str, float] = {}
    for arm in arms:
        X = _arm_matrix(Zfeat, Mfull, names, spec, arm)
        clf = RandomForestClassifier(n_estimators=C.RF_N_ESTIMATORS, random_state=SEED, n_jobs=1)
        p = cross_val_predict(clf, X, yi, cv=cv, method="predict_proba")[:, 1]
        y_auc[arm] = float(roc_auc_score(yi, p))

    windows: dict[str, dict[str, dict[str, float]]] = {}
    for scen in SCENARIOS[spec]:
        acc = {arm: {"auc": [], "r2": [], "extreme": [], "var_resid": []} for arm in arms}
        for b in range(R):
            rep_seed = SEED + 10_007 * (zlib.crc32(scen.encode()) % 997) + 17 * b
            Ze, ye, Zn, yn, _, _, Mfe, Mfn = make_windows(
                scen, Zfeat, Mfull, y, names, C.N_DOC, np.random.default_rng(rep_seed)
            )
            Mwin = np.vstack([Mfe, Mfn])
            Zwin = np.vstack([Ze, Zn])
            W = np.r_[np.zeros(len(ye), int), np.ones(len(yn), int)]
            for arm in arms:
                Xw = _arm_matrix(Zwin, Mwin, names, spec, arm)
                e = _crossfit_e(Xw, W, rep_seed + 3)
                ec = np.clip(e, CLIP, 1.0 - CLIP)
                acc[arm]["auc"].append(roc_auc_score(W, e))
                acc[arm]["r2"].append(1.0 - np.mean((W - e) ** 2) / np.var(W))
                acc[arm]["extreme"].append(float(np.mean((e < 0.02) | (e > 0.98))))
                acc[arm]["var_resid"].append(float(np.var(W - ec)))
        base_var = float(np.mean(acc["without_meta"]["var_resid"]))
        windows[scen] = {
            arm: {
                "W_auc": float(np.mean(v["auc"])),
                "W_r2": float(np.mean(v["r2"])),
                "e_extreme_share": float(np.mean(v["extreme"])),
                "var_resid_W": float(np.mean(v["var_resid"])),
                "var_resid_ratio_vs_Z": float(np.mean(v["var_resid"]) / base_var),
            }
            for arm, v in acc.items()
        }
    return {
        "encoder": enc,
        "k": k_label,
        "dim_Z": int(Zfeat.shape[1]),
        "spec": spec,
        "y_auc": y_auc,
        "windows": windows,
    }


def run_diagnostics(encoders: list[str], k_list: list[str], R: int, n_jobs: int) -> dict[str, Any]:
    FULLGRID_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [(enc, k, spec) for enc in encoders for k in k_list for spec in ("gift", "cross")]
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_diag_cell)(enc, k, spec, R) for enc, k, spec in tasks
    )
    payload: dict[str, Any] = {"R_windows": R, "seed": SEED, "clip_e": CLIP, "cells": {}}
    for item in results:
        key = f"{item['encoder']}|{item['k']}|{item['spec']}"
        payload["cells"][key] = item
    out = FULLGRID_DIR / "fullgrid_diagnostics.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}", file=sys.stderr, flush=True)
    return payload


def run_stage(
    *,
    tests: str,
    k_label: str,
    split: str,
    tag: str,
    spec: str,
    encoders: list[str],
    scenarios: list[str] | None,
    arms: list[str] | None,
    B: int,
) -> None:
    FULLGRID_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["RF_PERM_BACKEND"] = "python"
    os.environ["RF_SPLIT"] = split
    os.environ["POWER_DR_NULL"] = "residual"
    if tests == "RFPerm":
        os.environ["RF_ONLY_RUN"] = "1"
    else:
        os.environ.pop("RF_ONLY_RUN", None)
    os.environ["POWER_TESTS"] = tests
    os.environ.setdefault("POWER_BACKEND", "loky")
    n_jobs = int(os.environ.get("POWER_N_JOBS", "4"))
    pca_k = k_to_pca_k(k_label)
    use_arms = arms if arms else list(arms_for_k(k_label))

    for s, ds in _datasets(spec, encoders):
        scens = [x for x in SCENARIOS[s] if scenarios is None or x in scenarios]
        if not scens:
            continue
        run_power(
            B=B,
            n_perm=N_PERM,
            n_doc=C.N_DOC,
            pca_k=pca_k,
            n_estimators=C.RF_N_ESTIMATORS,
            n_max=C.ROW_CAP,
            seed=SEED,
            datasets=[ds],
            scenarios=scens,
            feature_sets=use_arms,
            n_jobs=n_jobs,
            out_dir=FULLGRID_DIR,
            embed_model=ds.split("_", 1)[1],
            checkpoint_tag=tag,
        )


def _extract_rf_from_ckpt(ckpt_path: Path, alpha: float = 0.05) -> dict[tuple[str, str, str], dict]:
    out: dict[tuple[str, str, str], dict] = {}
    if not ckpt_path.is_file():
        return out
    rows = np.load(ckpt_path, allow_pickle=True)["rows"]
    bucket = defaultdict(list)
    for r in rows:
        rf = r["RFPerm"]
        if not rf.get("skipped"):
            bucket[(r["dataset"], r["scenario"], r["feature_set"])].append(rf)
    for (ds, scen, arm), vals in bucket.items():
        if arm not in ALL_ARMS:
            continue
        out[(ds, scen, arm)] = {
            "n_reps": len(vals),
            "power": float(np.mean([x["p_value"] < alpha for x in vals])),
            "power_abs": float(np.mean([x["p_value_abs"] < alpha for x in vals])),
            "mean_stat": float(np.mean([x["statistic"] for x in vals])),
        }
    return out


def _extract_tests_from_ckpt(ckpt_path: Path, alpha: float = 0.05) -> dict[tuple[str, str, str, str], dict]:
    out: dict[tuple[str, str, str, str], dict] = {}
    if not ckpt_path.is_file():
        return out
    rows = np.load(ckpt_path, allow_pickle=True)["rows"]
    bucket = defaultdict(list)
    for r in rows:
        for t in ("RFPerm", "DRPerm", "RRPerm", "CFPerm"):
            res = r.get(t)
            if res and not res.get("skipped"):
                bucket[(r["dataset"], r["scenario"], r["feature_set"], t)].append(res)
    for key, vals in bucket.items():
        out[key] = {
            "n_reps": len(vals),
            "power": float(np.mean([bool(x["reject"]) for x in vals])),
            "mean_p": float(np.mean([float(x["p_value"]) for x in vals])),
        }
    return out


def _spec_for_ds(ds: str) -> str:
    return "cross" if ds.startswith("amazon-cross_") else "gift"


def _enc_for_ds(ds: str) -> str:
    return ds.split("_", 1)[1]


def summarize_all() -> dict[str, Any]:
    FULLGRID_DIR.mkdir(parents=True, exist_ok=True)
    diag_path = FULLGRID_DIR / "fullgrid_diagnostics.json"
    diag = json.loads(diag_path.read_text(encoding="utf-8")) if diag_path.is_file() else {"cells": {}}

    rf_data: dict[str, dict[str, dict]] = {"random": {}, "ordered": {}}
    # 1. Read-only reuse of k=24 RFPerm B=150 from parent results/amazon/
    for split, tag in (("random", "amazon_rf_b150"), ("ordered", "amazon_rf_b150_ordered")):
        for ck in sorted(OUT_DIR.glob(f"checkpoint_{tag}_*.npz")):
            ds = ck.stem[len(f"checkpoint_{tag}_"):]
            if not ds.startswith(("amazon_", "amazon-cross_")):
                continue
            for (d, scen, arm), val in _extract_rf_from_ckpt(ck).items():
                rf_data[split][f"{d}|24|{scen}|{arm}"] = val

    # 2. Fullgrid RF checkpoints
    for k_label in ("64", "128", "full"):
        for split, suf in (("random", ""), ("ordered", "_ordered")):
            tag = f"fg_rf_k{k_label}{suf}"
            for ck in sorted(FULLGRID_DIR.glob(f"checkpoint_{tag}_*.npz")):
                ds = ck.stem[len(f"checkpoint_{tag}_"):]
                if not ds.startswith(("amazon_", "amazon-cross_")):
                    continue
                for (d, scen, arm), val in _extract_rf_from_ckpt(ck).items():
                    rf_data[split][f"{d}|{k_label}|{scen}|{arm}"] = val

    # 3. Fullgrid DR/RR/CF checkpoints
    b50_data: dict[str, dict] = {}
    for k_label in K_VALUES:
        for tag in (f"fg_drrr_k{k_label}", f"fg_cf_k{k_label}", f"fg_dr_k{k_label}", f"fg_rr_k{k_label}"):
            for ck in sorted(FULLGRID_DIR.glob(f"checkpoint_{tag}_*.npz")):
                for (d, scen, arm, t), val in _extract_tests_from_ckpt(ck).items():
                    b50_data[f"{d}|{k_label}|{scen}|{arm}|{t}"] = val

    summary = {"rf_b150": rf_data, "b50": b50_data, "diagnostics": diag}
    (FULLGRID_DIR / "fullgrid_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Render Markdown tables
    _write_rf_md(summary)
    _write_power_md(summary)
    return summary


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.2f}"


def _dfmt(a: float | None, b: float | None) -> str:
    return "—" if a is None or b is None else f"{a - b:+.2f}"


def _write_rf_md(summary: dict[str, Any]) -> str:
    rnd = summary["rf_b150"]["random"]
    ordr = summary["rf_b150"]["ordered"]
    diag_cells = summary["diagnostics"].get("cells", {})
    lines = [
        "# Amazon Full-Grid RFPerm B=150 (upper-tail, random & ordered)",
        "",
        "| 编码器 | k | 场景 | w/o | meta | spacy | Δmeta | Δspacy | meta_dec | spacy_dec | Δm_dec | Δs_dec | ord w/o / meta / spacy (Δm / Δs) | y_AUC (Z/M/S) | AUC_W (Z/M/S) | VarRatio (M/S) |",
        "|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|---|---|---|",
    ]
    for enc in ENCODERS:
        for k_label in K_VALUES:
            for spec in ("gift", "cross"):
                ds = f"{PREFIX[spec]}_{enc}"
                dc = diag_cells.get(f"{enc}|{k_label}|{spec}", {})
                y_auc = dc.get("y_auc", {})
                win_d = dc.get("windows", {})
                for scen in SCENARIOS[spec]:
                    g = lambda mp, a: (mp.get(f"{ds}|{k_label}|{scen}|{a}") or {}).get("power")
                    wo = g(rnd, "without_meta")
                    if wo is None:
                        continue
                    wm, ws = g(rnd, "with_meta"), g(rnd, "with_spacy")
                    wmd, wsd = g(rnd, "with_meta_decorr"), g(rnd, "with_spacy_decorr")
                    owo, owm, ows = g(ordr, "without_meta"), g(ordr, "with_meta"), g(ordr, "with_spacy")
                    ord_str = f"{_fmt(owo)} / {_fmt(owm)} / {_fmt(ows)} ({_dfmt(owm, owo)} / {_dfmt(ows, owo)})"
                    ya = f"{_fmt(y_auc.get('without_meta'))}/{_fmt(y_auc.get('with_meta'))}/{_fmt(y_auc.get('with_spacy'))}"
                    wd = win_d.get(scen, {})
                    wa = "/".join(_fmt((wd.get(a) or {}).get("W_auc")) for a in BASE_ARMS)
                    vr = "/".join(_fmt((wd.get(a) or {}).get("var_resid_ratio_vs_Z")) for a in ("with_meta", "with_spacy"))
                    lines.append(
                        f"| {enc} | {k_label} | {scen} | {_fmt(wo)} | {_fmt(wm)} | {_fmt(ws)} | "
                        f"{_dfmt(wm, wo)} | {_dfmt(ws, wo)} | {_fmt(wmd)} | {_fmt(wsd)} | "
                        f"{_dfmt(wmd, wo)} | {_dfmt(wsd, wo)} | {ord_str} | {ya} | {wa} | {vr} |"
                    )
    md = "\n".join(lines) + "\n"
    (FULLGRID_DIR / "fullgrid_rf_table.md").write_text(md, encoding="utf-8")
    return md


def _write_power_md(summary: dict[str, Any]) -> str:
    b50 = summary["b50"]
    diag_cells = summary["diagnostics"].get("cells", {})
    vd_path = FULLGRID_DIR / "vader_diagnostics.json"
    vd_cells = json.loads(vd_path.read_text(encoding="utf-8")).get("cells", {}) if vd_path.is_file() else {}
    lines = [
        "# Amazon Full-Grid DR / RR / CF (B=50, n_perm=150) + Diagnostics",
        "",
        "| 编码器 | k | 场景 | 检验 | w/o | meta | Δmeta | spacy | Δspacy | vader4 | Δvader4 | meta_dec | Δm_dec | y_AUC (Z/M/S/V4/Md) | AUC_W (Z/M/S/V4/Md) | VarRatio (M/S/V4/Md) |",
        "|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|---|---|",
    ]
    show_arms = ("without_meta", "with_meta", "with_spacy", "with_vader4", "with_meta_decorr")
    for enc in ENCODERS:
        for k_label in K_VALUES:
            for spec in ("gift", "cross"):
                ds = f"{PREFIX[spec]}_{enc}"
                dc = diag_cells.get(f"{enc}|{k_label}|{spec}", {})
                vc = vd_cells.get(f"{enc}|{k_label}|{spec}", {})
                y_auc = {**dc.get("y_auc", {}), **vc.get("y_auc", {})}
                win_d = dc.get("windows", {})
                vwin_d = vc.get("windows", {})
                for scen in SCENARIOS[spec]:
                    for t in ("DRPerm", "RRPerm", "CFPerm"):
                        g = lambda a: (b50.get(f"{ds}|{k_label}|{scen}|{a}|{t}") or {}).get("power")
                        vals = [g(a) for a in show_arms]
                        if all(v is None for v in vals):
                            continue
                        wo, wm, ws, wv, wmd = vals
                        ya = "/".join(_fmt(y_auc.get(a)) for a in show_arms)
                        wd = {**win_d.get(scen, {}), **vwin_d.get(scen, {})}
                        wa = "/".join(_fmt((wd.get(a) or {}).get("W_auc")) for a in show_arms)
                        vr = "/".join(_fmt((wd.get(a) or {}).get("var_resid_ratio_vs_Z")) for a in show_arms[1:])
                        lines.append(
                            f"| {enc} | {k_label} | {scen} | {t} | {_fmt(wo)} | {_fmt(wm)} | {_dfmt(wm, wo)} | "
                            f"{_fmt(ws)} | {_dfmt(ws, wo)} | {_fmt(wv)} | {_dfmt(wv, wo)} | "
                            f"{_fmt(wmd)} | {_dfmt(wmd, wo)} | {ya} | {wa} | {vr} |"
                        )
    md = "\n".join(lines) + "\n"
    (FULLGRID_DIR / "fullgrid_power_table.md").write_text(md, encoding="utf-8")
    return md


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--diag-r", type=int, default=50)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--tests", default="RFPerm")
    ap.add_argument("--k", choices=K_VALUES, default="full")
    ap.add_argument("--split", choices=["random", "ordered"], default="random")
    ap.add_argument("--tag", default="")
    ap.add_argument("--spec", choices=["gift", "cross", "all"], default="all")
    ap.add_argument("--encoders", nargs="*", default=list(ENCODERS))
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--b", type=int, default=50)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()

    n_jobs = int(os.environ.get("POWER_N_JOBS", "4"))
    if args.diag:
        run_diagnostics(args.encoders, list(K_VALUES), args.diag_r, n_jobs)
    elif args.run:
        tag = args.tag or f"fg_{args.tests.lower().replace(',', '_')}_k{args.k}"
        run_stage(
            tests=args.tests,
            k_label=args.k,
            split=args.split,
            tag=tag,
            spec=args.spec,
            encoders=args.encoders,
            scenarios=args.scenarios,
            arms=args.arms,
            B=args.b,
        )
    elif args.summarize:
        summarize_all()
        print("Summary written under", FULLGRID_DIR)
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
