#!/usr/bin/env python3
"""Amazon meta-vs-embedding ablation (B=50, n_perm=150, R RFPerm).

Requires ``amazon_extract`` (``--spec gift`` and ``--spec cross``) to have passed
alignment. Tests run in tagged passes (``--tests``) so DRPerm can be added later
and merged with ``--merge``.

    python -m experiments.meta_vs_embed_power.run_amazon_b50 --spec gift --tests RFPerm,CFPerm,RRPerm --tag main
    python -m experiments.meta_vs_embed_power.run_amazon_b50 --spec cross --tests RFPerm,CFPerm,RRPerm --tag main
    python -m experiments.meta_vs_embed_power.run_amazon_b50 --spec all --tests DRPerm --tag dr
    python -m experiments.meta_vs_embed_power.run_amazon_b50 --merge
    python -m experiments.meta_vs_embed_power.run_amazon_b50 --aniso
    python -m experiments.meta_vs_embed_power.run_amazon_b50 --shift-evidence
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.meta_vs_embed_power import config as C
from experiments.meta_vs_embed_power.amazon_extract import (
    ENCODERS,
    OUT_DIR,
    emb_path,
    load_cache_embedding,
    load_frame,
    meta_path,
)
from experiments.meta_vs_embed_power.anisotropy import geometry_metrics, sweep_n_pc, write_anisotropy
from experiments.meta_vs_embed_power.pipeline import run_power

B = 50
N_PERM = 150
SEED = 2026
SCENARIOS = {
    "gift": ["null_shift", "sentiment_meta", "syntax_meta", "train_vs_heldout"],
    "cross": ["null_shift", "domain_shift"],
}
PREFIX = {"gift": "amazon", "cross": "amazon-cross"}
SCENARIO_ORDER = ["null_shift", "sentiment_meta", "syntax_meta", "train_vs_heldout", "domain_shift"]
KEY_FEATURES = ("char_len", "word_count", "vader_compound", "vader_pos", "vader_neg", "verb_noun_ratio")


def _specs(spec: str) -> list[str]:
    return list(SCENARIOS) if spec == "all" else [spec]


def _datasets(spec: str, encoders: list[str]) -> list[tuple[str, str]]:
    return [(s, f"{PREFIX[s]}_{e}") for s in _specs(spec) for e in encoders if emb_path(e, s).is_file()]


def run_tests(tests: str, tag: str, spec: str, encoders: list[str], scenarios: list[str] | None = None) -> None:
    os.environ.setdefault("RF_PERM_BACKEND", "python")
    os.environ.pop("RF_ONLY_RUN", None)
    os.environ["POWER_TESTS"] = tests
    os.environ.setdefault("POWER_BACKEND", "loky")
    n_jobs = int(os.environ.get("POWER_N_JOBS", "4"))
    for s, ds in _datasets(spec, encoders):
        scens = [x for x in SCENARIOS[s] if scenarios is None or x in scenarios]
        if not scens:
            continue
        run_power(
            B=B,
            n_perm=N_PERM,
            n_doc=C.N_DOC,
            pca_k=C.PCA_K,
            n_estimators=C.RF_N_ESTIMATORS,
            n_max=C.ROW_CAP,
            seed=SEED,
            datasets=[ds],
            scenarios=scens,
            feature_sets=list(C.FEATURE_SETS),
            n_jobs=n_jobs,
            out_dir=OUT_DIR,
            embed_model=ds.split("_", 1)[1],
            checkpoint_tag=f"amazon_{tag}",
        )


def merge(encoders: list[str]) -> dict:
    tables: dict[str, list[dict]] = {}
    order = {s: i for i, s in enumerate(SCENARIO_ORDER)}
    fs_order = {f: i for i, f in enumerate(C.FEATURE_SETS)}
    for _, ds in _datasets("all", encoders):
        merged: dict[tuple, dict] = {}
        for path in sorted(OUT_DIR.glob(f"power_dataset_amazon_*_{ds}_B{B}.json")):
            for row in json.loads(path.read_text(encoding="utf-8"))["summary"]["table"]:
                key = (row["scenario"], row["feature_set"])
                dst = merged.setdefault(
                    key,
                    {"scenario": key[0], "feature_set": key[1], "n_reps": row.get("n_reps"),
                     "power": {}, "mean_p": {}, "source": {}},
                )
                for m in C.METHODS:
                    if row["power"].get(m) is not None:
                        dst["power"][m] = row["power"][m]
                        dst["mean_p"][m] = row["mean_p"][m]
                        dst["source"][m] = path.name
        if merged:
            tables[ds] = sorted(merged.values(), key=lambda r: (order[r["scenario"]], fs_order[r["feature_set"]]))
    payload = {"B": B, "n_perm": N_PERM, "rf_backend": "rfperm_r", "alpha": 0.05, "tables": tables}
    out = OUT_DIR / f"power_amazon_merged_B{B}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}", file=sys.stderr)
    return payload


def aniso(encoders: list[str]) -> dict:
    by_key, full = {}, {}
    for s, ds in _datasets("all", encoders):
        Z = load_cache_embedding(ds.split("_", 1)[1], s)
        by_key[ds] = sweep_n_pc(Z, C.N_PC_GRID)
        m = geometry_metrics(StandardScaler().fit_transform(Z))
        full[ds] = {"dim": int(Z.shape[1]), **m, "n_pc_suggested": int(np.ceil(m["effective_rank"]))}
    write_anisotropy(OUT_DIR / "anisotropy_effective_rank_amazon.json", {"sweep": by_key, "full_dim": full})
    print(json.dumps(full, indent=2))
    return full


def _compare(A: np.ndarray, Bm: np.ndarray, names: list[str]) -> dict:
    from scipy.stats import ks_2samp

    rows = {}
    for j, n in enumerate(names):
        a, b = A[:, j], Bm[:, j]
        sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) or 1.0
        ks = ks_2samp(a, b)
        rows[n] = {
            "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "median_a": float(np.median(a)), "median_b": float(np.median(b)),
            "smd": float((b.mean() - a.mean()) / sd),
            "ks": float(ks.statistic), "ks_p": float(ks.pvalue),
        }
    k = len(names)
    return {
        "features": rows,
        "n_features": k,
        "n_ks_sig_bonf": int(sum(r["ks_p"] < 0.05 / k for r in rows.values())),
        "mean_abs_smd": float(np.mean([abs(r["smd"]) for r in rows.values()])),
        "top_abs_smd": sorted(((n, round(r["smd"], 3)) for n, r in rows.items()), key=lambda t: -abs(t[1]))[:8],
    }


def _auc(X: np.ndarray, g: np.ndarray) -> float:
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    return float(cross_val_score(clf, X, g, cv=cv, scoring="roc_auc").mean())


def shift_evidence(encoders: list[str]) -> dict:
    out: dict = {}
    for s, col, label in (("gift", "shift_time", "time: early vs late (Gift_Cards)"),
                          ("cross", "shift_domain", "domain: Gift_Cards vs Magazine_Subscriptions")):
        df = load_frame(s)
        meta = np.load(meta_path(s), allow_pickle=True)
        M, names = meta["M"], list(meta["meta_names"])
        g = df[col].to_numpy()
        y = df["y"].to_numpy()
        ev = {"label": label, "n_per_group": [int((g == 0).sum()), int((g == 1).sum())],
              "y_mean_by_group": [float(y[g == 0].mean()), float(y[g == 1].mean())]}
        ev["meta_all"] = _compare(M[g == 0], M[g == 1], names)
        ki = [names.index(f) for f in KEY_FEATURES]
        ev["key_by_y"] = {
            int(c): _compare(M[(g == 0) & (y == c)][:, ki], M[(g == 1) & (y == c)][:, ki], list(KEY_FEATURES))["features"]
            for c in (0, 1)
        }
        ev["group_auc"] = {"meta_only": _auc(M, g)}
        for e in encoders:
            if emb_path(e, s).is_file():
                Z = load_cache_embedding(e, s)
                ev["group_auc"][f"{e}_pca{C.PCA_K}"] = _auc(PCA(C.PCA_K, random_state=SEED).fit_transform(
                    StandardScaler().fit_transform(Z)), g)
        out[s] = ev
    path = OUT_DIR / "shift_evidence_amazon.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {path}", file=sys.stderr)
    return out


def _fmt(v) -> str:
    return "—" if v is None else f"{v:.2f}"


def _dfmt(a, b) -> str:
    return "—" if a is None or b is None else f"{a - b:+.2f}"


def power_table(encoders: list[str]) -> str:
    """Markdown: arms × test × shift scenario with Δ columns; null size collapsed to one line."""
    payload = merge(encoders)
    tests = ("RFPerm", "CFPerm", "RRPerm", "DRPerm")
    lines = [
        "| 编码器 | 场景 | 检验 | without_meta | with_meta | meta_only | Δ(with−without) | Δ(meta_only−without) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    null_rates: dict[str, list[float]] = {}
    for ds, rows in payload["tables"].items():
        enc = ds.split("_", 1)[1]
        by = {(r["scenario"], r["feature_set"]): r["power"] for r in rows}
        for scen in [s for s in SCENARIO_ORDER if s != "null_shift"]:
            if not any((scen, fs) in by for fs in C.FEATURE_SETS):
                continue
            for t in tests:
                p = {fs: by.get((scen, fs), {}).get(t) for fs in C.FEATURE_SETS}
                if all(v is None for v in p.values()):
                    continue
                lines.append(
                    f"| {enc} | {scen} | {t} | {_fmt(p['without_meta'])} | {_fmt(p['with_meta'])} | "
                    f"{_fmt(p['meta_only'])} | {_dfmt(p['with_meta'], p['without_meta'])} | "
                    f"{_dfmt(p['meta_only'], p['without_meta'])} |"
                )
        for fs in C.FEATURE_SETS:
            for t in tests:
                v = by.get(("null_shift", fs), {}).get(t)
                if v is not None:
                    null_rates.setdefault(t, []).append(v)
    if null_rates:
        parts = [f"{t} {min(v):.2f}–{max(v):.2f}" for t, v in null_rates.items()]
        lines.append("")
        lines.append(f"null_shift 拒绝率（size，α=0.05，所有编码器×臂）：{'；'.join(parts)}。")
    md = "\n".join(lines)
    (OUT_DIR / f"power_table_amazon_B{B}.md").write_text(md + "\n", encoding="utf-8")
    return md


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", choices=["gift", "cross", "all"], default="gift")
    ap.add_argument("--tests", default="RFPerm,CFPerm,RRPerm,DRPerm")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--encoders", nargs="*", default=list(ENCODERS))
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--aniso", action="store_true")
    ap.add_argument("--shift-evidence", action="store_true")
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()
    if args.table:
        print(power_table(args.encoders))
    elif args.aniso:
        aniso(args.encoders)
    elif args.shift_evidence:
        shift_evidence(args.encoders)
    elif args.merge:
        print(json.dumps(merge(args.encoders), indent=2))
    else:
        run_tests(args.tests, args.tag, args.spec, args.encoders, args.scenarios)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
