#!/usr/bin/env python3
"""Incremental value of spaCy+VADER meta-features on BiasFinder BERT embeddings.

Builds X = [Z_PCA(embed), M_meta] vs X = Z_PCA(embed) only, then compares power of:

  - RFPerm
  - DRPerm (PO-risk)
  - CFPerm (LOCO on the meta block / matched trailing dims)

Settings (as requested):
  N_perm = 120, B = 150, RandomForest outcome model.

Data
----
  datasets/train_biasfinder_bert_embedding.csv  (Z, class)  — aligned 1:1 with
  GEN-sarc-notsarc.csv (text) from datasets/dataset.zip

Scenarios
---------
  null_shift          : random split (Type-I)
  sentiment_meta      : new window = low VADER compound (meta-targeted)
  train_vs_heldout    : first half vs second half by original order (mild)

Does NOT run other benchmark methods (MMD/C2ST/...).

Usage
-----
  python power_embed_meta.py --smoke          # tiny B for wiring
  python power_embed_meta.py                  # full B=150, n_perm=120
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from extract_features import BLOCKS, FEATURE_NAMES, build_matrix, extract_meta
from perm_tests import cfperm_loco_pvalue, drperm_pvalue, rfperm_pvalue

ROOT = Path(__file__).resolve().parents[3]  # SAPerm repo root
EMB_TRAIN = ROOT / "datasets" / "train_biasfinder_bert_embedding.csv"
DATA_ZIP = ROOT / "datasets" / "dataset.zip"
OUT_DIR = Path(__file__).resolve().parent / "results"


def load_aligned_embed_text(
    n_max: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return Z (n,768), y (n,), texts (n,) with GEN↔embedding row alignment."""
    # texts
    texts: list[str] = []
    classes_txt: list[str] = []
    with zipfile.ZipFile(DATA_ZIP) as zf:
        with zf.open("GEN-sarc-notsarc.csv") as raw:
            import io

            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"))
            for row in reader:
                classes_txt.append(row["class"])
                texts.append(row["text"])

    Z_list: list[list[float]] = []
    y_list: list[int] = []
    classes_emb: list[str] = []
    with EMB_TRAIN.open(encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header[-1] == "class"
        # drop leading pandas index column if present
        start = 1 if header[0] in ("", "Unnamed: 0") or header[0].lower() == "unnamed: 0" else 0
        if header[0] == "0" and header[1] == "1":
            # numbered embedding dims starting at 0 — keep all but class
            start = 0
        for row in reader:
            classes_emb.append(row[-1])
            feats = row[start:-1]
            # if 769 dims and first looks like a row id, drop it
            if len(feats) == 769:
                feats = feats[1:]
            Z_list.append([float(v) for v in feats])
            y_list.append(1 if row[-1] == "sarc" else 0)

    if classes_txt != classes_emb:
        raise RuntimeError("GEN text classes are not aligned with embedding rows")

    Z = np.asarray(Z_list, dtype=float)
    y = np.asarray(y_list, dtype=float)
    if n_max is not None and n_max < len(y):
        rng = np.random.default_rng(0)
        # stratified subsample
        idx0 = np.where(y == 0)[0]
        idx1 = np.where(y == 1)[0]
        k0 = n_max // 2
        k1 = n_max - k0
        take = np.concatenate(
            [
                rng.choice(idx0, size=min(k0, len(idx0)), replace=False),
                rng.choice(idx1, size=min(k1, len(idx1)), replace=False),
            ]
        )
        take.sort()
        Z, y = Z[take], y[take]
        texts = [texts[i] for i in take]
    return Z, y, texts


def compute_or_cache_meta(texts: list[str], cache: Path) -> tuple[np.ndarray, list[str]]:
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        names = list(data["names"])
        if names == FEATURE_NAMES:
            return data["M"], names
        print(
            f"Meta cache schema mismatch ({len(names)} vs {len(FEATURE_NAMES)}); rebuilding.",
            file=sys.stderr,
        )
    print(f"Extracting meta-features for {len(texts)} docs (spaCy+VADER)...", file=sys.stderr)
    M_list, names = build_matrix(texts, "meta")
    M = np.asarray(M_list, dtype=float)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, M=M, names=np.array(names, dtype=object))
    return M, names


def build_Zpca(Z: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    pca = PCA(n_components=k, random_state=seed)
    return pca.fit_transform(StandardScaler().fit_transform(Z))


def make_windows(
    scenario: str,
    Zpca: np.ndarray,
    M: np.ndarray,
    y: np.ndarray,
    texts: list[str],
    n_doc: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Xe, ye, Xn, yn for embed-only and for embed+meta (same rows)."""
    n = len(y)
    compounds = M[:, FEATURE_NAMES.index("vader_compound")]

    if scenario == "null_shift":
        idx = rng.permutation(n)
        a, b = idx[:n_doc], idx[n_doc : 2 * n_doc]
    elif scenario == "sentiment_meta":
        # ref: mid polarity; new: strongly negative — meta-targeted alternative
        order = np.argsort(compounds)
        neg = order[: max(n_doc * 3, n_doc)]
        mid = order[len(order) // 3 : 2 * len(order) // 3]
        a = rng.choice(mid, size=n_doc, replace=len(mid) < n_doc)
        b = rng.choice(neg, size=n_doc, replace=len(neg) < n_doc)
    elif scenario == "syntax_meta":
        # HARD DGP analog on real embeddings: route by POS structure
        # (verb_noun_ratio) — length-matched surface, syntax flips.
        score = M[:, FEATURE_NAMES.index("verb_noun_ratio")]
        order = np.argsort(score)
        lo = order[: max(n_doc * 3, n_doc)]
        hi = order[-max(n_doc * 3, n_doc) :]
        a = rng.choice(lo, size=n_doc, replace=len(lo) < n_doc)
        b = rng.choice(hi, size=n_doc, replace=len(hi) < n_doc)
    elif scenario == "train_vs_heldout":
        half = n // 2
        a = rng.choice(np.arange(half), size=n_doc, replace=False)
        b = rng.choice(np.arange(half, n), size=n_doc, replace=False)
    else:
        raise ValueError(scenario)

    Ze, Zn = Zpca[a], Zpca[b]
    Me, Mn = M[a], M[b]
    ye, yn = y[a], y[b]
    Xe_embed, Xn_embed = Ze, Zn
    Xe_both = np.hstack([Ze, Me])
    Xn_both = np.hstack([Zn, Mn])
    return Xe_embed, ye, Xn_embed, yn, Xe_both, Xn_both


def run_one_rep(
    scenario: str,
    feature_set: str,
    Zpca: np.ndarray,
    M: np.ndarray,
    y: np.ndarray,
    texts: list[str],
    *,
    n_doc: int,
    n_perm: int,
    seed: int,
    n_estimators: int,
    pca_k: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    Xe_e, ye, Xn_e, yn, Xe_b, Xn_b = make_windows(
        scenario, Zpca, M, y, texts, n_doc, rng
    )
    if feature_set == "embed":
        Xe, Xn = Xe_e, Xn_e
        meta_idx = np.array([], dtype=int)
    else:
        Xe, Xn = Xe_b, Xn_b
        meta_idx = np.arange(pca_k, pca_k + M.shape[1])

    # RFPerm
    rf = rfperm_pvalue(
        Xe, ye, Xn, yn, n_perm=n_perm, seed=seed, n_estimators=n_estimators
    )

    # DRPerm on pooled with W=batch
    X = np.vstack([Xe, Xn])
    Y = np.concatenate([ye, yn])
    W = np.concatenate([np.zeros(len(ye), dtype=int), np.ones(len(yn), dtype=int)])
    dr = drperm_pvalue(
        X, Y, W, n_perm=n_perm, seed=seed + 3, n_estimators=n_estimators
    )

    # CFPerm: for embed-only, test last min(8, p) PCA dims as a placebo block;
    # for embed+meta, test the meta block (incremental value localization).
    if feature_set == "embed":
        test_idx = np.arange(max(0, Xe.shape[1] - 8), Xe.shape[1])
    else:
        test_idx = meta_idx
    cf = cfperm_loco_pvalue(
        X,
        Y,
        W,
        test_indices=test_idx,
        n_perm=n_perm,
        seed=seed + 5,
        n_estimators=n_estimators,
    )

    return {
        "scenario": scenario,
        "feature_set": feature_set,
        "RFPerm": rf,
        "DRPerm": dr,
        "CFPerm": cf,
    }


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    return obj


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from collections import defaultdict

    bucket: dict[tuple[str, str], list] = defaultdict(list)
    for r in rows:
        bucket[(r["scenario"], r["feature_set"])].append(r)

    table = []
    for (scenario, fs), reps in sorted(bucket.items()):
        entry = {
            "scenario": scenario,
            "feature_set": fs,
            "B": len(reps),
            "power": {},
            "mean_p": {},
            "mean_stat": {},
        }
        for method in ("RFPerm", "DRPerm", "CFPerm"):
            ps = [float(rep[method]["p_value"]) for rep in reps]
            rejs = [bool(rep[method]["reject"]) for rep in reps]
            stats = [float(rep[method]["statistic"]) for rep in reps]
            entry["power"][method] = float(np.mean(rejs))
            entry["mean_p"][method] = float(np.mean(ps))
            entry["mean_stat"][method] = float(np.mean(stats))
        table.append(entry)

    # deltas meta - embed
    deltas = []
    scenarios = sorted({t["scenario"] for t in table})
    for sc in scenarios:
        e = next(t for t in table if t["scenario"] == sc and t["feature_set"] == "embed")
        m = next(t for t in table if t["scenario"] == sc and t["feature_set"] == "embed_meta")
        deltas.append(
            {
                "scenario": sc,
                "delta_power": {
                    method: m["power"][method] - e["power"][method]
                    for method in ("RFPerm", "DRPerm", "CFPerm")
                },
            }
        )
    return {"table": table, "deltas": deltas}


def render_md(result: dict[str, Any]) -> str:
    cfg = result["config"]
    lines = [
        "# Embedding ± Meta-feature power (RFPerm / DRPerm / CFPerm)",
        "",
        "Question: do spaCy+VADER meta-features (M) give **incremental value** on top of "
        "BiasFinder BERT embeddings (Z) for distribution-shift tests?",
        "",
        f"Config: B={cfg['B']}, N_perm={cfg['n_perm']}, n_doc={cfg['n_doc']}, "
        f"PCA(Z)={cfg['pca_k']}, RF n_estimators≈{cfg['n_estimators']}, α=0.05.",
        "",
        "Feature sets: `embed` = Z_PCA only; `embed_meta` = [Z_PCA, M].",
        "",
        "## Power / Type-I",
        "",
        "| scenario | set | RFPerm | DRPerm | CFPerm |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for t in result["summary"]["table"]:
        metric = "typeI" if t["scenario"] == "null_shift" else "power"
        lines.append(
            "| {sc} ({met}) | {fs} | {rf:.2f} | {dr:.2f} | {cf:.2f} |".format(
                sc=t["scenario"],
                met=metric,
                fs=t["feature_set"],
                rf=t["power"]["RFPerm"],
                dr=t["power"]["DRPerm"],
                cf=t["power"]["CFPerm"],
            )
        )
    lines += ["", "## Δ power (embed_meta − embed)", ""]
    for d in result["summary"]["deltas"]:
        lines.append(
            "- `{sc}`: RFPerm {rf:+.2f}, DRPerm {dr:+.2f}, CFPerm {cf:+.2f}".format(
                sc=d["scenario"],
                rf=d["delta_power"]["RFPerm"],
                dr=d["delta_power"]["DRPerm"],
                cf=d["delta_power"]["CFPerm"],
            )
        )
    lines += [
        "",
        "## PO-risk interpretation (DRPerm)",
        "",
        result.get(
            "po_risk_note",
            "PO-risk = E[τ(X)^2] with τ≈(Y−μ(X))(W−e(X)).",
        ),
        "",
        "Reading guide for ±meta deltas:",
        "",
        "1. **Covariate shift absorbed by M** (e.g. `sentiment_meta`): adding M should "
        "*lower* DRPerm power if μ(X), e(X) become better nuisances — residual "
        "(Y−μ)(W−e) collapses. That is incremental value as *adjustment*, not as "
        "a louder alarm.",
        "2. **Concept / label-channel shift**: if batches differ in P(Y|X) even after "
        "M, DRPerm power stays high on both feature sets; CFPerm then localizes "
        "which block of X mediates the residual batch effect.",
        "3. **RFPerm** responds to predictive-risk gaps (OOB vs new). Meta can raise "
        "power when M makes the risk gap sharper, or lower it when M stabilizes "
        "extrapolation across batches.",
        "4. **CFPerm LOCO** on the meta block: importance = R(−M) − R(full) with "
        "R = E[(μ₁−μ₀)²]. Positive Δ power vs the embed placebo block ⇒ M carries "
        "shift-localizing signal beyond Z.",
        "",
        "## CFPerm block note",
        "",
        "CFPerm here tests the **meta block** under `embed_meta` (LOCO R-risk). "
        "Under `embed` it tests a trailing PCA placebo block of similar size. "
        "A positive Δ on CFPerm is direct evidence that M carries shift-localizing signal beyond Z.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--B", type=int, default=150)
    p.add_argument("--n-perm", type=int, default=120)
    p.add_argument("--n-doc", type=int, default=180)
    p.add_argument("--pca-k", type=int, default=32)
    p.add_argument("--n-estimators", type=int, default=60)
    p.add_argument("--n-max", type=int, default=2000, help="Cap aligned rows for speed")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--smoke", action="store_true", help="B=4, small n for wiring")
    p.add_argument(
        "--scenarios",
        nargs="*",
        default=["null_shift", "sentiment_meta", "syntax_meta", "train_vs_heldout"],
    )
    p.add_argument("--n-jobs", type=int, default=4)
    args = p.parse_args(argv)

    if args.smoke:
        args.B = 4
        args.n_doc = 80
        args.n_perm = 40
        args.n_estimators = 30
        args.n_max = 800
        args.pca_k = 16
        args.n_jobs = 1

    t0 = time.time()
    Z, y, texts = load_aligned_embed_text(n_max=args.n_max)
    cache = OUT_DIR / f"meta_cache_n{len(texts)}.npz"
    M, meta_names = compute_or_cache_meta(texts, cache)
    assert meta_names == FEATURE_NAMES or set(FEATURE_NAMES).issubset(set(meta_names))
    # align columns to FEATURE_NAMES order
    name_to_j = {n: i for i, n in enumerate(meta_names)}
    M = M[:, [name_to_j[n] for n in FEATURE_NAMES]]

    Zpca = build_Zpca(Z, args.pca_k, seed=args.seed)
    print(
        f"Loaded n={len(y)}, p_embed={Z.shape[1]} → PCA {args.pca_k}, "
        f"p_meta={M.shape[1]}, B={args.B}, n_perm={args.n_perm}, jobs={args.n_jobs}",
        file=sys.stderr,
    )

    jobs: list[tuple[str, str, int]] = []
    for scenario in args.scenarios:
        for fs in ("embed", "embed_meta"):
            for b in range(args.B):
                jobs.append((scenario, fs, b))

    def _worker(item: tuple[str, str, int]) -> dict[str, Any]:
        scenario, fs, b = item
        seed = args.seed + 1000 * (0 if fs == "embed" else 1) + 17 * b + (abs(hash(scenario)) % 97)
        return run_one_rep(
            scenario,
            fs,
            Zpca,
            M,
            y,
            texts,
            n_doc=args.n_doc,
            n_perm=args.n_perm,
            seed=seed,
            n_estimators=args.n_estimators,
            pca_k=args.pca_k,
        )

    ckpt = OUT_DIR / f"power_embed_meta_{'smoke' if args.smoke else 'full'}.ckpt.npz"
    done: dict[tuple[str, str, int], dict[str, Any]] = {}
    if ckpt.exists() and not args.smoke:
        raw = np.load(ckpt, allow_pickle=True)
        for rec in raw["rows"]:
            r = rec.item() if isinstance(rec, np.ndarray) else rec
            # rows saved as dicts
            if isinstance(r, dict) and "scenario" in r:
                # recover job key from rows_meta if present
                pass
        # structured checkpoint: keys + rows
        if "keys" in raw.files:
            keys = [tuple(k) for k in raw["keys"]]
            for k, rec in zip(keys, raw["rows"]):
                done[(str(k[0]), str(k[1]), int(k[2]))] = rec.item() if hasattr(rec, "item") else rec
            print(f"Resumed {len(done)}/{len(jobs)} reps from {ckpt}", file=sys.stderr)

    pending = [j for j in jobs if j not in done]

    def _save_ckpt(finished: dict[tuple[str, str, int], dict[str, Any]]) -> None:
        keys = np.array([[k[0], k[1], k[2]] for k in finished.keys()], dtype=object)
        rows_arr = np.array([finished[k] for k in finished.keys()], dtype=object)
        np.savez_compressed(ckpt, keys=keys, rows=rows_arr)

    if args.n_jobs == 1:
        for i, item in enumerate(pending, 1):
            rep = _worker(item)
            done[item] = rep
            if i % 10 == 0 or i == len(pending):
                print(
                    f"[{len(done)}/{len(jobs)}] {item[0]}/{item[1]} "
                    f"RF={rep['RFPerm']['p_value']:.3f} "
                    f"DR={rep['DRPerm']['p_value']:.3f} "
                    f"CF={rep['CFPerm']['p_value']:.3f}",
                    file=sys.stderr,
                )
            if i % 20 == 0 or i == len(pending):
                _save_ckpt(done)
    else:
        from joblib import Parallel, delayed

        chunk = max(args.n_jobs * 4, 8)
        for start in range(0, len(pending), chunk):
            batch = pending[start : start + chunk]
            batch_rows = Parallel(n_jobs=args.n_jobs, verbose=5)(
                delayed(_worker)(item) for item in batch
            )
            for item, rep in zip(batch, batch_rows):
                done[item] = rep
            _save_ckpt(done)
            print(
                f"[{len(done)}/{len(jobs)}] checkpointed after chunk "
                f"{start // chunk + 1}",
                file=sys.stderr,
            )

    rows = [done[j] for j in jobs]

    summary = summarize(rows)
    result = {
        "config": {
            "B": args.B,
            "n_perm": args.n_perm,
            "n_doc": args.n_doc,
            "pca_k": args.pca_k,
            "n_estimators": args.n_estimators,
            "n_max": args.n_max,
            "seed": args.seed,
            "scenarios": args.scenarios,
            "meta_blocks": BLOCKS,
            "meta_dim": M.shape[1],
            "packages": ["spacy:en_core_web_sm", "vaderSentiment", "sklearn.RF"],
            "elapsed_sec": time.time() - t0,
        },
        "summary": summary,
        "po_risk_note": (
            "DRPerm PO-risk = E[τ(X)^2], τ≈(Y−μ(X))(W−e(X)). "
            "If adding M shrinks residual batch–outcome dependence, DRPerm power may drop "
            "(better nuisance adjustment). If M exposes concept signal that Z missed, power rises. "
            "Both are scientifically informative for incremental value."
        ),
        "rows_tail": [
            {
                "scenario": r["scenario"],
                "feature_set": r["feature_set"],
                "RFPerm": {k: (bool(v) if k == "reject" else v) for k, v in r["RFPerm"].items() if k != "interpretation"},
                "DRPerm": {
                    k: (bool(v) if isinstance(v, (bool, np.bool_)) else v)
                    for k, v in r["DRPerm"].items()
                    if k != "interpretation"
                },
                "CFPerm": {k: (bool(v) if k == "reject" else v) for k, v in r["CFPerm"].items()},
            }
            for r in rows[-6:]
        ],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if args.smoke else "full"
    out_json = OUT_DIR / f"power_embed_meta_{tag}.json"
    out_md = OUT_DIR / f"power_embed_meta_{tag}.md"
    out_json.write_text(json.dumps(_jsonify(result), indent=2), encoding="utf-8")
    out_md.write_text(render_md(result), encoding="utf-8")
    print(render_md(result))
    print(f"Wrote {out_json}", file=sys.stderr)
    print(f"Wrote {out_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
