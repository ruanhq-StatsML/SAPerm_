"""Four permutation tests only: RFPerm / CFPerm / DRPerm / RRPerm (RF outcome)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from experiments.meta_vs_embed_power.perm_lib import cfperm_loco_pvalue, drperm_pvalue, rrperm_pvalue
from experiments.meta_vs_embed_power.rfperm_official import rfperm_single_pvalue


def _col_index(meta_names: list[str], *candidates: str) -> int:
    for c in candidates:
        if c in meta_names:
            return meta_names.index(c)
    return 0


def make_windows(
    scenario: str,
    Zpca: np.ndarray,
    M: np.ndarray,
    y: np.ndarray,
    meta_names: list[str],
    n_doc: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(y)
    if scenario == "null_shift":
        idx = rng.permutation(n)
        a, b = idx[:n_doc], idx[n_doc : 2 * n_doc]
    elif scenario == "sentiment_meta":
        score = M[:, _col_index(meta_names, "vader_compound")]
        order = np.argsort(score)
        neg = order[: max(n_doc * 3, n_doc)]
        mid = order[len(order) // 3 : 2 * len(order) // 3]
        a = rng.choice(mid, n_doc, replace=len(mid) < n_doc)
        b = rng.choice(neg, n_doc, replace=len(neg) < n_doc)
    elif scenario == "syntax_meta":
        score = M[:, _col_index(meta_names, "verb_noun_ratio", "verb_ratio")]
        order = np.argsort(score)
        lo, hi = order[: max(n_doc * 3, n_doc)], order[-max(n_doc * 3, n_doc) :]
        a = rng.choice(lo, n_doc, replace=len(lo) < n_doc)
        b = rng.choice(hi, n_doc, replace=len(hi) < n_doc)
    elif scenario in ("train_vs_heldout", "train_vs_test"):
        half = n // 2
        a = rng.choice(np.arange(half), n_doc, replace=half < n_doc)
        b = rng.choice(np.arange(half, n), n_doc, replace=(n - half) < n_doc)
    else:
        raise ValueError(scenario)

    ye, yn = y[a], y[b]
    Ze, Zn = Zpca[a], Zpca[b]
    Me, Mn = M[a], M[b]
    return Ze, ye, Zn, yn, np.hstack([Ze, Me]), np.hstack([Zn, Mn]), Me, Mn


def run_one_rep(
    scenario: str,
    feature_set: str,
    Zpca: np.ndarray,
    M: np.ndarray,
    y: np.ndarray,
    meta_names: list[str],
    *,
    n_doc: int,
    n_perm: int,
    seed: int,
    n_estimators: int,
    pca_k: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    Xe_e, ye, Xn_e, yn, Xe_b, Xn_b, Me, Mn = make_windows(
        scenario, Zpca, M, y, meta_names, n_doc, rng
    )
    if feature_set == "without_meta":
        Xe, Xn = Xe_e, Xn_e
        test_idx = np.arange(max(0, Xe.shape[1] - min(8, M.shape[1])), Xe.shape[1])
    elif feature_set == "with_meta":
        Xe, Xn = Xe_b, Xn_b
        test_idx = np.arange(pca_k, pca_k + M.shape[1])
    elif feature_set == "meta_only":
        Xe, Xn = Me, Mn
        test_idx = np.arange(M.shape[1])
    else:
        raise ValueError(feature_set)

    X = np.vstack([Xe, Xn])
    Y = np.concatenate([ye, yn])
    W = np.concatenate([np.zeros(len(ye), dtype=int), np.ones(len(yn), dtype=int)])

    out: dict[str, Any] = {
        "scenario": scenario,
        "feature_set": feature_set,
        "RFPerm": rfperm_single_pvalue(
            Xe, ye, Xn, yn, n_perm=n_perm, seed=seed, n_estimators=max(n_estimators, 150)
        ),
        "DRPerm": drperm_pvalue(X, Y, W, n_perm=n_perm, seed=seed + 3, n_estimators=n_estimators),
        "RRPerm": rrperm_pvalue(X, Y, W, n_perm=n_perm, seed=seed + 11, n_estimators=n_estimators),
    }
    out["CFPerm"] = cfperm_loco_pvalue(
        X, Y, W, test_indices=test_idx, n_perm=n_perm, seed=seed + 5, n_estimators=n_estimators
    )
    return out


def summarize(rows: list[dict[str, Any]], methods: tuple[str, ...]) -> dict[str, Any]:
    bucket: dict[tuple[str, str, str], list] = defaultdict(list)
    for r in rows:
        bucket[(r["dataset"], r["scenario"], r["feature_set"])].append(r)

    table = []
    for (dataset, scenario, fs), reps in sorted(bucket.items()):
        entry = {"dataset": dataset, "scenario": scenario, "feature_set": fs, "B": len(reps), "power": {}, "mean_p": {}}
        for method in methods:
            ps = [float(rep[method]["p_value"]) for rep in reps]
            rejs = [bool(rep[method]["reject"]) for rep in reps]
            entry["power"][method] = float(np.mean(rejs))
            entry["mean_p"][method] = float(np.mean(ps))
        table.append(entry)
    return {"table": table}