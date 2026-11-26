#!/usr/bin/env python3
"""DRPerm diagnostics: overlap / propensity separation / null calibration per feature arm.

Usage: python3 -m experiments.meta_vs_embed_power.diag_drperm [dataset] [n_rep] [n_perm] [refit|residual] [scenarios,comma]
"""

from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from experiments.meta_vs_embed_power import config as C
from experiments.meta_vs_embed_power.perm_lib import drperm_pvalue
from experiments.meta_vs_embed_power.pipeline import OUT, prepare_dataset
from experiments.meta_vs_embed_power.power import make_windows
from experiments.meta_vs_embed_power.stats_utils import make_folds, rf_classifier

SENT_COLS = ("vader_neg", "vader_neu", "vader_pos", "vader_compound", "polar_gap", "sentiment_std")


def crossfit_e(X, W, seed, n_estimators=60):
    e = np.zeros(len(W))
    for k, te in enumerate(make_folds(len(W), 2, seed)):
        tr = np.setdiff1d(np.arange(len(W)), te)
        m = rf_classifier(seed + 100 + k, n_estimators=max(20, n_estimators // 2)).fit(X[tr], W[tr])
        e[te] = m.predict_proba(X[te])[:, list(m.classes_).index(1)]
    return e


def windows(scenario, Z, M, y, names, n_doc, rng):
    if scenario in ("soft_sentiment", "soft_plus_concept"):
        v = M[:, names.index("vader_compound")]
        s = (v - v.mean()) / v.std()
        p_b = 1 / (1 + np.exp(1.5 * s))
        idx = rng.permutation(len(y))[: 2 * n_doc]
        w = (rng.random(len(idx)) < p_b[idx]).astype(int)
        a, b = idx[w == 0], idx[w == 1]
        yb = y[b].copy()
        if scenario == "soft_plus_concept":
            flip = rng.random(len(b)) < 0.35
            yb[flip] = 1 - yb[flip]
        return Z[a], y[a], Z[b], yb, M[a], M[b]
    if scenario == "concept_shift":
        idx = rng.permutation(len(y))
        a, b = idx[:n_doc], idx[n_doc : 2 * n_doc]
        yb = y[b].copy()
        flip = rng.random(n_doc) < 0.35
        yb[flip] = 1 - yb[flip]
        return Z[a], y[a], Z[b], yb, M[a], M[b]
    Ze, ye, Zn, yn, _, _, Me, Mn = make_windows(scenario, Z, M, y, names, n_doc, rng)
    return Ze, ye, Zn, yn, Me, Mn


def main() -> int:
    ds = sys.argv[1] if len(sys.argv) > 1 else "gender"
    n_rep = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    n_perm = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    scheme = sys.argv[4] if len(sys.argv) > 4 else "refit"
    scenarios = sys.argv[5].split(",") if len(sys.argv) > 5 else ["sentiment_meta", "null_shift", "soft_sentiment", "concept_shift"]
    Z, M, y, names = prepare_dataset(ds, n_max=C.ROW_CAP, seed=2026, pca_k=C.PCA_K, out_dir=OUT)
    keep = np.array([i for i, c in enumerate(names) if c not in SENT_COLS])
    arms = {
        "without_meta": lambda Ze, Me: Ze,
        "with_meta": lambda Ze, Me: np.hstack([Ze, Me]),
        "with_meta_no_sentiment": lambda Ze, Me: np.hstack([Ze, Me[:, keep]]),
        "meta_only": lambda Ze, Me: Me,
    }
    out = {}
    for sc in scenarios:
        for arm, f in arms.items():
            recs = []
            for r in range(n_rep):
                rng = np.random.default_rng(1000 + r)
                Ze, ye, Zn, yn, Me, Mn = windows(sc, Z, M, y, names, C.N_DOC, rng)
                X = np.vstack([f(Ze, Me), f(Zn, Mn)])
                Y = np.concatenate([ye, yn])
                W = np.r_[np.zeros(len(ye), int), np.ones(len(yn), int)]
                e = crossfit_e(X, W, seed=r)
                res = drperm_pvalue(X, Y, W, n_perm=n_perm, seed=r, n_estimators=60, null_scheme=scheme)
                recs.append(
                    {
                        "p": res["p_value"],
                        "auc_e": roc_auc_score(W, e),
                        "clip_frac": float(np.mean((e <= 0.05) | (e >= 0.95))),
                        "abs_W_minus_e": float(np.mean(np.abs(W - np.clip(e, 0.05, 0.95)))),
                        "stat": res["statistic"],
                    }
                )
            agg = {k: float(np.mean([x[k] for x in recs])) for k in recs[0]}
            agg["power"] = float(np.mean([x["p"] < 0.05 for x in recs]))
            out[f"{sc}/{arm}"] = agg
            print(
                f"{sc:15s} {arm:24s} power={agg['power']:.2f} mean_p={agg['p']:.2f} "
                f"AUC_e={agg['auc_e']:.2f} clip={agg['clip_frac']:.2f} |W-e|={agg['abs_W_minus_e']:.2f} "
                f"stat={agg['stat']:.4f}",
                flush=True,
            )
    (OUT / f"diag_drperm_{ds}_{scheme}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
