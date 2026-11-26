#!/usr/bin/env python3
"""Absorption evidence on cached Amazon artifacts (no new extraction, single core).

For each shift scenario and feature set, on the same windows as the power runs
(worker rep seeds, b = 0..R-1) and the same cross-fitted propensity as DRPerm:
  - W predictability: CV AUC and R^2 = 1 - Brier / Var(W) of e(X)
  - positivity: share of raw e(X) < 0.02 or > 0.98
  - Var(W - e_clip) relative to Z_pca (the DR residual-W variance)
On the full frame: CV AUC of y from each feature set, and top-5 |spearman(feature, y)|.

    nice -n 19 python -m experiments.meta_vs_embed_power.absorption_table [R]
"""

from __future__ import annotations

import json
import sys
import zlib

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from experiments.meta_vs_embed_power import config as C
from experiments.meta_vs_embed_power.amazon_extract import OUT_DIR, load_aligned
from experiments.meta_vs_embed_power.power import make_windows
from experiments.meta_vs_embed_power.stats_utils import make_folds, rf_classifier

SEED = 2026
SCENARIOS = {"gift": ["sentiment_meta", "syntax_meta", "train_vs_heldout"], "cross": ["domain_shift"]}
SETS = ("Z", "M", "S", "Z+M", "Z+S")
CLIP = 0.05


def _split(Mfull: np.ndarray, names: list[str]):
    sp = [i for i, n in enumerate(names) if n.startswith("spacy:")]
    keep = [i for i, n in enumerate(names) if not n.startswith("spacy:")]
    return Mfull[:, keep], [names[i] for i in keep], Mfull[:, sp], [names[i][6:] for i in sp]


def _sets(Z, M, S) -> dict[str, np.ndarray]:
    return {"Z": Z, "M": M, "S": S, "Z+M": np.hstack([Z, M]), "Z+S": np.hstack([Z, S])}


def _crossfit_e(X: np.ndarray, W: np.ndarray, seed: int) -> np.ndarray:
    """Same folds / model / seeds as perm_lib.drperm_pvalue (seed = rep_seed + 3)."""
    n = len(W)
    out = np.zeros(n)
    for k, te in enumerate(make_folds(n, 2, seed)):
        tr = np.setdiff1d(np.arange(n), te)
        e = rf_classifier(seed + 100 + k, n_estimators=max(20, C.RF_N_ESTIMATORS // 2))
        e.fit(X[tr], W[tr])
        out[te] = e.predict_proba(X[te])[:, list(e.classes_).index(1)]
    return out


def window_metrics(R: int, encoder: str) -> dict:
    res: dict = {}
    for spec, scens in SCENARIOS.items():
        Zpca, Mfull, y, names = load_aligned(encoder, C.PCA_K, spec=spec)
        for scen in scens:
            acc = {s: {"auc": [], "r2": [], "extreme": [], "var_resid": []} for s in SETS}
            for b in range(R):
                rep_seed = SEED + 10_007 * (zlib.crc32(scen.encode()) % 997) + 17 * b
                Ze, ye, Zn, yn, _, _, Mfe, Mfn = make_windows(
                    scen, Zpca, Mfull, y, names, C.N_DOC, np.random.default_rng(rep_seed)
                )
                Mw, _, Sw, _ = _split(np.vstack([Mfe, Mfn]), names)
                W = np.r_[np.zeros(len(ye), int), np.ones(len(yn), int)]
                for s, X in _sets(np.vstack([Ze, Zn]), Mw, Sw).items():
                    e = _crossfit_e(X, W, rep_seed + 3)
                    ec = np.clip(e, CLIP, 1 - CLIP)
                    acc[s]["auc"].append(roc_auc_score(W, e))
                    acc[s]["r2"].append(1.0 - np.mean((W - e) ** 2) / np.var(W))
                    acc[s]["extreme"].append(float(np.mean((e < 0.02) | (e > 0.98))))
                    acc[s]["var_resid"].append(float(np.var(W - ec)))
            base = np.mean(acc["Z"]["var_resid"])
            res[scen] = {
                s: {
                    "W_auc": float(np.mean(v["auc"])),
                    "W_r2": float(np.mean(v["r2"])),
                    "e_extreme_share": float(np.mean(v["extreme"])),
                    "var_resid_W": float(np.mean(v["var_resid"])),
                    "var_resid_ratio_vs_Z": float(np.mean(v["var_resid"]) / base),
                }
                for s, v in acc.items()
            }
            print(f"[absorb] {scen} done", file=sys.stderr, flush=True)
    return res


def y_metrics(encoder: str) -> dict:
    out: dict = {}
    for spec in SCENARIOS:
        Zpca, Mfull, y, names = load_aligned(encoder, C.PCA_K, spec=spec)
        M, mn, S, sn = _split(Mfull, names)
        yi = y.astype(int)
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
        auc = {}
        for s, X in _sets(Zpca, M, S).items():
            clf = RandomForestClassifier(n_estimators=C.RF_N_ESTIMATORS, random_state=SEED, n_jobs=1)
            p = cross_val_predict(clf, X, yi, cv=cv, method="predict_proba")[:, 1]
            auc[s] = float(roc_auc_score(yi, p))

        def top(X, nm):
            rho = [(nm[j], float(spearmanr(X[:, j], yi).correlation)) for j in range(X.shape[1]) if X[:, j].std() > 0]
            return sorted(rho, key=lambda t: -abs(t[1]))[:5]

        out[spec] = {"y_auc": auc, "top5_spearman_M": top(M, mn), "top5_spearman_S": top(S, sn)}
    return out


def main() -> int:
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    encoder = "MiniLM_L6"
    payload = {"encoder": encoder, "R_windows": R, "n_doc": C.N_DOC, "clip_e": CLIP,
               "windows": window_metrics(R, encoder), "y": y_metrics(encoder)}
    path = OUT_DIR / "absorption_evidence_amazon.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
