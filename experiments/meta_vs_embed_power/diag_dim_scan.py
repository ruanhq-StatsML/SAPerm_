"""Per-k diagnostics for the gender dimension scan (MiniLM and Qwen).

* y 5-fold CV AUC on the 1000-row frame (RF classifier, 150 trees) for each arm's feature matrix.
* Per scenario, 10 paired windows (b=0..9): AUC_W and Var(W - e_hat) of the cross-fitted e(X)
  that DRPerm uses (2 folds, ``rf_classifier`` with 75 trees, clipped to [0.05, 0.95]).

    python3 -m experiments.meta_vs_embed_power.diag_dim_scan MiniLM_L6 qwen3-0.6b
"""

from __future__ import annotations

import json
import sys

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from experiments.meta_vs_embed_power import run_dim_scan as D
from experiments.meta_vs_embed_power.stats_utils import make_folds, rf_classifier

KS = ("24", "64", "128", "full")
KINDS = ("without", "with", "spacy", "with_decorr")
N_WIN = 10
PATH = D.OUT / "dim_scan_gender_diag.json"


def frame(kind: str, k: str, emb: str) -> np.ndarray:
    d = D.load_data(emb)
    Zk = D.z_k(k, emb)
    blk = {"without": None, "with": d["M"], "spacy": d["S"], "with_decorr": d["M"][:, d["M_keep"]]}[kind]
    return Zk if blk is None else np.hstack([Zk, blk])


def y_auc(kind: str, k: str, emb: str) -> float:
    X = frame(kind, k, emb)
    y = D.load_data(emb)["y"].astype(int)
    pred = np.zeros(len(y))
    for i, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=D.SEED).split(X, y)):
        m = rf_classifier(D.SEED + i, n_estimators=150).fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, pred))


def w_diag(kind: str, k: str, sc: str, b: int, emb: str) -> dict:
    Xe, ye, Xn, yn, _, seed = D.build(kind, k, sc, b, emb)
    X = np.vstack([Xe, Xn])
    W = np.r_[np.zeros(len(ye), dtype=int), np.ones(len(yn), dtype=int)]
    e = np.zeros(len(W))
    for f, te in enumerate(make_folds(len(W), 2, seed + 3)):
        tr = np.setdiff1d(np.arange(len(W)), te)
        clf = rf_classifier(seed + 3 + 100 + f, n_estimators=75).fit(X[tr], W[tr])
        e[te] = clf.predict_proba(X[te])[:, 1]
    e = np.clip(e, 0.05, 0.95)
    return {"auc_w": float(roc_auc_score(W, e)), "var_w_resid": float(np.var(W - e))}


def main() -> None:
    embs = sys.argv[1:] or [D.BASE_EMB]
    out = json.loads(PATH.read_text()) if PATH.is_file() else {}
    for emb in embs:
        D.load_data(emb)
        grid = [(kd, k) for k in KS for kd in KINDS]
        yres = Parallel(n_jobs=4)(delayed(y_auc)(kd, k, emb) for kd, k in grid)
        jobs = [(kd, k, sc, b) for kd, k in grid for sc in D.SCENARIOS for b in range(N_WIN)]
        wres = Parallel(n_jobs=4)(delayed(w_diag)(kd, k, sc, b, emb) for kd, k, sc, b in jobs)
        agg: dict = {}
        for (kd, k, sc, _), r in zip(jobs, wres):
            agg.setdefault(f"{kd}|{k}|{sc}", []).append(r)
        out[emb] = {
            "y_auc_5fold": {f"{kd}|{k}": v for (kd, k), v in zip(grid, yres)},
            "window": {key: {m: float(np.mean([x[m] for x in v])) for m in ("auc_w", "var_w_resid")}
                       for key, v in agg.items()},
            "n_windows": N_WIN,
        }
        PATH.write_text(json.dumps(out, indent=1, sort_keys=True))
        print(f"[diag] {emb} done", flush=True)


if __name__ == "__main__":
    main()
