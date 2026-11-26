#!/usr/bin/env python3
"""DRPerm-only rerun with the residual-permutation null.

Usage: python3 -m experiments.meta_vs_embed_power.run_dr_rerun DATASET ARM[,ARM] [SCENARIO,...]
Env: POWER_N_JOBS (default all CPUs), POWER_B (default 50), POWER_NULL (default residual).
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from joblib import Parallel, delayed

from experiments.meta_vs_embed_power import config as C
from experiments.meta_vs_embed_power.perm_lib import drperm_pvalue
from experiments.meta_vs_embed_power.pipeline import OUT, prepare_dataset
from experiments.meta_vs_embed_power.power import make_windows

SCENARIOS = ["sentiment_meta", "syntax_meta", "train_vs_heldout", "null_shift"]


def _arm_x(arm, Ze, Zn, Xe_b, Xn_b, Me, Mn):
    if arm == "without_meta":
        return Ze, Zn
    if arm == "with_meta":
        return Xe_b, Xn_b
    if arm == "meta_only":
        return Me, Mn
    raise ValueError(arm)


def one(ds_cache, scenario, arm, b, *, n_perm, null_scheme, base_seed):
    Z, M, y, names = ds_cache
    seed = base_seed + 10_007 * SCENARIOS.index(scenario) + b
    rng = np.random.default_rng(seed)
    Ze, ye, Zn, yn, Xe_b, Xn_b, Me, Mn = make_windows(scenario, Z, M, y, names, C.N_DOC, rng)
    Xe, Xn = _arm_x(arm, Ze, Zn, Xe_b, Xn_b, Me, Mn)
    X = np.vstack([Xe, Xn])
    Y = np.concatenate([ye, yn])
    W = np.r_[np.zeros(len(ye), int), np.ones(len(yn), int)]
    r = drperm_pvalue(X, Y, W, n_perm=n_perm, seed=seed + 3, n_estimators=C.RF_N_ESTIMATORS, null_scheme=null_scheme)
    return {"scenario": scenario, "feature_set": arm, "b": b, "p_value": r["p_value"],
            "reject": bool(r["reject"]), "statistic": r["statistic"], "abs_w_resid_mean": r["abs_w_resid_mean"]}


def main() -> int:
    ds = sys.argv[1]
    arms = sys.argv[2].split(",")
    scenarios = sys.argv[3].split(",") if len(sys.argv) > 3 else SCENARIOS
    B = int(os.environ.get("POWER_B", 50))
    null_scheme = os.environ.get("POWER_NULL", "residual")
    n_jobs = int(os.environ.get("POWER_N_JOBS", os.cpu_count() or 4))
    out_path = OUT / f"power_{ds}_B{B}_dr_{null_scheme}.json"
    state = json.loads(out_path.read_text(encoding="utf-8")) if out_path.is_file() else {"table": {}}
    cache = prepare_dataset(ds, n_max=C.ROW_CAP, seed=2026, pca_k=C.PCA_K, out_dir=OUT)
    for arm in arms:
        for sc in scenarios:
            key = f"{sc}/{arm}"
            if key in state["table"]:
                continue
            t0 = time.time()
            reps = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(one)(cache, sc, arm, b, n_perm=C.N_PERM, null_scheme=null_scheme, base_seed=2026)
                for b in range(B)
            )
            ps = np.array([r["p_value"] for r in reps])
            state["table"][key] = {
                "scenario": sc, "feature_set": arm, "B": B, "n_perm": C.N_PERM,
                "power_DR": float(np.mean(ps < 0.05)), "mean_p_DR": float(ps.mean()),
                "abs_w_resid_mean": float(np.mean([r["abs_w_resid_mean"] for r in reps])),
                "elapsed_sec": time.time() - t0,
            }
            state.update({"dataset": ds, "null_scheme": null_scheme, "outcome_model": C.OUTCOME_MODEL})
            out_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            e = state["table"][key]
            print(f"[dr {null_scheme} {ds} {key}] power={e['power_DR']:.2f} mean_p={e['mean_p_DR']:.2f} "
                  f"|W-e|={e['abs_w_resid_mean']:.2f} ({e['elapsed_sec']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
