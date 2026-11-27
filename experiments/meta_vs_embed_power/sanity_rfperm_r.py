#!/usr/bin/env python3
"""Quick Type-I + planted shift check for R RFPerm backend."""

from __future__ import annotations

import os

import numpy as np

os.environ["RF_PERM_BACKEND"] = "rfperm_r"
os.environ.setdefault("R_LIBS_USER", os.path.expanduser("~/R/library"))

from experiments.meta_vs_embed_power.pipeline import OUT, prepare_dataset
from experiments.meta_vs_embed_power.power import make_windows, run_one_rep


def main() -> int:
    Zpca, M, y, meta_names = prepare_dataset(
        "gender", n_max=1000, seed=2026, pca_k=24, out_dir=OUT, embed_model="MiniLM_L6"
    )
    checks = []
    for scenario, want in (("null_shift", "typeI"), ("sentiment_meta", "power")):
        rng = np.random.default_rng(99)
        reps = []
        for b in range(20):
            rep = run_one_rep(
                scenario, "without_meta", Zpca, M, y, meta_names,
                n_doc=80, n_perm=150, seed=2026 + b, n_estimators=150, pca_k=24,
            )
            reps.append(rep["RFPerm"]["p_value"])
        rate = float(np.mean([p < 0.05 for p in reps]))
        checks.append((scenario, want, rate, float(np.mean(reps))))
        print(f"{scenario} {want}: reject_rate={rate:.2f} mean_p={np.mean(reps):.3f}")
    ok_null = checks[0][2] <= 0.2
    ok_shift = checks[1][2] >= 0.35
    print("SANITY", "PASS" if ok_null and ok_shift else "WARN", checks)
    return 0 if ok_null and ok_shift else 1


if __name__ == "__main__":
    raise SystemExit(main())