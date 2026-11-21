# Reference: text back-translation embedding benchmark (user upload).
# Kept for the covariate_backtrans design analogue — not executed as-is here.
# See expand_shift_board.py / EVALUATION_FRAMEWORK.md.
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from benchmark_textembedding_additional import benchmark_method

DEFAULT_CONFIG = {
    "alpha": 0.05,
    "seed": 2026,
    "n_perm_cf": 2,
    "n_perm_cate": 2,
    "n_perm_rf": 2,
    "n_perm_drperm": 2,
    "n_perm_rrperm": 2,
    "n_perm_tst": 2,
    "n_conformal": 2,
    "n_splits": 5,
    "kernel": "rbf",
    "outcome_model": "rf_regressor",
    "ps_model": "rf_classifier",
    "tau_model": "rf_regressor",
    "nu_model": "rf_regressor",
    "xgbperm_model": "xgb_regressor",
    "run_cfperm_loco": False,
    "run_cfperm_permu": False,
    "run_cfperm_grf": False,
    "run_rfperm": True,
    "run_xgbperm": True,
    "run_conformal": True,
}


def main():
    ap = argparse.ArgumentParser(description="Translation embedding benchmark")
    ap.add_argument("--emb1", type=str, required=True)
    ap.add_argument("--emb2", type=str, required=True)
    ap.add_argument("--data-trans", type=str, required=True)
    ap.add_argument("--B", type=int, default=75, help="Number of bootstrap repetitions")
    ap.add_argument("--response", type=str, default="rating")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--pc", type=int, default=30)
    ap.add_argument("--n-rows", type=int, default=500)
    ap.add_argument("--output-dir", type=str, default="benchmark_results")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    cfg = DEFAULT_CONFIG.copy()
    cfg["seed"] = args.seed
    df_join = pd.read_csv(args.data_trans)
    df_emb1 = np.asarray(np.load(args.emb1), dtype=float)
    df_emb2 = np.asarray(np.load(args.emb2), dtype=float)
    if df_emb1.shape != df_emb2.shape:
        raise ValueError(f"emb1 shape {df_emb1.shape} != emb2 shape {df_emb2.shape}")
    n_pc = min(args.pc, df_emb1.shape[0], df_emb1.shape[1])
    pca = PCA(n_components=n_pc, random_state=args.seed)
    df_emb1 = pca.fit_transform(df_emb1)
    df_emb2 = pca.fit_transform(df_emb2)
    n = min(df_emb1.shape[0], df_join.shape[0])
    df_join = df_join.iloc[:n, :]
    y = df_join[args.response].astype(float).to_numpy()
    df1 = np.column_stack([df_emb1, y])
    df2 = np.column_stack([df_emb2, y])
    n_sample = min(args.n_rows, n)
    idx1 = rng.choice(n, size=n_sample, replace=True)
    idx2 = rng.choice(n, size=n_sample, replace=True)
    df1 = df1[idx1]
    df2 = df2[idx2]
    summary_df, raw_df = benchmark_method(df1, df2, B=args.B, config=cfg)
    os.makedirs(args.output_dir, exist_ok=True)
    emb1_name = os.path.basename(args.emb1).replace(".npy", "")
    emb2_name = os.path.basename(args.emb2).replace(".npy", "")
    summary_path = (
        f"{args.output_dir}/{emb1_name}_with_comparison_{emb2_name}_testing_results.csv"
    )
    raw_path = summary_path.replace(".csv", "_raw.csv")
    summary_df.to_csv(summary_path, index=False)
    raw_df.to_csv(raw_path, index=False)
    print(f"Saved {summary_path}", flush=True)
    print(summary_df.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
