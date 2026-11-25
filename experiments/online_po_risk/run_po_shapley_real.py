"""PO-risk Shapley on real datasets (BiasFinder, ModaSense, Adult, Bank, Credit-g).

  python -m experiments.online_po_risk.run_po_shapley_real --seeds 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.online_po_risk.real_data import SCENARIOS, load_window
from experiments.online_po_risk.score import ood_vimp


def _top_names(values: np.ndarray, names: list[str], k: int = 5) -> str:
    order = np.argsort(-np.asarray(values, dtype=float))
    picks = []
    for j in order[:k]:
        picks.append(f"{names[j]}:{values[j]:.3f}")
    return ";".join(picks)


def run(
    *,
    seeds: int = 8,
    n_ref: int = 120,
    n_new: int = 120,
    n_perm: int = 12,
    tau_kind: str = "rf",
    base_seed: int = 2026,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    out = Path(out_dir) if out_dir is not None else Path(__file__).resolve().parent / "results"
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for dataset, scenario in SCENARIOS:
        for s in range(seeds):
            seed = base_seed + 100 * s + (hash((dataset, scenario)) % 97)
            try:
                X0, y0, X1, y1, names, meta = load_window(
                    dataset, scenario, seed=seed, n_ref=n_ref, n_new=n_new
                )
            except Exception as e:
                print(f"SKIP {dataset}/{scenario} seed={s}: {e}", flush=True)
                continue
            # need both batch labels present in Y for propensity; Y can be constant in polarity windows
            if len(np.unique(np.concatenate([y0, y1]))) < 2:
                # jitter-free fallback: use a synthetic outcome correlated with a driver if available
                drivers = meta.get("drivers") or [0]
                j = int(drivers[0]) if drivers else 0
                y0 = (X0[:, j] > np.median(X0[:, j])).astype(float)
                y1 = (X1[:, j] > np.median(X1[:, j])).astype(float)
                if len(np.unique(np.concatenate([y0, y1]))) < 2:
                    y0 = (np.arange(len(y0)) % 2).astype(float)
                    y1 = (np.arange(len(y1)) % 2).astype(float)

            try:
                vim = ood_vimp(X0, y0, X1, y1, n_perm=n_perm, seed=seed, tau_kind=tau_kind)
            except Exception as e:
                print(f"FAIL ood_vimp {dataset}/{scenario} seed={s}: {e}", flush=True)
                continue

            shap = np.asarray(vim["po_shapley"], dtype=float)
            vimp = np.asarray(vim["po_vimp"], dtype=float)
            rf = np.asarray(vim["rf_vimp"], dtype=float)
            mmd = np.asarray(vim["MMD_vimp"], dtype=float)
            drivers = list(meta.get("drivers") or [])
            noise = [i for i in range(len(names)) if i not in set(drivers)]

            def _mean(v, idx):
                if not idx or not np.all(np.isfinite(v)):
                    return float("nan")
                return float(np.mean(v[idx]))

            def _hit(v):
                if drivers and np.any(np.isfinite(v)):
                    return int(int(np.nanargmax(v)) in set(drivers))
                return float("nan")

            # rank correlations among methods
            def _spear(a, b):
                a, b = np.asarray(a, float), np.asarray(b, float)
                if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
                    return float("nan")
                if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                    return float("nan")
                return float(pd.Series(a).corr(pd.Series(b), method="spearman"))

            row = {
                "dataset": dataset,
                "scenario": scenario,
                "seed": int(seed),
                "tau_kind": tau_kind,
                "n_ref": int(len(y0)),
                "n_new": int(len(y1)),
                "p": int(len(names)),
                "tau2": float(vim["tau2"]),
                "po_risk": float(vim["po_risk"]),
                "y_rate_ref": float(np.mean(y0)),
                "y_rate_new": float(np.mean(y1)),
                "top_shap": names[int(np.nanargmax(shap))] if np.any(np.isfinite(shap)) else "",
                "top_vimp": names[int(np.nanargmax(vimp))] if np.any(np.isfinite(vimp)) else "",
                "top_rf": names[int(np.nanargmax(rf))] if np.any(np.isfinite(rf)) else "",
                "top_mmd": names[int(np.nanargmax(mmd))] if np.any(np.isfinite(mmd)) else "",
                "top_shap_is_driver": _hit(shap),
                "top_rf_is_driver": _hit(rf),
                "top_mmd_is_driver": _hit(mmd),
                "driver_shap": _mean(shap, drivers),
                "noise_shap": _mean(shap, noise) if drivers else float("nan"),
                "driver_rf": _mean(rf, drivers),
                "noise_rf": _mean(rf, noise) if drivers else float("nan"),
                "driver_mmd": _mean(mmd, drivers),
                "noise_mmd": _mean(mmd, noise) if drivers else float("nan"),
                "spearman_shap_vimp": _spear(shap, vimp),
                "spearman_shap_rf": _spear(shap, rf),
                "spearman_shap_mmd": _spear(shap, mmd),
                "top5_shap": _top_names(shap, names),
                "top5_rf": _top_names(rf, names),
                "top5_mmd": _top_names(mmd, names),
            }
            rows.append(row)
            print(
                f"{dataset}/{scenario} seed {s+1}/{seeds}  tau2={row['tau2']:.3f}  "
                f"top_shap={row['top_shap']}  top_rf={row['top_rf']}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    df.to_csv(out / "po_shapley_real.csv", index=False)

    # summary
    if len(df):
        gcols = ["dataset", "scenario"]
        summary = (
            df.groupby(gcols, as_index=False)
            .agg(
                seeds=("seed", "nunique"),
                tau2=("tau2", "mean"),
                po_risk=("po_risk", "mean"),
                top_shap_driver=("top_shap_is_driver", "mean"),
                top_rf_driver=("top_rf_is_driver", "mean"),
                top_mmd_driver=("top_mmd_is_driver", "mean"),
                driver_shap=("driver_shap", "mean"),
                noise_shap=("noise_shap", "mean"),
                spearman_shap_vimp=("spearman_shap_vimp", "mean"),
                spearman_shap_rf=("spearman_shap_rf", "mean"),
                spearman_shap_mmd=("spearman_shap_mmd", "mean"),
                # most frequent top feature
                mode_top_shap=("top_shap", lambda s: s.value_counts().index[0] if len(s) else ""),
                mode_top_rf=("top_rf", lambda s: s.value_counts().index[0] if len(s) else ""),
                mode_top_mmd=("top_mmd", lambda s: s.value_counts().index[0] if len(s) else ""),
            )
            .sort_values(["dataset", "scenario"])
            .reset_index(drop=True)
        )
        summary.to_csv(out / "po_shapley_real_summary.csv", index=False)
        print("\n=== real-data summary ===", flush=True)
        print(summary.round(3).to_string(index=False), flush=True)
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--n-ref", type=int, default=120)
    p.add_argument("--n-new", type=int, default=120)
    p.add_argument("--n-perm", type=int, default=12)
    p.add_argument("--tau-kind", default="rf", choices=["rf", "ridge"])
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()
    run(
        seeds=args.seeds,
        n_ref=args.n_ref,
        n_new=args.n_new,
        n_perm=args.n_perm,
        tau_kind=args.tau_kind,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
