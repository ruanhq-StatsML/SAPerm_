"""Fuller PO-risk Shapley / three-way OOD attribution sweep.

Writes under ``results/``:
  - ``po_shapley_window.csv`` — one ref/new window per seed × scenario × phase
  - ``po_shapley_window_summary.csv``
  - ``online_po_shapley.csv`` — sliding-window Shapley of ``tau``
  - ``online_po_shapley_summary.csv``
  - ``po_shapley_feature_means.csv`` — mean attribution by feature (post only)

  python -m experiments.online_po_risk.run_po_shapley --seeds 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.online_po_risk.score import ood_vimp, online_po_shapley
from experiments.online_po_risk.stream import SIGNAL_IDX, linear_stream

COV_DRIVER = 0  # covariate stream shifts coordinate 0


def _driver_set(scenario: str) -> np.ndarray:
    if scenario == "concept":
        return np.asarray(SIGNAL_IDX, dtype=int)
    if scenario == "covariate":
        return np.asarray([COV_DRIVER], dtype=int)
    raise ValueError(scenario)


def window_rows(
    *,
    seeds: int,
    n: int,
    shift_at: int,
    ref_n: int,
    new_n: int,
    n_perm: int,
    base_seed: int,
    tau_kinds: tuple[str, ...],
) -> list[dict]:
    rows: list[dict] = []
    for scenario in ("concept", "covariate"):
        drivers = set(_driver_set(scenario).tolist())
        for s in range(seeds):
            seed = base_seed + 1000 * s + (0 if scenario == "concept" else 17)
            x, y = linear_stream(n=n, shift_at=shift_at, scenario=scenario, rng=seed)
            phases = {
                "pre": (0, ref_n, ref_n, ref_n + new_n),
                "post": (0, ref_n, shift_at, shift_at + new_n),
            }
            for phase, (a0, a1, b0, b1) in phases.items():
                for tau_kind in tau_kinds:
                    out = ood_vimp(
                        x[a0:a1],
                        y[a0:a1],
                        x[b0:b1],
                        y[b0:b1],
                        n_perm=n_perm,
                        seed=seed + (11 if phase == "pre" else 29),
                        tau_kind=tau_kind,
                    )
                    shap = np.asarray(out["po_shapley"], dtype=float)
                    vimp = np.asarray(out["po_vimp"], dtype=float)
                    rf = np.asarray(out["rf_vimp"], dtype=float)
                    mmd = np.asarray(out["MMD_vimp"], dtype=float)
                    p = len(shap)
                    noise = [j for j in range(p) if j not in drivers]
                    drive = list(drivers)

                    def _mean(v, idx):
                        return float(np.mean(v[idx])) if idx and np.all(np.isfinite(v)) else float("nan")

                    rows.append(
                        {
                            "scenario": scenario,
                            "phase": phase,
                            "seed": int(seed),
                            "tau_kind": tau_kind,
                            "tau2": float(out["tau2"]),
                            "po_risk": float(out["po_risk"]),
                            "top_shap": int(np.nanargmax(shap)) if np.any(np.isfinite(shap)) else -1,
                            "top_vimp": int(np.nanargmax(vimp)) if np.any(np.isfinite(vimp)) else -1,
                            "top_rf": int(np.nanargmax(rf)) if np.any(np.isfinite(rf)) else -1,
                            "top_mmd": int(np.nanargmax(mmd)) if np.any(np.isfinite(mmd)) else -1,
                            "top_shap_is_driver": int(
                                int(np.nanargmax(shap)) in drivers if np.any(np.isfinite(shap)) else 0
                            ),
                            "top_rf_is_driver": int(
                                int(np.nanargmax(rf)) in drivers if np.any(np.isfinite(rf)) else 0
                            ),
                            "top_mmd_is_driver": int(
                                int(np.nanargmax(mmd)) in drivers if np.any(np.isfinite(mmd)) else 0
                            ),
                            "driver_shap": _mean(shap, drive),
                            "noise_shap": _mean(shap, noise),
                            "driver_vimp": _mean(vimp, drive),
                            "noise_vimp": _mean(vimp, noise),
                            "driver_rf": _mean(rf, drive),
                            "noise_rf": _mean(rf, noise),
                            "driver_mmd": _mean(mmd, drive),
                            "noise_mmd": _mean(mmd, noise),
                            **{f"shap_x{j}": float(shap[j]) for j in range(p)},
                            **{f"vimp_x{j}": float(vimp[j]) for j in range(p)},
                            **{f"rf_x{j}": float(rf[j]) for j in range(p)},
                            **{f"mmd_x{j}": float(mmd[j]) for j in range(p)},
                        }
                    )
            print(f"window {scenario} seed {s + 1}/{seeds}", flush=True)
    return rows


def slide_rows(
    *,
    seeds: int,
    n: int,
    shift_at: int,
    window: int,
    step: int,
    n_perm: int,
    base_seed: int,
    tau_kind: str,
) -> list[dict]:
    rows: list[dict] = []
    for scenario in ("concept", "covariate"):
        drivers = _driver_set(scenario)
        for s in range(seeds):
            seed = base_seed + 1000 * s + (0 if scenario == "concept" else 17)
            x, y = linear_stream(n=n, shift_at=shift_at, scenario=scenario, rng=seed)
            for item in online_po_shapley(
                x,
                y,
                window=window,
                step=step,
                seed=seed,
                signal_idx=drivers,
                tau_kind=tau_kind,
                n_perm=n_perm,
            ):
                item = dict(item)
                item["scenario"] = scenario
                item["seed"] = int(seed)
                item["shifted"] = int(item["m"] > shift_at)
                item["tau_kind"] = tau_kind
                # rename signal_* → driver_* for covariate clarity
                item["driver_shapley"] = item.pop("signal_shapley")
                item["noise_shapley"] = item["noise_shapley"]
                item["top_is_driver"] = item.pop("top_is_signal")
                rows.append(item)
            print(f"slide {scenario} seed {s + 1}/{seeds}", flush=True)
    return rows


def summarize_windows(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["scenario", "phase", "tau_kind"]
    return (
        df.groupby(keys, as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            tau2=("tau2", "mean"),
            po_risk=("po_risk", "mean"),
            top_shap_driver=("top_shap_is_driver", "mean"),
            top_rf_driver=("top_rf_is_driver", "mean"),
            top_mmd_driver=("top_mmd_is_driver", "mean"),
            driver_shap=("driver_shap", "mean"),
            noise_shap=("noise_shap", "mean"),
            driver_vimp=("driver_vimp", "mean"),
            noise_vimp=("noise_vimp", "mean"),
            driver_rf=("driver_rf", "mean"),
            noise_rf=("noise_rf", "mean"),
            driver_mmd=("driver_mmd", "mean"),
            noise_mmd=("noise_mmd", "mean"),
        )
        .sort_values(["scenario", "tau_kind", "phase"])
        .reset_index(drop=True)
    )


def summarize_slides(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["phase"] = np.where(df["shifted"] == 1, "post", "pre")
    return (
        df.groupby(["scenario", "phase", "tau_kind"], as_index=False)
        .agg(
            n_windows=("m", "size"),
            seeds=("seed", "nunique"),
            top_driver_rate=("top_is_driver", "mean"),
            driver_shap=("driver_shapley", "mean"),
            noise_shap=("noise_shapley", "mean"),
            tau2=("tau2", "mean"),
            po_risk=("po_risk", "mean"),
        )
        .sort_values(["scenario", "tau_kind", "phase"])
        .reset_index(drop=True)
    )


def feature_means(df: pd.DataFrame) -> pd.DataFrame:
    """Post-phase mean SHAP / VIMP / RF / MMD by feature."""
    post = df[df["phase"] == "post"].copy()
    records = []
    for (scenario, tau_kind), block in post.groupby(["scenario", "tau_kind"]):
        drivers = set(_driver_set(scenario).tolist())
        shap_cols = sorted([c for c in block.columns if c.startswith("shap_x")], key=lambda c: int(c.split("x")[1]))
        for c in shap_cols:
            j = int(c.split("x")[1])
            records.append(
                {
                    "scenario": scenario,
                    "tau_kind": tau_kind,
                    "feature": f"x{j}",
                    "is_driver": int(j in drivers),
                    "po_shapley": float(block[c].mean()),
                    "po_vimp": float(block[f"vimp_x{j}"].mean()),
                    "rf_vimp": float(block[f"rf_x{j}"].mean()),
                    "MMD_vimp": float(block[f"mmd_x{j}"].mean()),
                }
            )
    return pd.DataFrame(records)


def run(
    out_dir: Path | None = None,
    seeds: int = 20,
    n: int = 280,
    shift_at: int = 140,
    ref_n: int = 50,
    new_n: int = 50,
    window: int = 60,
    step: int = 20,
    n_perm: int = 15,
    base_seed: int = 2026,
) -> None:
    out = Path(out_dir) if out_dir is not None else Path(__file__).resolve().parent / "results"
    out.mkdir(parents=True, exist_ok=True)

    win = pd.DataFrame(
        window_rows(
            seeds=seeds,
            n=n,
            shift_at=shift_at,
            ref_n=ref_n,
            new_n=new_n,
            n_perm=n_perm,
            base_seed=base_seed,
            tau_kinds=("rf", "ridge"),
        )
    )
    win.to_csv(out / "po_shapley_window.csv", index=False)
    win_sum = summarize_windows(win)
    win_sum.to_csv(out / "po_shapley_window_summary.csv", index=False)
    feat = feature_means(win)
    feat.to_csv(out / "po_shapley_feature_means.csv", index=False)

    slide = pd.DataFrame(
        slide_rows(
            seeds=seeds,
            n=n,
            shift_at=shift_at,
            window=window,
            step=step,
            n_perm=max(8, n_perm // 2),
            base_seed=base_seed,
            tau_kind="rf",
        )
    )
    slide.to_csv(out / "online_po_shapley.csv", index=False)
    slide_sum = summarize_slides(slide)
    slide_sum.to_csv(out / "online_po_shapley_summary.csv", index=False)

    print("\n=== window summary ===", flush=True)
    print(win_sum.round(3).to_string(index=False), flush=True)
    print("\n=== feature means (post) ===", flush=True)
    print(feat.round(3).to_string(index=False), flush=True)
    print("\n=== slide summary ===", flush=True)
    print(slide_sum.round(3).to_string(index=False), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--n-perm", type=int, default=15)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()
    run(out_dir=args.out_dir, seeds=args.seeds, n_perm=args.n_perm)


if __name__ == "__main__":
    main()
