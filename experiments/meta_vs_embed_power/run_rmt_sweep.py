#!/usr/bin/env python3
"""RMT / Lumley on cached embeddings; PC grid from config.N_PC_GRID."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments.meta_vs_embed_power import config as C
from experiments.meta_vs_embed_power.data import load_gender, load_train_bias_finder_tab
from experiments.meta_vs_embed_power.rmt_geometry import concat_pca24, covariance_stats, pca_scores

OUT = Path(__file__).resolve().parent / "results"
STAT_METRICS = (
    "anisotropy",
    "effective_rank",
    "rmt_signal_rank",
    "rmt_bbp_snr",
    "lumley_intensity",
    "rmt_signal_energy_ratio",
)


def _pc_grid(Z: np.ndarray) -> list[int]:
    n, p = Z.shape
    cap = min(n - 1, p)
    return sorted({int(min(k, cap)) for k in C.N_PC_GRID if k <= cap})


def _load_gender_bias() -> list[dict]:
    out = []
    for name, loader, emb_file in [
        ("gender", load_gender, OUT / "gender_MiniLM_L6_embedding.npy"),
        ("bias", load_train_bias_finder_tab, OUT / "bias_MiniLM_L6_embedding.npy"),
    ]:
        if not emb_file.is_file():
            continue
        _, _, texts = loader(C.ROW_CAP, 2026)
        meta_npz = OUT / f"meta_{name}_n1000.npz"
        if not meta_npz.is_file():
            continue
        z = np.load(meta_npz, allow_pickle=True)
        M, meta_names = z["M"], list(z["meta_names"])
        spacy_cache = OUT / f"spacy_{name}_n1000.npz"
        if spacy_cache.is_file():
            S = np.load(spacy_cache, allow_pickle=True)["S"]
        else:
            from experiments.meta_vs_embed_power.spacy_features import spacy_matrix

            print(f"spacy {name}...", file=sys.stderr)
            S, sn = spacy_matrix(texts)
            np.savez_compressed(spacy_cache, S=S, spacy_names=np.array(sn, dtype=object))
        out.append({"dataset": name, "encoder": "MiniLM_L6", "spec": "-", "Z": np.load(emb_file), "M": M, "S": S})
    return out


def _load_amazon() -> list[dict]:
    from experiments.meta_vs_embed_power import amazon_extract as ax

    rows = []
    for spec in ("gift", "cross"):
        if not ax.meta_path(spec).exists():
            continue
        meta = np.load(ax.meta_path(spec), allow_pickle=True)
        M = meta["M"]
        S = None
        if ax.spacy_path(spec).exists():
            S = np.load(ax.spacy_path(spec), allow_pickle=True)["S"]
        for enc in ("MiniLM_L6", "qwen3-0.6b"):
            ep = ax.emb_path(enc, spec)
            if not ep.exists():
                continue
            rows.append(
                {
                    "dataset": f"amazon_{spec}",
                    "encoder": enc,
                    "spec": spec,
                    "Z": np.load(ep, allow_pickle=True)["Z"],
                    "M": M,
                    "S": S,
                }
            )
    return rows


def sweep_item(item: dict, seed: int) -> list[dict]:
    Z = item["Z"]
    recs = []
    for k in _pc_grid(Z):
        for variant, X in _variants(Z, item, k, seed):
            st = covariance_stats(X)
            recs.append(
                {
                    "dataset": item["dataset"],
                    "encoder": item["encoder"],
                    "spec": item["spec"],
                    "variant": variant,
                    "n_pc": k,
                    **st,
                }
            )
    return recs


def _variants(Z, item, k: int, seed: int) -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = [("without_meta", pca_scores(Z, k, seed))]
    if item["M"] is not None:
        out.append(("with_meta", concat_pca24(Z, item["M"], seed=seed, pca_k=k)))
    if item["S"] is not None:
        out.append(("with_spacy", concat_pca24(Z, item["S"], seed=seed, pca_k=k)))
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def plot_stats(rows: list[dict], out_png: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    labels = sorted({f"{r['dataset']}|{r['encoder']}|{r['variant']}" for r in rows})
    for ax, metric in zip(axes.ravel(), STAT_METRICS):
        for lab in labels:
            ds, enc, var = lab.split("|", 2)
            sub = [r for r in rows if f"{r['dataset']}|{r['encoder']}|{r['variant']}" == lab]
            sub = sorted(sub, key=lambda x: x["n_pc"])
            ax.plot([s["n_pc"] for s in sub], [float(s.get(metric, np.nan)) for s in sub], ms=3, label=lab)
        ax.set_title(metric)
        ax.set_xlabel("n_pc")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=5, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.25))
    fig.suptitle("Geometry / RMT vs n_pc")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-amazon", action="store_true")
    args = p.parse_args()
    items = _load_gender_bias()
    if not args.skip_amazon:
        items.extend(_load_amazon())
    rows: list[dict] = []
    for it in items:
        rows.extend(sweep_item(it, 2026))
    csv_p = OUT / "rmt_geometry_sweep.csv"
    write_csv(csv_p, rows)
    (OUT / "rmt_geometry_sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    from experiments.meta_vs_embed_power import plot_pc_sweep

    plot_pc_sweep.main()
    print(f"{csv_p} rows={len(rows)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())