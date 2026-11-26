#!/usr/bin/env python3
"""Plot RMT / geometry stats vs n_pc (no power curves)."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "results"
MEDIA = Path("/cursor/stores/bc-87419b75-0666-4144-88dc-9832a2b2d6ae/media")
STAT_METRICS = (
    "anisotropy",
    "effective_rank",
    "rmt_signal_rank",
    "rmt_bbp_snr",
    "lumley_intensity",
    "rmt_signal_energy_ratio",
)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


LABEL_MAP = {
    "gender|MiniLM_L6|without_meta": "gender · MiniLM · PCA(Z)",
    "gender|MiniLM_L6|with_meta": "gender · MiniLM · PCA+meta",
    "gender|MiniLM_L6|with_spacy": "gender · MiniLM · PCA+spacy",
    "bias|MiniLM_L6|without_meta": "bias · MiniLM · PCA(Z)",
    "bias|MiniLM_L6|with_meta": "bias · MiniLM · PCA+meta",
    "bias|MiniLM_L6|with_spacy": "bias · MiniLM · PCA+spacy",
    "amazon_gift|MiniLM_L6|without_meta": "amazon gift · MiniLM · PCA(Z)",
    "amazon_gift|MiniLM_L6|with_meta": "amazon gift · MiniLM · PCA+meta",
}


def _load_elbows() -> dict[str, float]:
    p = OUT / "effective_rank_elbow.json"
    if not p.is_file():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for row in data.get("elbow_by_curve", []):
        med = row.get("bootstrap_median")
        if med is None:
            continue
        lab = f"{row['dataset']}|{row['encoder']}|{row['variant']}"
        out[lab] = float(med)
    return out


def main() -> None:
    stats = read_csv(OUT / "rmt_geometry_sweep.csv")
    elbows = _load_elbows()
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    by_lab: dict[str, list] = defaultdict(list)
    line_colors: dict[str, str] = {}
    for r in stats:
        lab = f"{r['dataset']}|{r['encoder']}|{r['variant']}"
        by_lab[lab].append(r)
    handles, labels = [], []
    for ax, met in zip(axes.ravel(), STAT_METRICS):
        for lab, pts in sorted(by_lab.items()):
            pts = sorted(pts, key=lambda x: int(float(x["n_pc"])))
            (line,) = ax.plot(
                [int(float(p["n_pc"])) for p in pts],
                [float(p[met]) for p in pts],
                marker="o",
                ms=2,
                label=LABEL_MAP.get(lab, lab),
            )
            line_colors[lab] = line.get_color()
            if met == STAT_METRICS[0] and LABEL_MAP.get(lab, lab) not in labels:
                handles.append(line)
                labels.append(LABEL_MAP.get(lab, lab))
            if met == "effective_rank" and lab in elbows:
                k = elbows[lab]
                ax.axvline(k, color=line.get_color(), ls="--", lw=0.9, alpha=0.85)
        if met == "effective_rank":
            ax.axvline(24, color="0.4", ls=":", lw=1.2, alpha=0.9, label="_spec PCA_K=24")
        ax.set_title(met)
        ax.set_xlabel("n_pc (30–330, 30 pts)")
        ax.grid(alpha=0.3)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        fontsize=8,
        framealpha=0.95,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle("Embedding geometry / RMT vs PCA (cache-only; no power curves)", y=1.06, fontsize=11)
    fig.subplots_adjust(top=0.88, hspace=0.35, wspace=0.28)
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEDIA / "pc_grid_rmt_stats.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / "pc_grid_rmt_stats.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {MEDIA / 'pc_grid_rmt_stats.png'}")


if __name__ == "__main__":
    main()