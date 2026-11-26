#!/usr/bin/env python3
"""Plot RMT / geometry stats vs n_pc (no power curves)."""

from __future__ import annotations

import csv
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


def main() -> None:
    stats = read_csv(OUT / "rmt_geometry_sweep.csv")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    by_lab: dict[str, list] = defaultdict(list)
    for r in stats:
        lab = f"{r['dataset']}|{r['encoder']}|{r['variant']}"
        by_lab[lab].append(r)
    for ax, met in zip(axes.ravel(), STAT_METRICS):
        for lab, pts in sorted(by_lab.items()):
            pts = sorted(pts, key=lambda x: int(float(x["n_pc"])))
            ax.plot(
                [int(float(p["n_pc"])) for p in pts],
                [float(p[met]) for p in pts],
                marker="o",
                ms=2,
                label=lab,
            )
        ax.set_title(met)
        ax.set_xlabel("n_pc (linspace 30–330, 30 pts)")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=4, ncol=1, loc="upper left", bbox_to_anchor=(0, 1.35))
    fig.suptitle("Embedding geometry / RMT vs PCA dimension (cache-only)")
    fig.tight_layout()
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEDIA / "pc_grid_rmt_stats.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / "pc_grid_rmt_stats.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {MEDIA / 'pc_grid_rmt_stats.png'}")


if __name__ == "__main__":
    main()