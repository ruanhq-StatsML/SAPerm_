"""Demo: residual-interaction shift in a synthetic factor panel."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.dependence_strength_shift.factor_residuals import (
    factor_residual_dependence_shift,
    make_factor_residual_demo,
)
from experiments.dependence_strength_shift.metrics import ranking_recovery

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def main():
    demo = make_factor_residual_demo(n=450, p=6, seed=7)
    out = factor_residual_dependence_shift(
        demo["F_ref"],
        demo["F_new"],
        G_ref=demo["G_ref"],
        G_new=demo["G_new"],
        n_common=1,
        model_m="ridge_regressor",
        seed=11,
        compute_incremental=True,
    )
    active = demo["active"]
    # Oracle signed order on active residual channels: weaken < flip≈0 < strengthen
    oracle = np.zeros(demo["F_ref"].shape[1])
    oracle[0] = -0.4
    oracle[1] = 0.3
    oracle[2] = 0.0
    metrics = ranking_recovery(
        oracle[active],
        out["strength_shift"][active],
        k=2,
        active_mask=np.array([True, True, False]),
    )
    po_metrics = ranking_recovery(
        oracle[active],
        out["po_gap"][active],
        k=2,
        active_mask=np.array([True, True, False]),
    )

    lines = [
        "# Factor residual dependence shift — demo",
        "",
        "Common factor removed; DSS / PO-risk / pairwise residual corr on ε.",
        "",
        "| Factor | kind | A (strength) | PO-risk gap | mean |Δρ| |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for j in range(demo["F_ref"].shape[1]):
        lines.append(
            "| f{j} | {k} | {a:+.3f} | {p:.3f} | {c:.3f} |".format(
                j=j,
                k=demo["kinds"][j],
                a=float(out["strength_shift"][j]),
                p=float(out["po_gap"][j]),
                c=float(out["mean_abs_pairwise_corr_shift"][j]),
            )
        )
    lines.extend(
        [
            "",
            f"Active-block Spearman(A) strength: **{metrics['spearman']:.3f}**, "
            f"sign agree: **{metrics['sign_agreement']:.3f}**",
            f"Active-block Spearman(A) PO-risk: **{po_metrics['spearman']:.3f}**, "
            f"sign agree: **{po_metrics['sign_agreement']:.3f}**",
            "",
            "Incremental residual R² shift `V1-V0` (rows = target residual, cols = source):",
            "",
            "```",
            np.array2string(out["incremental_shift"], precision=3, suppress_small=True),
            "```",
            "",
        ]
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    board = "\n".join(lines)
    (RESULTS / "FACTOR_RESIDUALS.md").write_text(board)
    payload = {
        "strength_shift": out["strength_shift"].tolist(),
        "po_gap": out["po_gap"].tolist(),
        "mean_abs_pairwise_corr_shift": out["mean_abs_pairwise_corr_shift"].tolist(),
        "residual_corr_shift": out["residual_corr_shift"].tolist(),
        "incremental_shift": out["incremental_shift"].tolist(),
        "metrics_strength": metrics,
        "metrics_po": po_metrics,
        "kinds": [str(k) for k in demo["kinds"]],
    }
    (RESULTS / "factor_residuals_demo.json").write_text(json.dumps(payload, indent=2))
    print(board)


if __name__ == "__main__":
    main()
