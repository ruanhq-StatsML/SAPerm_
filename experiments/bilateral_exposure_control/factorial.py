"""2x2 factorial: candidate dynamic frequency x company dynamic capacity.

Uses *cluster* randomization over marketplace clusters (not independent
candidate assignment) because candidates and jobs compete for the same
traffic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .metrics import RunMetrics, merge_metrics
from .simulator import MarketplaceSim, SimConfig
from .types import ControlFlags


TREATMENTS = {
    "control": ControlFlags(candidate_rules=False, company_rules=False),
    "t1_candidate": ControlFlags(candidate_rules=True, company_rules=False),
    "t2_company": ControlFlags(candidate_rules=False, company_rules=True),
    "t3_both": ControlFlags(candidate_rules=True, company_rules=True),
}


@dataclass
class FactorialResult:
    by_arm: Dict[str, RunMetrics]
    table_md: str


def run_factorial(
    seeds: Optional[List[int]] = None,
    base: Optional[SimConfig] = None,
    results_dir: Optional[Path] = None,
) -> FactorialResult:
    seeds = seeds or [2026, 2027, 2028, 2029]
    base = base or SimConfig()
    by_arm: Dict[str, RunMetrics] = {}

    for name, flags in TREATMENTS.items():
        runs: List[RunMetrics] = []
        for s in seeds:
            cfg = SimConfig(**{**base.__dict__, "seed": s})
            sim = MarketplaceSim(cfg=cfg, flags=flags)
            runs.append(sim.run(seed=s))
        by_arm[name] = merge_metrics(runs)

    lines = [
        "# Bilateral exposure control — 2x2 factorial",
        "",
        "Cluster-randomized synthetic marketplace. Arms:",
        "",
        "| Arm | Candidate rules | Company rules |",
        "| --- | --- | --- |",
        "| control | no | no |",
        "| t1_candidate | yes | no |",
        "| t2_company | no | yes |",
        "| t3_both | yes | yes |",
        "",
        f"Seeds: {seeds}. Days={base.n_days}, candidates={base.n_candidates}, "
        f"companies={base.n_companies}.",
        "",
        "| Arm | Impr | Mkt apps | Interviews | Fills | Interv/1k impr | "
        "Fills/100 open-days | Unsub rate | Mean review wait | Net value |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, m in by_arm.items():
        lines.append(
            f"| {name} | {m.impressions} | {m.marketing_apps} | {m.interviews} | {m.fills} | "
            f"{m.interviews_per_1k_impr:.2f} | {m.fills_per_100_opening_days:.2f} | "
            f"{m.unsubscribe_rate:.4f} | {m.mean_review_wait:.2f} | {m.net_value:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Read",
            "",
            "- Prefer higher interviews per 1k impressions and fills per 100 opening-days.",
            "- Prefer lower unsubscribe rate and mean review wait.",
            "- `t2_company` / `t3_both` should cut waste when backlog is high "
            "(Backlog Control + Pacing).",
            "- `t1_candidate` / `t3_both` should cut fatigue-driven unsubscribes.",
            "",
            "## Next phases",
            "",
            "1. LightGBM fatigue / capacity predictions replace fixed caps.",
            "2. DeepHit fill probabilities enter the DP state, not the rank score.",
            "3. Causal Transformer for marginal reach value.",
            "4. Min-cost flow / DP under the same hard caps.",
            "",
        ]
    )
    table_md = "\n".join(lines)
    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "FACTORIAL.md").write_text(table_md + "\n", encoding="utf-8")
        # Also dump a flat csv-ish summary.
        csv_lines = [
            "arm,impressions,marketing_apps,interviews,fills,"
            "interviews_per_1k_impr,fills_per_100_opening_days,"
            "unsubscribe_rate,mean_review_wait,net_value"
        ]
        for name, m in by_arm.items():
            d = m.as_dict()
            csv_lines.append(
                ",".join(
                    [
                        name,
                        str(int(d["impressions"])),
                        str(int(d["marketing_apps"])),
                        str(int(d["interviews"])),
                        str(int(d["fills"])),
                        f"{d['interviews_per_1k_impr']:.6f}",
                        f"{d['fills_per_100_opening_days']:.6f}",
                        f"{d['unsubscribe_rate']:.6f}",
                        f"{d['mean_review_wait']:.6f}",
                        f"{d['net_value']:.6f}",
                    ]
                )
            )
        (results_dir / "FACTORIAL.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    return FactorialResult(by_arm=by_arm, table_md=table_md)
