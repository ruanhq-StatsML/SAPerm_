"""2x2: analyst attention rules × desk capacity rules (model signals on)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .metrics import RunMetrics, merge_metrics
from .simulator import DueDiligenceSim, SimConfig
from .types import ControlFlags


TREATMENTS = {
    "control": ControlFlags(False, False, True),
    "t1_analyst": ControlFlags(True, False, True),
    "t2_desk": ControlFlags(False, True, True),
    "t3_both": ControlFlags(True, True, True),
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
    seeds = seeds or [2026, 2027, 2028]
    base = base or SimConfig()
    by_arm: Dict[str, RunMetrics] = {}
    for name, flags in TREATMENTS.items():
        runs = []
        for s in seeds:
            cfg = SimConfig(**{**base.__dict__, "seed": s})
            runs.append(DueDiligenceSim(cfg=cfg, flags=flags).run(seed=s))
        by_arm[name] = merge_metrics(runs)

    lines = [
        "# Due-diligence exposure control — 2x2 factorial",
        "",
        "Synthetic desk. Finance model = heuristic risk lexicon (FinanceGPT/FinBERT pluggable).",
        "",
        "| Arm | Analyst rules | Desk rules | Model signals |",
        "| --- | --- | --- | --- |",
        "| control | no | no | yes |",
        "| t1_analyst | yes | no | yes |",
        "| t2_desk | no | yes | yes |",
        "| t3_both | yes | yes | yes |",
        "",
        f"Seeds: {seeds}. Days={base.n_days}, analysts={base.n_analysts}, deals={base.n_deals}.",
        "",
        "| Arm | Coverage | Flags | Confirmed | IC-ready | Confirmed/1k | "
        "IC/100 mandate-days | Mute rate | Mean review wait | Net value |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, m in by_arm.items():
        lines.append(
            f"| {name} | {m.coverage} | {m.flags} | {m.confirmed} | {m.ic_ready} | "
            f"{m.confirmed_per_1k_coverage:.2f} | {m.ic_per_100_mandate_days:.2f} | "
            f"{m.mute_rate:.4f} | {m.mean_review_wait:.2f} | {m.net_value:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Read",
            "",
            "- Higher confirmed risks per 1k coverage = less noisy reading.",
            "- Desk rules should cut review wait when data rooms flood.",
            "- Analyst rules should cut mute / fatigue.",
            "- Models rank *what* to read; rules decide *how much* and *when*.",
            "",
            "## FinanceGPT opportunity",
            "",
            "1. Replace heuristic with FinBERT first-pass + FinanceGPT/FinGPT memo QA.",
            "2. Keep backlog/pacing hard caps — model scores never bypass them.",
            "3. Fine-tune LoRA on labeled red flags + IC outcomes; plug via `PromptedRiskProvider`.",
            "4. Reuse FSDS lambda*w on sector congestion shares for even desk effort.",
            "",
        ]
    )
    table_md = "\n".join(lines)
    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "FACTORIAL.md").write_text(table_md + "\n", encoding="utf-8")
    return FactorialResult(by_arm=by_arm, table_md=table_md)
