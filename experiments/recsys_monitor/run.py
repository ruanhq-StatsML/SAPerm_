"""Run the recsys OnlineRFPerm × CFPerm board."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.recsys_monitor.dgp import T_STAR
from experiments.recsys_monitor.pipeline import run_scenario, summarize_run

SCENARIOS = ("none", "concept", "covariate", "both")
SCENARIO_SEED = {"none": 0, "concept": 11, "covariate": 23, "both": 37}
RESULTS = Path(__file__).resolve().parent / "results"


def run_board(
    n_rep: int = 5,
    seed0: int = 2026,
    fdr: str = "addis",
    ewma: float | None = None,
    model_kind: str = "rf",
) -> pd.DataFrame:
    rows = []
    for scenario in SCENARIOS:
        for r in range(n_rep):
            df, tickets = run_scenario(
                scenario,
                seed=seed0 + 17 * r + SCENARIO_SEED[scenario],
                fdr=fdr,
                ewma_lambda=ewma,
                model_kind=model_kind,
                auto_refresh=True,
            )
            summary = summarize_run(df, scenario, t_star=T_STAR)
            summary["rep"] = r
            if tickets:
                summary["ticket_top3"] = ",".join(tickets[0].top_blocks)
            else:
                summary["ticket_top3"] = ""
            rows.append(summary)
    return pd.DataFrame(rows)


def aggregate(board: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, g in board.groupby("scenario"):
        rows.append(
            {
                "scenario": scenario,
                "detect_rate": float(g["detected"].mean()),
                "mean_delay": float(
                    np.nanmean(pd.to_numeric(g["delay"], errors="coerce"))
                    if g["detected"].any()
                    else float("nan")
                ),
                "mean_false_alarms": float(g["false_alarms"].mean()),
                "top_block_mode": (
                    g["top_block_at_detect"].replace("", np.nan).mode().iloc[0]
                    if g["top_block_at_detect"].replace("", np.nan).notna().any()
                    else ""
                ),
            }
        )
    order = {s: i for i, s in enumerate(SCENARIOS)}
    out = pd.DataFrame(rows)
    return out.sort_values("scenario", key=lambda s: s.map(order))


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.3f}" if np.isfinite(v) else "nan")
            else:
                cells.append(str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *body])


def write_board(summary: pd.DataFrame, detail: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# RecSys OnlineRFPerm × CFPerm board",
        "",
        "Synthetic rating stream. Reference = batch 0, burn-in = 4, planted",
        f"onset at batch `{T_STAR}`. FDR = ADDIS (α=0.05). Auto-refresh resets",
        "state after each reject (Appendix A.4).",
        "",
        "## Aggregate",
        "",
        _md_table(summary),
        "",
        "## Reading",
        "",
        "- `none`: false alarms should stay near zero (FDR sanity).",
        "- `concept`: item-taste flip — expect detection after onset; top block often `item` / `review`.",
        "- `covariate`: user-demo shift — top block often `user`.",
        "- `both`: either driver is acceptable; delay should stay small.",
        "",
        "## Per-replicate",
        "",
        "```json",
        json.dumps(detail.to_dict(orient="records"), indent=2, default=str),
        "```",
        "",
    ]
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-rep", type=int, default=4)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--fdr", default="addis", choices=["addis", "saffron", "lord", "fixed"])
    p.add_argument("--ewma", type=float, default=None)
    p.add_argument("--model", default="rf", choices=["rf", "ridge"])
    p.add_argument("--quick", action="store_true", help="1 replicate, ridge, faster")
    args = p.parse_args(argv)

    n_rep = 1 if args.quick else args.n_rep
    model = "ridge" if args.quick else args.model
    detail = run_board(
        n_rep=n_rep,
        seed0=args.seed,
        fdr=args.fdr,
        ewma=args.ewma,
        model_kind=model,
    )
    summary = aggregate(detail)
    RESULTS.mkdir(parents=True, exist_ok=True)
    detail.to_csv(RESULTS / "board_detail.csv", index=False)
    summary.to_csv(RESULTS / "board_summary.csv", index=False)
    write_board(summary, detail, RESULTS / "BOARD.md")
    print(summary.to_string(index=False))
    print(f"wrote {RESULTS / 'BOARD.md'}")


if __name__ == "__main__":
    main()
