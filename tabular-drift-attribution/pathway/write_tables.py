"""Build the pathway tables from results/pathway.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pathway.metrics import signflip_pvalue

ROOT = Path(__file__).resolve().parents[1]


def _abs_share(blocks, field):
    vals = {name: abs(row[field]) for name, row in blocks.items() if row[field] is not None}
    total = sum(vals.values()) + 1e-12
    return {name: vals[name] / total for name in vals}


def _fmt(value, digits=3):
    return f"{value:.{digits}f}"


def _ci(pair):
    return f"[{pair[0]:.3f}, {pair[1]:.3f}]"


def _signflip_table(pathway, fields):
    names = list(pathway[0]["blocks"])
    lines = []
    for field in fields:
        series = {
            name: np.array([abs(row["blocks"][name][field]) for row in pathway])
            for name in names
            if pathway[0]["blocks"][name][field] is not None
        }
        if not series:
            continue
        means = {name: float(values.mean()) for name, values in series.items()}
        top = max(means, key=means.get)
        for name in names:
            if name == top or name not in series:
                continue
            diff = series[top] - series[name]
            lines.append((field, top, name, float(diff.mean()), signflip_pvalue(diff)))
    return lines


def synthetic_pathway_rows(panel):
    rows = []
    for row in panel["pathway"]:
        po = _abs_share(row["blocks"], "po_drop")
        br = _abs_share(row["blocks"], "brier_drop")
        mse = _abs_share(row["blocks"], "mse_drop")
        mmd = _abs_share(row["blocks"], "mmd")
        rows.append(
            f"{row['step']} & {_fmt(row['po_risk'])} & {_fmt(row['brier'])} & {_fmt(row['mse'])} "
            f"& {_fmt(po['driver'])} & {_fmt(br['driver'])} & {_fmt(mse['driver'])} "
            f"& {_fmt(mmd['driver'])} & {_fmt(mmd['context'])} \\\\"
        )
    return "\n".join(rows)


def block_rows(panel, with_mse=False):
    last = panel["pathway"][-1]
    tests = panel["last_day_tests"]
    po = _abs_share(last["blocks"], "po_drop")
    br = _abs_share(last["blocks"], "brier_drop")
    lines = []
    for name, row in last["blocks"].items():
        mse = ""
        if with_mse:
            mse = f" & {_fmt(row['mse_drop'], 3)}"
        lines.append(
            f"{name} & {_fmt(row['po_drop'])} & {_ci(tests[name]['po_drop_ci'])} "
            f"& {_fmt(row['brier_drop'])} & {_ci(tests[name]['brier_drop_ci'])}{mse} "
            f"& {_fmt(row['mmd'])} & {_fmt(tests[name]['mmd_p'])} "
            f"& {_fmt(row['hsic_delta'], 4)} & {_fmt(tests[name]['hsic_p'])} "
            f"& {_fmt(po[name])} & {_fmt(br[name])} \\\\"
        )
    return "\n".join(lines)


def sign_rows(panel, fields):
    lines = []
    for field, top, other, mean, pvalue in _signflip_table(panel["pathway"], fields):
        label = {"po_drop": "PO-risk", "brier_drop": "Brier", "mse_drop": "MSE"}[field]
        lines.append(f"{label} & {top} $>$ {other} & {_fmt(mean)} & {_fmt(pvalue)} \\\\")
    return "\n".join(lines)


def main():
    payload = json.loads((ROOT / "results" / "pathway.json").read_text())
    synthetic = payload["synthetic"]
    ieee = payload["ieee_cis"]
    text = r"""
\begin{table}[t]
\centering
\caption{Synthetic pathway. The anchor is day 0. Each step is one day closer to the drift peak. Shares are the block's fraction of the total absolute movement.}
\label{tab:pathway-synthetic}
\begin{tabular}{rcccccccc}
\toprule
step & PO-risk & Brier & MSE & driver PO & driver Brier & driver MSE & driver MMD & context MMD \\
\midrule
""" + synthetic_pathway_rows(synthetic) + r"""
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering
\caption{Synthetic day 7, the drift peak. Deltas are the change in the loss when that block is replaced by the TabDDPM draw from the anchor conditional. Intervals are bootstrap percentiles. MMD and HSIC $p$-values use 40 permutations.}
\label{tab:pathway-synthetic-day7}
\small
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccccccc}
\toprule
block & PO $\Delta$ & PO interval & Brier $\Delta$ & Brier interval & MSE $\Delta$ & MMD & MMD $p$ & HSIC $\Delta$ & HSIC $p$ & PO share & Brier share \\
\midrule
""" + block_rows(synthetic, with_mse=True) + r"""
\bottomrule
\end{tabular}}
\end{table}

\begin{table}[t]
\centering
\caption{Sign-flip test over the seven synthetic steps. The contrast is the absolute movement of the driver minus the other block.}
\label{tab:pathway-synthetic-sign}
\begin{tabular}{llcc}
\toprule
loss & comparison & mean gap & $p$ \\
\midrule
""" + sign_rows(synthetic, ("po_drop", "brier_drop", "mse_drop")) + r"""
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering
\caption{IEEE-CIS transaction day 180, seven days after the fraud-rate peak on day 173. Same columns as the synthetic peak day. No latent MSE on this table.}
\label{tab:pathway-ieee-day}
\small
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccccccc}
\toprule
block & PO $\Delta$ & PO interval & Brier $\Delta$ & Brier interval & MMD & MMD $p$ & HSIC $\Delta$ & HSIC $p$ & PO share & Brier share \\
\midrule
""" + block_rows(ieee, with_mse=False) + r"""
\bottomrule
\end{tabular}}
\end{table}

\begin{table}[t]
\centering
\caption{Sign-flip test over the seven IEEE-CIS steps after the peak. PO-risk does not separate the blocks. The transferred Brier movement is largest for the card block.}
\label{tab:pathway-ieee-sign}
\begin{tabular}{llcc}
\toprule
loss & comparison & mean gap & $p$ \\
\midrule
""" + sign_rows(ieee, ("po_drop", "brier_drop")) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    # The synthetic pathway header has 9 columns but I listed 8 c's. Fix below by rewriting the header in the string carefully.
    out = ROOT / "results" / "pathway_tables.tex"
    out.write_text(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
