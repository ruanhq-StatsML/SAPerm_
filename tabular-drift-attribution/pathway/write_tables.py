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


def _cell(value, digits=3):
    if value is None:
        return "--"
    return _fmt(value, digits)


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


def leader_rows(panel):
    lines = []
    has_mse = panel["pathway"][0]["mse"] is not None
    for row in panel["pathway"]:
        def top(field):
            vals = {
                name: abs(block[field])
                for name, block in row["blocks"].items()
                if block[field] is not None
            }
            total = sum(vals.values())
            if total < 1e-4:
                return "--", None
            name = max(vals, key=vals.get)
            return name, vals[name] / total
        po_name, po_share = top("po_drop")
        br_name, br_share = top("brier_drop")
        mmd_name, mmd_share = top("mmd")
        mse_cell = ""
        if has_mse:
            mse_name, mse_share = top("mse_drop")
            mse_cell = f" & {mse_name} & {_cell(mse_share)}"
        lines.append(
            f"{row['label']} & {_fmt(row['fraud_rate'])} & {_fmt(row['po_risk'])} & {_fmt(row['brier'])} "
            f"& {po_name} & {_cell(po_share)} & {br_name} & {_cell(br_share)}{mse_cell} "
            f"& {mmd_name} & {_cell(mmd_share)} \\\\"
        )
    return "\n".join(lines), has_mse


def panel_tables(panel, caption_path, caption_day, label_prefix):
    leaders, has_mse = leader_rows(panel)
    mse_head = " & MSE top & MSE share" if has_mse else ""
    mse_cols = "cc" if has_mse else ""
    day = panel_tables_day(panel, has_mse)
    return f"""
\\begin{{table}}[t]
\\centering
\\caption{{{caption_path}}}
\\label{{tab:{label_prefix}-path}}
\\small
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{lccccccc{mse_cols}cc}}
\\toprule
step & rate & PO-risk & Brier & PO top & PO share & Brier top & Brier share{mse_head} & MMD top & MMD share \\\\
\\midrule
{leaders}
\\bottomrule
\\end{{tabular}}}}
\\end{{table}}

\\begin{{table}}[t]
\\centering
\\caption{{{caption_day}}}
\\label{{tab:{label_prefix}-day}}
\\small
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{{'lccccccccccc' if has_mse else 'lcccccccccc'}}}
\\toprule
block & PO $\\Delta$ & PO interval & Brier $\\Delta$ & Brier interval{' & MSE $\\Delta$' if has_mse else ''} & MMD & MMD $p$ & HSIC $\\Delta$ & HSIC $p$ & PO share & Brier share \\\\
\\midrule
{day}
\\bottomrule
\\end{{tabular}}}}
\\end{{table}}

\\begin{{table}}[t]
\\centering
\\caption{{Sign-flip test across the seven steps. The contrast is absolute movement.}}
\\label{{tab:{label_prefix}-sign}}
\\begin{{tabular}}{{llcc}}
\\toprule
loss & comparison & mean gap & $p$ \\\\
\\midrule
{sign_rows(panel, ('po_drop', 'brier_drop', 'mse_drop') if has_mse else ('po_drop', 'brier_drop'))}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def panel_tables_day(panel, has_mse):
    return block_rows(panel, with_mse=has_mse)


def sampler_lines(payload):
    lines = []
    titles = (
        ("synthetic", "synthetic"),
        ("ieee_cis", "IEEE"),
        ("movielens", "MovieLens"),
        ("home_credit", "Home Credit"),
    )
    for key, title in titles:
        panel = payload.get(key) or {}
        last = (panel.get("sampler_audit") or {}).get("last")
        if not last:
            continue
        for name, row in last.items():
            lines.append(
                f"{title} & {name} "
                f"& {_fmt(row['floor_po'])} & {_fmt(row['day_po'])} & {_fmt(row['excess_po'])} "
                f"& {_fmt(row['raw_po_share'])} & {_fmt(row['excess_po_share'])} "
                f"& {_fmt(row['floor_brier'])} & {_fmt(row['day_brier'])} & {_fmt(row['excess_brier'])} "
                f"& {_fmt(row['raw_brier_share'])} & {_fmt(row['excess_brier_share'])} "
                f"& {_fmt(row['anchor_impute_mmd'])} & {_fmt(row['day_covariate_mmd'])} "
                f"& {row['verdict']} \\\\"
            )
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
\caption{Sign-flip test over the seven IEEE-CIS steps after the peak. PO-risk does not name a single block. The transferred Brier movement is largest for the card block.}
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
    if "movielens" in payload:
        text += panel_tables(
            payload["movielens"],
            "MovieLens-1M, seven calendar days after a peak in the high-rating rate. "
            "Rating is the outcome. Shares use absolute movement.",
            "MovieLens last day of the pathway. Rating MSE is the predictive companion of Brier.",
            "movielens",
        )
    if "home_credit" in payload:
        text += panel_tables(
            payload["home_credit"],
            "Home Credit along SK\\_ID\\_CURR. Seven slices after a peak in the default rate. "
            "There is no event time. Every block MMD stays below 0.005. "
            "An MMD entry of -- is a total below $10^{-4}$, and a large MMD share is still a split of a near-zero total.",
            "Home Credit last slice. PO-risk and transferred Brier on the default label.",
            "home-credit",
        )
    audit = sampler_lines(payload)
    if audit:
        text += r"""
\begin{table}[t]
\centering
\caption{Sampler floor on the last batch for PermuCATE PO-risk and transferred Brier. The floor is the change from redrawing that block on the peak day only. Excess is the last-batch change minus the floor. Raw and excess shares are each block's fraction of total absolute movement, before and after subtracting the floor. Verdict uses Brier excess against the floor and day MMD ($p<0.05$ for drift). Use this table to tell generator redraw (sampler), covariate shift (drift), and model sensitivity (model) apart.}
\label{tab:sampler-floor}
\scriptsize
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcccccccccccccc}
\toprule
data & block & PO floor & PO day & PO excess & PO raw & PO excess & Brier floor & Brier day & Brier excess & Brier raw & Brier excess & anchor MMD & day MMD & verdict \\
\midrule
""" + audit + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    # The synthetic pathway header has 9 columns but I listed 8 c's. Fix below by rewriting the header in the string carefully.
    out = ROOT / "results" / "pathway_tables.tex"
    out.write_text(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
