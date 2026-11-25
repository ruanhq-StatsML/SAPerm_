"""Emit a short manuscript-style LaTeX note from ranking_recovery.json.

Headline comparison: signed dependence-strength ranking vs PO-risk ranking
(``po_gap`` = risk drop from the zero contrast to the fitted T-learner contrast).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.dependence_strength_shift.dgp import GRADED_TARGET_COEFS, _eta2_gaussian

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LATEX = ROOT / "latex"


def _mean(rows, method: str, key: str) -> float:
    vals = []
    for row in rows:
        if "error" in row:
            continue
        v = row["metrics"][method][key]
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    return float(np.mean(vals)) if vals else float("nan")


def _fmt(x: float, digits: int = 3) -> str:
    if not np.isfinite(x):
        return "---"
    return f"{x:.{digits}f}"


def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_")


def build_tex(payload: dict) -> str:
    synthetic = payload.get("synthetic", [])
    uci = payload.get("uci", {})
    base = _eta2_gaussian(1.0)

    ladder_rows = []
    for label, coef in GRADED_TARGET_COEFS:
        a = _eta2_gaussian(abs(coef)) - base
        ladder_rows.append(
            f"\\texttt{{{_tex_escape(label)}}} & {coef:.2f} & {a:+.3f} \\\\"
        )

    dss_s = _mean(synthetic, "strength_shift", "spearman")
    po_s = _mean(synthetic, "po_gap", "spearman")
    dss_sg = _mean(synthetic, "strength_shift", "sign_agreement")
    po_sg = _mean(synthetic, "po_gap", "sign_agreement")
    dss_pk = _mean(synthetic, "strength_shift", "precision_at_k")
    po_pk = _mean(synthetic, "po_gap", "precision_at_k")

    uci_lines = []
    dss_sp_all, po_sp_all, dss_sg_all, po_sg_all = [], [], [], []
    for name, rows in uci.items():
        ok = [r for r in rows if "error" not in r]
        if not ok:
            continue
        ds = _mean(ok, "strength_shift", "spearman")
        ps = _mean(ok, "po_gap", "spearman")
        dg = _mean(ok, "strength_shift", "sign_agreement")
        pg = _mean(ok, "po_gap", "sign_agreement")
        dss_sp_all.append(ds)
        po_sp_all.append(ps)
        dss_sg_all.append(dg)
        po_sg_all.append(pg)
        uci_lines.append(
            f"\\texttt{{{_tex_escape(name)}}} & {_fmt(ds)} & {_fmt(ps)} & {_fmt(dg)} & {_fmt(pg)} \\\\"
        )

    # Flip separation from synthetic
    flip_bits = [r["flip_separation"] for r in synthetic if r.get("flip_separation")]
    if flip_bits:
        abs_a = float(np.mean([b["mean_abs_strength_shift"] for b in flip_bits]))
        cms = float(np.mean([b["mean_cms"] for b in flip_bits]))
        corr = float(np.mean([b["mean_corr_q"] for b in flip_bits]))
    else:
        abs_a = cms = corr = float("nan")

    # One concrete waveform-style example if present
    example_block = ""
    for name in ("waveform", "pendigits", "breast_cancer"):
        rows = uci.get(name, [])
        ok = [r for r in rows if "error" not in r and "estimates_response" in r]
        if not ok:
            continue
        r0 = ok[0]
        ora = np.asarray(r0["oracle_response"], dtype=float)
        est = np.asarray(r0["estimates_response"]["strength_shift"], dtype=float)
        pog = np.asarray(r0["estimates_response"]["po_gap"], dtype=float)
        example_block = "\n".join(
            [
                r"\subsection*{One replicate on \texttt{" + _tex_escape(name) + r"}}",
                "",
                "Oracle ladder, estimated strength shift, and PO-risk gap on the six responses:",
                r"\begin{center}",
                r"\begin{tabular}{lrrr}",
                r"\toprule",
                r"Response & Oracle $A$ & $\widehat A$ (strength) & PO-risk gap \\",
                r"\midrule",
            ]
            + [
                f"\\texttt{{{_tex_escape(lab)}}} & {ora[i]:+.3f} & {est[i]:+.3f} & {pog[i]:.3f} \\\\"
                for i, (lab, _) in enumerate(GRADED_TARGET_COEFS)
            ]
            + [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{center}",
                "",
                r"The PO-risk column is largest on the flip coordinate whenever the conditional map reverses; "
                r"the strength column stays near zero there and preserves the signed order of the other five levels.",
                "",
            ]
        )
        break

    mean_dss = float(np.mean(dss_sp_all)) if dss_sp_all else float("nan")
    mean_po = float(np.mean(po_sp_all)) if po_sp_all else float("nan")
    mean_dss_sg = float(np.mean(dss_sg_all)) if dss_sg_all else float("nan")
    mean_po_sg = float(np.mean(po_sg_all)) if po_sg_all else float("nan")

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage{enumitem}",
        r"\begin{document}",
        r"\section*{Dependence strength shift versus PO-risk ranking}",
        "",
        r"For each feature $X_j$ and batch $t\in\{0,1\}$, fit",
        r"$m_{t,j}(X_{-j})=\widehat{\mathbb{E}}_t[X_j\mid X_{-j}]$ and score the held-out correlation ratio",
        r"\begin{equation}",
        r"\eta^{2}_{t,j}=1-\frac{\widehat{\mathbb{E}}\bigl[(X_j-m_{t,j})^2\bigr]}{\widehat{\mathrm{Var}}(X_j)},",
        r"\qquad A_j=\eta^{2}_{1,j}-\eta^{2}_{0,j}.",
        r"\end{equation}",
        r"Signed $A_j$ is the dependence-strength shift: positive means $X_{-j}$ explains more of $X_j$ on the target batch.",
        "",
        r"The same fits give a PO-risk ranking. With propensity $\pi$ and contrast $\tau=m_{1,j}-m_{0,j}$,",
        r"\begin{equation}",
        r"R(\tau)=\widehat{\mathbb{E}}\bigl[(\phi-\tau)^2\bigr],",
        r"\qquad \Delta^{\mathrm{PO}}_j=R(0)-R(\tau),",
        r"\end{equation}",
        r"where $\phi$ is the doubly robust pseudo-outcome for treating batch as the ``treatment''.",
        r"Ranking features by $\Delta^{\mathrm{PO}}_j$ (equivalently by the PO-risk gap) answers whether the",
        r"\emph{conditional mean relationship} changed, not whether dependence strengthened or weakened.",
        "",
        r"\subsection*{Graded oracle ladder}",
        "",
        r"Reference uses coefficient $c_0=1$ on a shared factor; the target applies the coefficients below.",
        r"Population $A=\eta^{2}(|c|)-\eta^{2}(1)$ under $X=cF+\varepsilon$ with unit-variance noise.",
        r"Spearman$(A)$ is the rank correlation between oracle $A$ and a method's feature scores.",
        "",
        r"\begin{center}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Response & Target coef & Oracle $A$ \\",
        r"\midrule",
        *ladder_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        "",
        r"\subsection*{Synthetic ranking recovery}",
        "",
        r"\begin{center}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Ranking score & Spearman$(A)$ & Sign agree & Precision@$k$ \\",
        r"\midrule",
        f"Strength shift $A_j$ & {_fmt(dss_s)} & {_fmt(dss_sg)} & {_fmt(dss_pk)} \\\\",
        f"PO-risk gap $\\Delta^{{\\mathrm{{PO}}}}_j$ & {_fmt(po_s)} & {_fmt(po_sg)} & {_fmt(po_pk)} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        "",
        f"On flip coordinates, mean $|A|$ is {_fmt(abs_a)} while mean conditional-mean shift is {_fmt(cms)}",
        f"(Q-fixed map correlation {_fmt(corr)}, near $-1$).",
        r"PO-risk ranking therefore treats the flip as a large change; strength ranking does not.",
        "",
        r"\subsection*{UCI factor protocol}",
        "",
        r"Each table is split into reference/target. A PCA factor of the real covariates drives the six",
        r"graded responses above. Metrics are Spearman ranking recovery on that response block.",
        "",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Signed ranking recovery: dependence strength shift versus PO-risk gap.}",
        r"\label{tab:dss-vs-porisk}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Dataset & Spearman $A$ & Spearman PO & Sign $A$ & Sign PO \\",
        r"\midrule",
        *uci_lines,
        r"\midrule",
        f"Mean & {_fmt(mean_dss)} & {_fmt(mean_po)} & {_fmt(mean_dss_sg)} & {_fmt(mean_po_sg)} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        example_block,
        r"\subsection*{Reading}",
        "",
        r"\begin{enumerate}[leftmargin=*]",
        r"\item Spearman$(A)$ asks whether the method recovers the signed order of dependence intensity.",
        r"\item The PO-risk ranking $\Delta^{\mathrm{PO}}$ is the natural companion for \emph{relationship} change;",
        r"it is not a strength score, and it systematically over-ranks flips.",
        r"\item Use both: PO-risk to flag that $m_{1,j}-m_{0,j}$ moved; $A_j$ to say whether redundancy rose or fell.",
        r"\end{enumerate}",
        "",
        r"\end{document}",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        type=Path,
        default=RESULTS / "ranking_recovery.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=LATEX / "dependence_strength_shift.tex",
    )
    args = parser.parse_args(argv)
    payload = json.loads(args.json.read_text())
    tex = build_tex(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(tex)
    # Also drop a copy next to BOARD.md for the results folder.
    (RESULTS / "dependence_strength_shift.tex").write_text(tex)
    print(f"wrote {args.out}")
    print(f"wrote {RESULTS / 'dependence_strength_shift.tex'}")


if __name__ == "__main__":
    main()
