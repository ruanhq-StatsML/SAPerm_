"""Emit a delivery-ready applied LaTeX note: dual ranking (A_j + PO-risk).

Regenerate from ranking_recovery.json after the eval run.
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

    ladder = []
    for label, coef in GRADED_TARGET_COEFS:
        a = _eta2_gaussian(abs(coef)) - base
        ladder.append(f"\\texttt{{{_tex_escape(label)}}} & {coef:.2f} & {a:+.3f} \\\\")

    dss_s = _mean(synthetic, "strength_shift", "spearman")
    po_s = _mean(synthetic, "po_gap", "spearman")
    dss_sg = _mean(synthetic, "strength_shift", "sign_agreement")
    po_sg = _mean(synthetic, "po_gap", "sign_agreement")

    flip_bits = [r["flip_separation"] for r in synthetic if r.get("flip_separation")]
    if flip_bits:
        abs_a = float(np.mean([b["mean_abs_strength_shift"] for b in flip_bits]))
        cms = float(np.mean([b["mean_cms"] for b in flip_bits]))
    else:
        abs_a = cms = float("nan")

    uci_lines = []
    dss_sp, po_sp, dss_sg_u, po_sg_u = [], [], [], []
    for name, rows in uci.items():
        ok = [r for r in rows if "error" not in r]
        if not ok:
            continue
        ds = _mean(ok, "strength_shift", "spearman")
        ps = _mean(ok, "po_gap", "spearman")
        dg = _mean(ok, "strength_shift", "sign_agreement")
        pg = _mean(ok, "po_gap", "sign_agreement")
        dss_sp.append(ds)
        po_sp.append(ps)
        dss_sg_u.append(dg)
        po_sg_u.append(pg)
        uci_lines.append(
            f"\\texttt{{{_tex_escape(name)}}} & {_fmt(ds)} & {_fmt(ps)} & {_fmt(dg)} & {_fmt(pg)} \\\\"
        )

    mean_dss = float(np.mean(dss_sp)) if dss_sp else float("nan")
    mean_po = float(np.mean(po_sp)) if po_sp else float("nan")
    mean_dss_sg = float(np.mean(dss_sg_u)) if dss_sg_u else float("nan")
    mean_po_sg = float(np.mean(po_sg_u)) if po_sg_u else float("nan")

    # Compact waveform (or first available) example
    example_rows = []
    example_name = None
    for name in ("waveform", "pendigits", "breast_cancer"):
        rows = [r for r in uci.get(name, []) if "error" not in r and "estimates_response" in r]
        if not rows:
            continue
        example_name = name
        r0 = rows[0]
        ora = np.asarray(r0["oracle_response"], dtype=float)
        est = np.asarray(r0["estimates_response"]["strength_shift"], dtype=float)
        pog = np.asarray(r0["estimates_response"]["po_gap"], dtype=float)
        for i, (lab, _) in enumerate(GRADED_TARGET_COEFS):
            example_rows.append(
                f"\\texttt{{{_tex_escape(lab)}}} & {ora[i]:+.3f} & {est[i]:+.3f} & {pog[i]:.3f} \\\\"
            )
        break

    example_tex = ""
    if example_name is not None:
        example_tex = "\n".join(
            [
                r"\paragraph{Illustration.}",
                f"One replicate on \\texttt{{{_tex_escape(example_name)}}}:",
                r"\begin{center}",
                r"\begin{tabular}{lrrr}",
                r"\toprule",
                r"Response & Oracle $A$ & $\widehat A$ & PO-risk $\Delta^{\mathrm{PO}}$ \\",
                r"\midrule",
                *example_rows,
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{center}",
                r"PO-risk peaks on \texttt{flip}; $\widehat A$ stays near zero there and keeps the signed order elsewhere.",
                "",
            ]
        )

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage{enumitem}",
        r"\begin{document}",
        "",
        r"\title{Dual ranking for dependence shift:\\ strength $A_j$ and PO-risk $\Delta^{\mathrm{PO}}_j$}",
        r"\author{Applied note --- SAPerm / OOD VIMP}",
        r"\date{}",
        r"\maketitle",
        "",
        r"\noindent\textit{Direct application.} Given two batches (or factor windows), fit one",
        r"conditional mean per feature and emit \textbf{two} rankings: dependence-strength",
        r"shift $A_j$ (signed redundancy) and PO-risk gap $\Delta^{\mathrm{PO}}_j$",
        r"(relationship / map change, including flips). Ranking recovery shows they are not",
        r"interchangeable. On factor panels, run the same scores on residuals after removing $G$.",
        "",
        r"\section{Two scores from one fit}",
        "",
        r"For coordinate $X_j$ and batch $t\in\{0,1\}$, fit $m_{t,j}=\widehat{\mathbb{E}}_t[X_j\mid X_{-j}]$.",
        r"Held-out correlation ratio and signed strength shift:",
        r"\begin{equation}",
        r"\eta^{2}_{t,j}=1-\frac{\mathrm{MSE}_t(X_j,m_{t,j})}{\widehat{\mathrm{Var}}_t(X_j)},",
        r"\qquad",
        r"A_j=\eta^{2}_{1,j}-\eta^{2}_{0,j}.",
        r"\label{eq:Aj}",
        r"\end{equation}",
        r"$A_j>0$: peers explain more of $X_j$ on the target (redundancy up).",
        r"$A_j<0$: dependence weakened.",
        "",
        r"Same fits, PO-risk ranking. With propensity $\pi$, contrast $\tau_j=m_{1,j}-m_{0,j}$,",
        r"and doubly robust pseudo-outcome $\phi$ (batch as treatment),",
        r"\begin{equation}",
        r"R(\tau)=\widehat{\mathbb{E}}[(\phi-\tau)^2],",
        r"\qquad",
        r"\Delta^{\mathrm{PO}}_j=R(0)-R(\tau_j).",
        r"\label{eq:PO}",
        r"\end{equation}",
        r"Rank by $\Delta^{\mathrm{PO}}_j$ to find where the \emph{conditional map} moved.",
        r"Large PO-risk does \emph{not} mean stronger dependence: a pure sign flip has $A_j\approx 0$",
        r"and large $\Delta^{\mathrm{PO}}_j$.",
        "",
        r"\begin{center}",
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Ranking & Score & Question \\",
        r"\midrule",
        r"Strength & $A_j$ \eqref{eq:Aj} & Intensity up/down (signed) \\",
        r"PO-risk & $\Delta^{\mathrm{PO}}_j$ \eqref{eq:PO} & Map changed? (incl.\ flip) \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        "",
        r"\section{Factor residuals (quant application)}",
        "",
        r"Write $F_j=\beta_j^\top G+\varepsilon_j$. Apply \eqref{eq:Aj}--\eqref{eq:PO} to the residual",
        r"panel $(\varepsilon_1,\ldots,\varepsilon_p)$ across two windows:",
        r"\begin{itemize}[leftmargin=*]",
        r"\item $A_j(\varepsilon)$: idiosyncratic crowding / diversification.",
        r"\item $\Delta^{\mathrm{PO}}_j(\varepsilon)$: residual hedge / conditional-map flip.",
        r"\item Optional $V_{j\leftarrow k}$: which partner residual's unique $R^2$ changed.",
        r"\end{itemize}",
        r"Pairwise $\mathrm{Corr}(\varepsilon_j,\varepsilon_k)$ is the linear special case.",
        r"API: \texttt{factor\_residual\_dependence\_shift(F\_ref, F\_new, n\_common=1)}.",
        "",
        r"\section{Ranking recovery}",
        "",
        r"Graded oracle (reference coef $c_0=1$):",
        r"\begin{center}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Response & Target coef & Oracle $A$ \\",
        r"\midrule",
        *ladder,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        "",
        r"Spearman$(A)=\mathrm{corr}(\mathrm{rank}(A^{\mathrm{oracle}}),\mathrm{rank}(\mathrm{score}))$.",
        r"Strength should recover the signed ladder; PO-risk should not (it over-ranks flips).",
        "",
        r"\paragraph{Synthetic.}",
        f"Strength: Spearman$(A)={_fmt(dss_s)}$, sign agree ${_fmt(dss_sg)}$. ",
        f"PO-risk: Spearman$(A)={_fmt(po_s)}$, sign agree ${_fmt(po_sg)}$. ",
        f"On flips, mean $|A|={_fmt(abs_a)}$ vs mean CMS ${_fmt(cms)}$.",
        "",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{UCI factor protocol: strength ranking vs PO-risk ranking (response block).}",
        r"\label{tab:dss-po}",
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
        example_tex,
        r"\section{How to read the two rankings}",
        "",
        r"\begin{center}",
        r"\begin{tabular}{lllp{5.8cm}}",
        r"\toprule",
        r"$A_j$ & $\Delta^{\mathrm{PO}}$ & Pattern & Action \\",
        r"\midrule",
        r"$\downarrow$ large & small--mid & Diversification & Peers explain less; residual risk more unique \\",
        r"$\uparrow$ large & small--mid & Crowding & Residuals more reconstructible \\",
        r"$\approx 0$ & $\uparrow$ large & Map / hedge flip & Re-check hedges, not only variance \\",
        r"large & large & Both moved & Redesign intensity and relationship \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        "",
        r"Ranking recovery's insight is the \emph{gap}: mean UCI Spearman$(A)$ is ",
        f"{_fmt(mean_dss)} for strength vs {_fmt(mean_po)} for PO-risk. ",
        r"Use both; neither replaces the other.",
        "",
        r"\paragraph{One-liner usage.}",
        r"\begin{verbatim}",
        r"from OOD_vimp_wrapper import dependence_strength_shift",
        r"out = dependence_strength_shift(X_ref, X_new)  # A_j, po_gap",
        r"# factors: factor_residual_dependence_shift(F_ref, F_new, n_common=1)",
        r"\end{verbatim}",
        "",
        r"\end{document}",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=RESULTS / "ranking_recovery.json")
    parser.add_argument("--out", type=Path, default=LATEX / "dependence_strength_shift.tex")
    args = parser.parse_args(argv)
    payload = json.loads(args.json.read_text())
    tex = build_tex(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(tex)
    (RESULTS / "dependence_strength_shift.tex").write_text(tex)
    print(f"wrote {args.out}")
    print(f"wrote {RESULTS / 'dependence_strength_shift.tex'}")


if __name__ == "__main__":
    main()
