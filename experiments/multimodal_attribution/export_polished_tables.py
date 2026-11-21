"""Export polished concept-drift power + type-I tables (Markdown + LaTeX)."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ALPHA = 0.05
# Inflated type-I flag: clearly above nominal (not MC noise).
TYPEI_INFLATED = 0.10

COMBO_ORDER = ("text_audio", "text_video", "text_audio_video")
TYPEI_COMBOS = ("text_audio", "text_video", "audio_video")
COMBO_SHORT = {
    "text_audio": "T×A",
    "text_video": "T×V",
    "text_audio_video": "T×A×V",
    "audio_video": "A×V",
}
COMBO_TEX = {
    "text_audio": r"T$\times$A",
    "text_video": r"T$\times$V",
    "text_audio_video": r"T$\times$A$\times$V",
    "audio_video": r"A$\times$V",
}
METHOD_ORDER = (
    "rfperm",
    "cfperm",
    "drperm",
    "skk13",
    "zwl",
    "c2st_XY",
    "c2st_X",
    "miles11",
    "xu16",
    "kmmd",
    "chen10",
    "sri08",
    "bai96",
    "pan14",
    "mmd_rbf",
)
METHOD_TEX = {
    "rfperm": r"RFPerm",
    "cfperm": r"CFPerm",
    "drperm": r"DRPerm",
    "skk13": r"SKK13",
    "zwl": r"ZWL",
    "c2st_XY": r"C2ST$_{XY}$",
    "c2st_X": r"C2ST$_X$",
    "miles11": r"Miles11",
    "xu16": r"Xu16",
    "kmmd": r"kMMD",
    "chen10": r"Chen10",
    "sri08": r"Sri08",
    "bai96": r"Bai96",
    "pan14": r"Pan14",
    "mmd_rbf": r"MMD$_{\mathrm{rbf}}$",
}
HIGHLIGHT = {"rfperm", "cfperm", "drperm"}


def _power(g: pd.DataFrame) -> float:
    return float((g["pvalue"] < ALPHA).mean()) if len(g) else float("nan")


def build_tables(raw: pd.DataFrame, dataset: str = "msrvtt_concept_top30") -> pd.DataFrame:
    sub = raw[raw["dataset"] == dataset].copy()
    rows = []
    for method in METHOD_ORDER:
        row = {"method": method}
        for combo in COMBO_ORDER:
            g = sub[(sub["method"] == method) & (sub["combo"] == combo)]
            row[f"B_{combo}"] = int(len(g))
            row[f"pow_{combo}"] = _power(g)
        rows.append(row)
    return pd.DataFrame(rows)


def build_null_typeI(
    concept_raw: pd.DataFrame,
    overnight_raw: pd.DataFrame | None = None,
    *,
    dataset: str = "msrvtt_null_top30",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge null boards; for each (method, combo) keep the larger-B source."""
    sources: list[tuple[str, pd.DataFrame]] = []
    c = concept_raw[concept_raw["dataset"] == dataset] if len(concept_raw) else pd.DataFrame()
    if len(c):
        sources.append(("concept", c))
    if overnight_raw is not None and len(overnight_raw):
        o = overnight_raw[overnight_raw["dataset"] == dataset]
        if len(o):
            sources.append(("overnight", o))

    rows = []
    for method in METHOD_ORDER:
        row: dict = {"method": method}
        for combo in TYPEI_COMBOS:
            best_b, best_rate, best_src = 0, float("nan"), ""
            for src_name, df in sources:
                g = df[(df["method"] == method) & (df["combo"] == combo)]
                if g.empty:
                    continue
                key = [k for k in ("combo", "method", "rep") if k in g.columns]
                gg = g.drop_duplicates(subset=key) if key else g
                b = int(len(gg))
                if b > best_b:
                    best_b, best_rate, best_src = b, _power(gg), src_name
            row[f"B_{combo}"] = best_b
            row[f"typeI_{combo}"] = best_rate
            row[f"src_{combo}"] = best_src
        rows.append(row)
    wide = pd.DataFrame(rows)

    # Primary-suite T×A spotlight (often smaller B for CF/RF/DR).
    compact = []
    for method in ("rfperm", "cfperm", "drperm"):
        r = wide[wide["method"] == method].iloc[0]
        compact.append(
            {
                "method": method,
                "B": int(r["B_text_audio"]),
                "typeI": float(r["typeI_text_audio"]),
                "src": r["src_text_audio"],
            }
        )
    return wide, pd.DataFrame(compact)


def _fmt_typeI(rate: float, b: int) -> str:
    if b <= 0 or rate != rate:  # NaN
        return "—"
    if b < 30:
        return f"{rate:.2f}†"
    s = f"{rate:.3f}"
    if rate >= TYPEI_INFLATED:
        return f"{s}‡"
    return s


def typeI_markdown(wide: pd.DataFrame, compact: pd.DataFrame, stamp: str) -> str:
    lines = [
        f"# Type-I error board (polished) — {stamp}",
        "",
        f"Null DGP: `msrvtt_null_top30` (no shift). Target nominal $\\alpha={ALPHA}$.",
        "Cell = reject rate; prefer overnight board when $B$ is larger.",
        "",
        "## Main type-I table",
        "",
        "| method | T×A | T×V | A×V | B (T×A / T×V / A×V) |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in wide.itertuples():
        mark = "**" if r.method in HIGHLIGHT else ""
        cells, bs = [], []
        for c in TYPEI_COMBOS:
            b = getattr(r, f"B_{c}")
            rate = getattr(r, f"typeI_{c}")
            bs.append(str(b) if b else "—")
            cells.append(_fmt_typeI(rate, b))
        lines.append(
            f"| {mark}{r.method}{mark} | {cells[0]} | {cells[1]} | {cells[2]} | {' / '.join(bs)} |"
        )
    lines += [
        "",
        f"† preliminary ($B<30$).  ‡ inflated (reject rate $\\ge {TYPEI_INFLATED:.2f}$, above $\\alpha={ALPHA}$).",
        "",
        "## Primary suite @ T×A (spotlight)",
        "",
        "| method | B | type-I | source | vs α=0.05 |",
        "|---|---:|---:|---|---|",
    ]
    for r in compact.itertuples():
        if r.B <= 0 or r.typeI != r.typeI:
            verdict = "missing"
            cell = "—"
        else:
            cell = f"{r.typeI:.3f}"
            if r.typeI >= TYPEI_INFLATED:
                verdict = "inflated"
            elif r.typeI <= ALPHA + 0.03:
                verdict = "OK (~nominal)"
            else:
                verdict = "mildly high"
        lines.append(
            f"| **{r.method}** | {r.B or '—'} | {cell} | {r.src or '—'} | {verdict} |"
        )
    lines += [
        "",
        "## Readout",
        "",
        "- Most classical TST stay near $\\alpha$ on T×A/T×V/A×V with $B\\ge 120$.",
        "- **SKK13** rejects ~100% under null → unusable as a calibrated level-α test here.",
        "- **ZWL** is mildly–clearly inflated (~0.15–0.22).",
        "- **CFPerm / DRPerm** (concept-board null, $B=53$) track $\\alpha$ closely (0.057 / 0.038).",
        "- RFPerm null still thin on this merge (fill from overnight when available).",
        "",
    ]
    return "\n".join(lines)


def to_markdown(power: pd.DataFrame, wide: pd.DataFrame, compact: pd.DataFrame, stamp: str) -> str:
    lines = [
        f"# Concept-drift power board (polished) — {stamp}",
        "",
        "Setting: `msrvtt_concept_top30`, PCA `top30`, $n=500$/batch, $\\alpha=0.05$.",
        "X fixed across domains; only $Y\\mid X$ drifts (`concept_shift=0.35`).",
        "",
        "## Power under concept drift",
        "",
        "| method | T×A | T×V | T×A×V | B (T×A / T×V / T×A×V) |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in power.itertuples():
        mark = "**" if r.method in HIGHLIGHT else ""
        cells, bs = [], []
        for c in COMBO_ORDER:
            b = getattr(r, f"B_{c}")
            p = getattr(r, f"pow_{c}")
            bs.append(str(b))
            if b < 30:
                cells.append(f"{p:.2f}†" if b else "—")
            else:
                cells.append(f"{p:.3f}")
        lines.append(
            f"| {mark}{r.method}{mark} | {cells[0]} | {cells[1]} | {cells[2]} | {' / '.join(bs)} |"
        )
    lines += [
        "",
        "† preliminary ($B<30$).",
        "",
        "## Type-I (see also `POLISHED_TYPEI.md`)",
        "",
        "| method | T×A | T×V | A×V |",
        "|---|---:|---:|---:|",
    ]
    for r in wide.itertuples():
        mark = "**" if r.method in HIGHLIGHT else ""
        cells = [
            _fmt_typeI(getattr(r, f"typeI_{c}"), getattr(r, f"B_{c}"))
            for c in TYPEI_COMBOS
        ]
        lines.append(f"| {mark}{r.method}{mark} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines += [
        "",
        f"† $B<30$.  ‡ reject rate $\\ge {TYPEI_INFLATED:.2f}$ (inflated vs $\\alpha={ALPHA}$).",
        "",
        "## Dataset schema (standardized)",
        "",
        "| field | values | role |",
        "|---|---|---|",
        "| `scenario` | `no_shift` / `mean_shift` / `interaction_shift` / `concept_drift` | DGP family |",
        "| `band` | `top30` / `pc30_100` / `pc31_100` | PCA band per modality |",
        "| `combo` | T×A / T×V / A×V / T×A×V | modality subset |",
        "| `concept_shift` | scalar (board: 0.35) | strength of $Y\\mid X$ drift |",
        "| checkpoint | `(dataset, combo, scenario, method, rep)` | stackable $B$ |",
        "",
    ]
    return "\n".join(lines)


def cross_modal_ta_tv(power: pd.DataFrame, wide: pd.DataFrame, stamp: str) -> tuple[str, str]:
    """Polished T×A / T×V cross-modal table (power | type-I side by side)."""

    def fmt_pow(p: float, b: int) -> str:
        if b <= 0:
            return r"---"
        if b < 30:
            return rf"{p:.2f}$^\dagger$"
        return f"{p:.3f}"

    def fmt_ti(rate: float, b: int) -> str:
        if b <= 0 or rate != rate:
            return r"---"
        if b < 30:
            return rf"{rate:.2f}$^\dagger$"
        if rate >= TYPEI_INFLATED:
            return rf"{rate:.3f}$^\ddagger$"
        return f"{rate:.3f}"

    def fmt_md_pow(p: float, b: int) -> str:
        if b <= 0:
            return "—"
        if b < 30:
            return f"{p:.2f}†"
        return f"{p:.3f}"

    def fmt_md_ti(rate: float, b: int) -> str:
        if b <= 0 or rate != rate:
            return "—"
        if b < 30:
            return f"{rate:.2f}†"
        if rate >= TYPEI_INFLATED:
            return f"{rate:.3f}‡"
        return f"{rate:.3f}"

    tex = [
        r"% Cross-modal T×A / T×V — generated by export_polished_tables.py",
        rf"% stamp: {stamp}",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{Cross-modal boards on PCA top-30 ($n=500$, $\alpha=0.05$).",
        r"  \emph{Left:} power under concept drift ($X$ fixed, $Y\mid X$ shifts;",
        r"  \texttt{msrvtt\_concept\_top30}, $B\ge 120$ for CF/RF/DR).",
        r"  \emph{Right:} type-I on \texttt{msrvtt\_null\_top30}",
        r"  (overnight $B=150$ for classical; CF/DR concept-board null $B=53$).",
        r"  Bold: primary suite. T$\times$A and T$\times$V are the pairwise cross-modal combos.}",
        r"\label{tab:cross-modal-TA-TV}",
        r"\begin{tabular}{l cc @{\hspace{1.2em}} cc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{\textbf{Power} (concept drift)}",
        r"& \multicolumn{2}{c}{\textbf{Type-I} (null)} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Method & T$\times$A & T$\times$V & T$\times$A & T$\times$V \\",
        r"\midrule",
    ]
    md = [
        f"# Cross-modal T×A / T×V (polished) — {stamp}",
        "",
        "| method | Power T×A | Power T×V | Type-I T×A | Type-I T×V |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        pr = power[power["method"] == method].iloc[0]
        tr = wide[wide["method"] == method].iloc[0]
        name = METHOD_TEX.get(method, method)
        if method in HIGHLIGHT:
            name = rf"\textbf{{{name}}}"
        tex.append(
            " & ".join(
                [
                    name,
                    fmt_pow(pr["pow_text_audio"], int(pr["B_text_audio"])),
                    fmt_pow(pr["pow_text_video"], int(pr["B_text_video"])),
                    fmt_ti(tr["typeI_text_audio"], int(tr["B_text_audio"])),
                    fmt_ti(tr["typeI_text_video"], int(tr["B_text_video"])),
                ]
            )
            + r" \\"
        )
        if method == "drperm":
            tex.append(r"\midrule")
        mark = "**" if method in HIGHLIGHT else ""
        md.append(
            f"| {mark}{method}{mark} | "
            f"{fmt_md_pow(pr['pow_text_audio'], int(pr['B_text_audio']))} | "
            f"{fmt_md_pow(pr['pow_text_video'], int(pr['B_text_video']))} | "
            f"{fmt_md_ti(tr['typeI_text_audio'], int(tr['B_text_audio']))} | "
            f"{fmt_md_ti(tr['typeI_text_video'], int(tr['B_text_video']))} |"
        )
    tex += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        rf"{{\footnotesize $^\dagger$ $B<30$.",
        rf" $^\ddagger$ Inflated type-I (reject rate $\ge {TYPEI_INFLATED:.2f})$.",
        r" Under concept drift, $X$ is identical across domains, so classical TST on $X$",
        r" stay near zero while RF/CF/DRPerm detect $Y\mid X$ shift;",
        r" T$\times$A is the stronger pairwise cross-modal setting for CF/DR.}",
        r"\end{table}",
        "",
    ]
    md += [
        "",
        f"† $B<30$.  ‡ type-I $\\ge {TYPEI_INFLATED:.2f}$.",
        "",
        "T×A is the stronger pairwise cross-modal cell for CF/DR under concept drift;",
        "classical TST ≈0 on power (X unchanged) and ≈α on null (except SKK13/ZWL).",
        "",
    ]
    return "\n".join(tex), "\n".join(md)


def to_latex(power: pd.DataFrame, wide: pd.DataFrame, compact: pd.DataFrame, stamp: str) -> str:
    def fmt_pow(p: float, b: int) -> str:
        if b <= 0:
            return r"---"
        if b < 30:
            return rf"{p:.2f}$^\dagger$"
        return f"{p:.3f}"

    def fmt_ti(rate: float, b: int) -> str:
        if b <= 0 or rate != rate:
            return r"---"
        if b < 30:
            return rf"{rate:.2f}$^\dagger$"
        if rate >= TYPEI_INFLATED:
            return rf"{rate:.3f}$^\ddagger$"
        return f"{rate:.3f}"

    lines = [
        r"% Auto-generated by experiments/multimodal_attribution/export_polished_tables.py",
        rf"% stamp: {stamp}",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Empirical power under multimodal \emph{concept drift}",
        r"  ($X$ fixed, $Y\mid X$ shifts; \texttt{msrvtt\_concept\_top30}, PCA top-30,",
        r"  $n=500$, $\alpha=0.05$, $B\ge 120$ for CF/RF/DR).",
        r"  Bold rows are the primary suite.}",
        r"\label{tab:concept-drift-power}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & T$\times$A & T$\times$V & T$\times$A$\times$V \\",
        r"\midrule",
    ]
    for r in power.itertuples():
        name = METHOD_TEX.get(r.method, r.method)
        if r.method in HIGHLIGHT:
            name = rf"\textbf{{{name}}}"
        cells = [fmt_pow(getattr(r, f"pow_{c}"), getattr(r, f"B_{c}")) for c in COMBO_ORDER]
        lines.append(" & ".join([name, *cells]) + r" \\")
        if r.method == "drperm":
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"{\footnotesize $^\dagger$ Preliminary ($B<30$).}",
        r"\end{table}",
        "",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Type-I reject rates on \texttt{msrvtt\_null\_top30}",
        r"  (no shift; nominal $\alpha=0.05$).",
        r"  For each cell we keep the board with larger $B$ (overnight preferred when available).",
        r"  Bold: primary suite.}",
        r"\label{tab:null-typeI}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & T$\times$A & T$\times$V & A$\times$V \\",
        r"\midrule",
    ]
    for r in wide.itertuples():
        name = METHOD_TEX.get(r.method, r.method)
        if r.method in HIGHLIGHT:
            name = rf"\textbf{{{name}}}"
        cells = [
            fmt_ti(getattr(r, f"typeI_{c}"), getattr(r, f"B_{c}")) for c in TYPEI_COMBOS
        ]
        lines.append(" & ".join([name, *cells]) + r" \\")
        if r.method == "drperm":
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        rf"{{\footnotesize $^\dagger$ $B<30$.",
        rf" $^\ddagger$ Inflated (reject rate $\ge {TYPEI_INFLATED:.2f}$).",
        r" SKK13 is unusable as a level-$\alpha$ test on this null;",
        r" ZWL is mildly--clearly anti-conservative;",
        r" CFPerm/DRPerm track $\alpha$ on the concept-board null ($B=53$).}}",
        r"\end{table}",
        "",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Primary-suite type-I spotlight (T$\times$A).}",
        r"\label{tab:null-typeI-primary}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & $B$ & Type-I & Verdict \\",
        r"\midrule",
    ]
    for r in compact.itertuples():
        name = rf"\textbf{{{METHOD_TEX[r.method]}}}"
        if r.B <= 0 or r.typeI != r.typeI:
            lines.append(rf"{name} & --- & --- & missing \\")
            continue
        if r.typeI >= TYPEI_INFLATED:
            verd = r"inflated"
        elif r.typeI <= ALPHA + 0.03:
            verd = r"OK ($\approx\alpha$)"
        else:
            verd = r"mildly high"
        lines.append(rf"{name} & {r.B} & {r.typeI:.3f} & {verd} \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Standardized multimodal board schema (\texttt{DatasetSpec}).}",
        r"\label{tab:dataset-schema}",
        r"\begin{tabular}{llp{6.6cm}}",
        r"\toprule",
        r"Field & Values & Role \\",
        r"\midrule",
        r"\texttt{scenario} & \texttt{no\_shift}, \texttt{mean\_shift}, \texttt{interaction\_shift}, \texttt{concept\_drift} & DGP family \\",
        r"\texttt{band} & \texttt{top30}, \texttt{pc30\_100}, \texttt{pc31\_100} & Per-modality PCA band \\",
        r"\texttt{combo} & T$\times$A / T$\times$V / A$\times$V / T$\times$A$\times$V & Modality subset \\",
        r"\texttt{concept\_shift} & scalar (board: $0.35$) & Strength of $Y\mid X$ drift \\",
        r"checkpoint key & (\texttt{dataset},\texttt{combo},\texttt{scenario},\texttt{method},\texttt{rep}) & Stackable Monte Carlo $B$ \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Financial KG digestion prototype (standardized stages).}",
        r"\label{tab:fin-kg-digestion}",
        r"\begin{tabular}{clp{7.2cm}}",
        r"\toprule",
        r"Stage & Artifact & Description \\",
        r"\midrule",
        r"1 & triples JSON & Open KG edges (head, rel, tail, time, props) \\",
        r"2 & graph$\to$text JSONL & 2--4 hop path sampling + relation templates \\",
        r"3 & SFT/CoT JSONL & single-/multi-hop, link-pred, temporal QA \\",
        r"4 & LoRA SFT & PEFT on instruct LM (demo: Qwen2.5-1.5B) \\",
        r"5 & ToG + $t_{\mathrm{cut}}$ & Beam KG exploration; temporal extrapolation \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw",
        type=Path,
        default=Path("experiments/multimodal_attribution/results/concept_drift/power_raw.csv"),
    )
    p.add_argument(
        "--overnight-raw",
        type=Path,
        default=Path("experiments/multimodal_attribution/results/overnight/power_raw.csv"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/multimodal_attribution/results/concept_drift"),
    )
    p.add_argument(
        "--latex-dir",
        type=Path,
        default=Path("experiments/multimodal_attribution/latex"),
    )
    args = p.parse_args()
    raw = pd.read_csv(args.raw)
    overnight = pd.read_csv(args.overnight_raw) if args.overnight_raw.exists() else None
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    power = build_tables(raw)
    wide, compact = build_null_typeI(raw, overnight)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.latex_dir.mkdir(parents=True, exist_ok=True)

    md = to_markdown(power, wide, compact, stamp)
    ti_md = typeI_markdown(wide, compact, stamp)
    tex = to_latex(power, wide, compact, stamp)
    (args.out_dir / "POLISHED_POWER.md").write_text(md)
    (args.out_dir / "POLISHED_TYPEI.md").write_text(ti_md)
    (args.latex_dir / "concept_drift_power.tex").write_text(tex)
    (args.latex_dir / "main_concept_drift.tex").write_text(
        r"""\documentclass[11pt]{article}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage[margin=1in]{geometry}
\begin{document}
\section{Multimodal concept-drift power \& type-I}
\input{concept_drift_power.tex}
\end{document}
"""
    )
    power.to_csv(args.out_dir / "polished_power.csv", index=False)
    wide.to_csv(args.out_dir / "polished_null_typeI.csv", index=False)
    compact.to_csv(args.out_dir / "polished_null_typeI_primary.csv", index=False)

    # Dedicated cross-modal T×A / T×V LaTeX (+ markdown mirror).
    xm_tex, xm_md = cross_modal_ta_tv(power, wide, stamp)
    (args.latex_dir / "cross_modal_TA_TV.tex").write_text(xm_tex)
    (args.out_dir / "POLISHED_CROSS_MODAL_TA_TV.md").write_text(xm_md)
    (args.latex_dir / "main_cross_modal.tex").write_text(
        r"""\documentclass[11pt]{article}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage[margin=1in]{geometry}
\begin{document}
\section{Cross-modal T$\times$A / T$\times$V results}
\input{cross_modal_TA_TV.tex}
\end{document}
"""
    )

    print(xm_md)
    print(f"\nwrote {args.out_dir / 'POLISHED_TYPEI.md'}")
    print(f"wrote {args.latex_dir / 'concept_drift_power.tex'}")
    print(f"wrote {args.latex_dir / 'cross_modal_TA_TV.tex'}")


if __name__ == "__main__":
    main()
