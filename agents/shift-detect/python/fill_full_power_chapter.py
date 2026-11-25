#!/usr/bin/env python3
"""Fill results/full_power_modasense.tex from benchmark_all_full.csv."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "results"
CSV = OUT / "benchmark_all_full.csv"
TEX = OUT / "full_power_modasense.tex"


def _fmt(x) -> str:
    try:
        v = float(x)
    except Exception:
        return "---"
    if v != v:
        return "---"
    return f"{v:.2f}"


def main() -> int:
    if not CSV.exists():
        print(f"missing {CSV}")
        return 1
    df = pd.read_csv(CSV)
    df = df[df["dataset"] == "modasense_emotion"] if "dataset" in df.columns else df

    power_cols = [c for c in df.columns if c.endswith("_power")]
    # prefer a readable order
    prefer = [
        "RFPerm_power",
        "DRPerm_power",
        "RRPerm_power",
        "C2ST_XY_power",
        "MMD_XY_power",
        "C2ST_X_power",
        "MMD_X_power",
        "CF_text_power",
        "CF_image_power",
        "CF_audio_power",
        "kernel_MMD_power",
        "classic_c2st_XY_power",
    ]
    cols = [c for c in prefer if c in power_cols] + [c for c in power_cols if c not in prefer]
    # shorten headers
    headers = [c.replace("_power", "").replace("_", "\\_") for c in cols]

    lines = [
        "% Auto-generated from benchmark_all_full.csv — do not edit by hand.",
        f"% B={int(df['B'].iloc[0]) if 'B' in df.columns else '?'} "
        f"scenarios={len(df)} methods={len(cols)}",
        "",
        "\\noindent",
        "ModaSense full method comparison. Outcome model: Random Forest. ",
        "Entries are empirical rejection rates (power / Type-I) at $\\alpha{=}0.05$.",
        "",
        "\\begin{center}",
        "\\small",
        "\\begin{tabular}{l" + "c" * len(cols) + "}",
        "\\toprule",
        "Scenario & " + " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        sc = str(r.get("scenario", "")).replace("_", "\\_")
        vals = " & ".join(_fmt(r.get(c)) for c in cols)
        lines.append(f"{sc} & {vals} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{center}",
        "",
    ]
    if "rho_po_mean" in df.columns:
        lines += [
            "\\paragraph{$\\rho_{\\mathrm{PO}}$ (mean).}",
            "\\begin{center}",
            "\\begin{tabular}{lc}",
            "\\toprule Scenario & $\\rho_{\\mathrm{PO}}$ \\\\",
            "\\midrule",
        ]
        for _, r in df.iterrows():
            sc = str(r.get("scenario", "")).replace("_", "\\_")
            lines.append(f"{sc} & {_fmt(r.get('rho_po_mean'))} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{center}", ""]

    TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {TEX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
