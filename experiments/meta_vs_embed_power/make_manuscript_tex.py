#!/usr/bin/env python3
"""Build the manuscript LaTeX (tables + reading) from merged Amazon / gender / bias results.

    python -m experiments.meta_vs_embed_power.make_manuscript_tex [OUT.tex]

The interpretive section is read verbatim from ``results/amazon/manuscript_reading.tex``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from experiments.meta_vs_embed_power.amazon_extract import OUT_DIR
from experiments.meta_vs_embed_power.pipeline import OUT
from experiments.meta_vs_embed_power.run_amazon_b50 import B, merge

ENCODERS = (("MiniLM_L6", "MiniLM-L6"), ("qwen3-0.6b", "Qwen3-Emb-0.6B"))
TESTS = (("RFPerm", "RF"), ("CFPerm", "CF"), ("RRPerm", "RR"), ("DRPerm", "DR"))
SCEN_LABEL = {
    "sentiment_meta": "Sentiment (VADER tail)",
    "syntax_meta": "Syntax (verb/noun tail)",
    "train_vs_heldout": "Time (early vs late)",
    "domain_shift": "Domain (Gift $\\to$ Magazine)",
}
FS = ("without_meta", "with_meta", "meta_only")
DEFAULT_OUT = Path("/cursor/stores/bc-87419b75-0666-4144-88dc-9832a2b2d6ae/docs/meta-vs-embed-manuscript.tex")


def _p(v, bold_at: float = 0.80) -> str:
    if v is None:
        return "--"
    s = f"{v:.2f}".replace("0.", ".", 1) if v < 1 else "1.00"
    return f"$\\mathbf{{{s}}}$" if v >= bold_at else f"${s}$"


def _d(a, b) -> str:
    if a is None or b is None:
        return "--"
    d = a - b
    s = f"{abs(d):.2f}".replace("0.", ".", 1) if abs(d) < 1 else "1.00"
    sign = "+" if d > 0 else ("-" if d < 0 else "\\pm")
    return f"${sign}{s}$"


def _by(payload: dict, ds: str) -> dict:
    return {(r["scenario"], r["feature_set"]): r["power"] for r in payload["tables"].get(ds, [])}


def _wresid(ds: str) -> dict:
    out: dict = {}
    for ck in OUT_DIR.glob(f"checkpoint_amazon_*_{ds}.npz"):
        for r in np.load(ck, allow_pickle=True)["rows"]:
            dr = r.get("DRPerm", {})
            if "abs_w_resid_mean" in dr:
                out.setdefault((r["scenario"], r["feature_set"]), []).append(dr["abs_w_resid_mean"])
    return {k: float(np.mean(v)) for k, v in out.items()}


def amazon_table(payload: dict) -> tuple[str, str]:
    bys = {e: {**_by(payload, f"amazon_{e}"), **_by(payload, f"amazon-cross_{e}")} for e, _ in ENCODERS}
    rows = []
    for scen, label in SCEN_LABEL.items():
        first = True
        for t, tl in TESTS:
            cells = []
            for e, _ in ENCODERS:
                p = {fs: bys[e].get((scen, fs), {}).get(t) for fs in FS}
                cells += [_p(p["without_meta"]), _p(p["with_meta"]), _d(p["with_meta"], p["without_meta"]), _p(p["meta_only"])]
            rows.append(f"{label if first else ''} & {tl} & " + " & ".join(cells) + " \\\\")
            first = False
        rows.append("\\midrule")
    rows = rows[:-1]
    nulls = {}
    for e, _ in ENCODERS:
        for fs in FS:
            for t, tl in TESTS:
                v = bys[e].get(("null_shift", fs), {}).get(t)
                if v is not None:
                    nulls.setdefault(tl, []).append(v)
    null_line = "; ".join(f"{tl} ${min(v):.2f}$--${max(v):.2f}$" for tl, v in nulls.items())
    head = " & ".join(f"\\multicolumn{{4}}{{c}}{{{lab}}}" for _, lab in ENCODERS)
    cmid = "\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}"
    sub = " & ".join(["w/o & with & $\\Delta$ & meta"] * len(ENCODERS))
    body = "\n".join(rows)
    return (
        f"""\\begin{{tabular}}{{llcccccccc}}
\\toprule
 & & {head} \\\\
{cmid}
Shift & Test & {sub} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}""",
        null_line,
    )


def dr_absorb_table() -> str:
    rows = []
    for scen, label in SCEN_LABEL.items():
        cells = []
        for e, _ in ENCODERS:
            ds = f"amazon-cross_{e}" if scen == "domain_shift" else f"amazon_{e}"
            w = _wresid(ds)
            cells += [("--" if (scen, fs) not in w else f"${w[(scen, fs)]:.2f}$") for fs in FS]
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    head = " & ".join(f"\\multicolumn{{3}}{{c}}{{{lab}}}" for _, lab in ENCODERS)
    return f"""\\begin{{tabular}}{{lcccccc}}
\\toprule
 & {head} \\\\
\\cmidrule(lr){{2-4}}\\cmidrule(lr){{5-7}}
Shift & w/o & with & meta & w/o & with & meta \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}"""


def shift_table() -> str:
    ev = json.loads((OUT_DIR / "shift_evidence_amazon.json").read_text(encoding="utf-8"))
    rows = []
    for key, label in (("gift", "Time (Gift, early vs late)"), ("cross", "Domain (Gift vs Magazine)")):
        e = ev[key]
        f = e["meta_all"]["features"]
        auc = e["group_auc"]
        rows.append(
            f"{label} & ${e['y_mean_by_group'][0]:.2f}/{e['y_mean_by_group'][1]:.2f}$"
            f" & ${e['meta_all']['n_ks_sig_bonf']}/{e['meta_all']['n_features']}$"
            f" & ${e['meta_all']['mean_abs_smd']:.2f}$"
            f" & ${f['char_len']['median_a']:.0f}\\to{f['char_len']['median_b']:.0f}$ (${f['char_len']['smd']:+.2f}$)"
            f" & ${f['vader_compound']['median_a']:.2f}\\to{f['vader_compound']['median_b']:.2f}$ (${f['vader_compound']['smd']:+.2f}$)"
            f" & ${auc.get('meta_only', float('nan')):.2f}$"
            f" & ${auc.get('MiniLM_L6_pca24', float('nan')):.2f}$"
            f" & ${auc.get('qwen3-0.6b_pca24', float('nan')):.2f}$ \\\\"
        )
    return f"""\\begin{{tabular}}{{lcccccccc}}
\\toprule
Shift & $\\bar y$ (a/b) & KS sig. & $\\overline{{|\\mathrm{{SMD}}|}}$ & char\\_len median & VADER median & AUC$_M$ & AUC$_{{\\mathrm{{MiniLM}}}}$ & AUC$_{{\\mathrm{{Qwen}}}}$ \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}"""


def absorption_table() -> str:
    path = OUT_DIR / "absorption_evidence_amazon.json"
    if not path.is_file():
        return "% absorption evidence not computed"
    ev = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for scen, label in SCEN_LABEL.items():
        w = ev["windows"].get(scen)
        if not w:
            continue
        cells = []
        for s in ("Z", "M", "S"):
            cells += [f"${w[s]['W_auc']:.2f}$", f"${w[s]['W_r2']:.2f}$", f"${w[s]['e_extreme_share']:.2f}$"]
        cells += [f"${w['Z+M']['var_resid_ratio_vs_Z']:.2f}$", f"${w['Z+S']['var_resid_ratio_vs_Z']:.2f}$"]
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    yrows = []
    for spec, label in (("gift", "Gift (time / sentiment / syntax)"), ("cross", "Gift $\\cup$ Magazine (domain)")):
        a = ev["y"][spec]["y_auc"]
        yrows.append(f"{label} & " + " & ".join(f"${a[s]:.2f}$" for s in ("Z", "M", "S", "Z+M", "Z+S")) + " \\\\")
    return f"""\\begin{{tabular}}{{lccccccccccc}}
\\toprule
 & \\multicolumn{{3}}{{c}}{{$Z_{{\\mathrm{{pca}}}}$}} & \\multicolumn{{3}}{{c}}{{$M$ (VADER+spaCy)}} & \\multicolumn{{3}}{{c}}{{$S$ (spaCy only)}} & \\multicolumn{{2}}{{c}}{{$\\mathrm{{Var}}(W-\\hat e)$ ratio}} \\\\
\\cmidrule(lr){{2-4}}\\cmidrule(lr){{5-7}}\\cmidrule(lr){{8-10}}\\cmidrule(lr){{11-12}}
Shift & AUC$_W$ & $R^2_W$ & ext. & AUC$_W$ & $R^2_W$ & ext. & AUC$_W$ & $R^2_W$ & ext. & $[Z,M]/Z$ & $[Z,S]/Z$ \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}

\\vspace{{0.6em}}
\\begin{{tabular}}{{lccccc}}
\\toprule
$y$ CV AUC & $Z_{{\\mathrm{{pca}}}}$ & $M$ & $S$ & $[Z,M]$ & $[Z,S]$ \\\\
\\midrule
{chr(10).join(yrows)}
\\bottomrule
\\end{{tabular}}"""


def _fork(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    t = d["summary"]["table"] if "summary" in d else [e for v in d["by_scenario"].values() for e in v]
    return {(r["scenario"], r["feature_set"]): r["power"] for r in t}


def appendix_table() -> str:
    rows = []
    for ds, path in (("gender", OUT / "power_gender_B50.json"), ("bias", OUT / "power_bias_B50.json")):
        if not path.is_file():
            continue
        by = _fork(path)
        first = True
        for t, tl in TESTS:
            p = {fs: by.get(("sentiment_meta", fs), {}).get(t) for fs in FS}
            mark = "$^\\dagger$" if t == "DRPerm" else ("$^\\ddagger$" if t == "CFPerm" else "")
            rows.append(
                f"{ds if first else ''} & {tl}{mark} & {_p(p['without_meta'])} & {_p(p['with_meta'])} & "
                f"{_d(p['with_meta'], p['without_meta'])} & {_p(p['meta_only'])} \\\\"
            )
            first = False
        rows.append("\\midrule")
    return f"""\\begin{{tabular}}{{llcccc}}
\\toprule
Dataset & Test & w/o & with & $\\Delta$ & meta \\\\
\\midrule
{chr(10).join(rows[:-1])}
\\bottomrule
\\end{{tabular}}"""


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    payload = merge([e for e, _ in ENCODERS])
    t1, null_line = amazon_table(payload)
    align = {s: json.loads((OUT_DIR / f).read_text(encoding="utf-8"))
             for s, f in (("gift", "alignment_report.json"), ("cross", "alignment_report_cross.json"))}
    reading = (OUT_DIR / "manuscript_reading.tex").read_text(encoding="utf-8")
    tex = f"""\\documentclass[11pt]{{article}}
\\usepackage[margin=0.7in]{{geometry}}
\\usepackage{{booktabs,amsmath,graphicx}}
\\usepackage[T1]{{fontenc}}

\\title{{Text embedding $\\pm$ meta-features: three-arm shift-detection power\\\\
{{\\large Amazon Reviews 2023 (Gift Cards, Magazine Subscriptions), $B{{=}}{B}$}}}}
\\author{{SAPerm shift-detect}}
\\date{{}}

\\begin{{document}}
\\maketitle

\\begin{{abstract}}
\\noindent
Three feature arms per encoder: \\emph{{w/o}} $=$ PCA$_{{24}}$ of the sentence embedding $Z$;
\\emph{{with}} $=[\\mathrm{{PCA}}_{{24}}(Z),\\,M]$; \\emph{{meta}} $=M$ only, where $M$ is the
32-dimensional \\texttt{{utils.text\\_meta}} vector (surface 14, syntax 12, sentiment 6).
$B{{=}}{B}$ replicates, $N_{{\\mathrm{{perm}}}}{{=}}150$, Random Forest outcome model,
$n_{{\\mathrm{{doc}}}}{{=}}120$ per window, $\\alpha{{=}}0.05$; entries are empirical rejection rates.
RF $=$ RFPerm (Python port of \\texttt{{RFPerm\\_Generalized.PermValTest}}, 70/30 train/val split);
CF $=$ CFPerm LOCO on the meta block (meta arm: per sub-block, Bonferroni over 3 blocks);
RR $=$ RRPerm; DR $=$ DRPerm with residual-permutation null.
Source domain: 1{{,}}000 Gift Cards reviews (rating 3 dropped, 20--1000 characters,
text-deduplicated, $y=\\mathbb{{1}}[\\text{{rating}}\\ge4]$, 500/500); target domain:
1{{,}}000 Magazine Subscriptions reviews under the same filters, disjoint texts.
Encoders: all-MiniLM-L6-v2 (384-d) and Qwen3-Embedding-0.6B (1024-d, fp32).
Every artifact is keyed by row id and text SHA-1; alignment passed on both frames
(frame hashes \\texttt{{{align['gift']['frame_hash'][:8]}}}, \\texttt{{{align['cross']['frame_hash'][:8]}}}).
\\end{{abstract}}

\\begin{{table*}}[ht]
\\centering
\\caption{{Amazon power by arm ($B{{=}}{B}$). $\\Delta=$ with $-$ w/o.
Sentiment / syntax windows select the tails of a meta coordinate (VADER compound,
verb/noun ratio); time compares early vs late Gift Cards reviews; domain compares
Gift Cards with Magazine Subscriptions.{(' Random split (Type-I) rejection rates over arms and encoders: ' + null_line + '.') if null_line else ''}
``--'' $=$ not run.}}
\\label{{tab:amazon_power}}
\\resizebox{{\\textwidth}}{{!}}{{%
{t1}%
}}
\\end{{table*}}

\\begin{{table}}[ht]
\\centering
\\caption{{DRPerm absorption diagnostic: mean $|W-\\hat e(X)|$ (cross-fitted).
$0.5$ means the window label $W$ is unpredictable from $X$; small values mean $X$
already encodes the window, so the shift is absorbed into $\\hat e$ rather than
left in the outcome residual.}}
\\label{{tab:amazon_dr_absorb}}
{dr_absorb_table()}
\\end{{table}}

\\begin{{table*}}[ht]
\\centering
\\caption{{Absorption evidence (MiniLM-L6; same windows and cross-fitted RF propensity as DRPerm,
10 windows per shift, $n_{{\\mathrm{{doc}}}}{{=}}120$ per side).
AUC$_W$ / $R^2_W=1-\\mathrm{{Brier}}/\\mathrm{{Var}}(W)$: how well the feature set predicts the window label $W$;
ext.\\ $=$ share of raw $\\hat e(X)<0.02$ or $>0.98$ (positivity); ratio $=$ variance of the DR residual
$W-\\hat e$ (clipped at $0.05$) relative to $Z_{{\\mathrm{{pca}}}}$ alone.
Bottom: 5-fold CV AUC for $y$ on the full frame.}}
\\label{{tab:amazon_absorption}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{minipage}}{{1.25\\textwidth}}\\centering
{absorption_table()}
\\end{{minipage}}%
}}
\\end{{table*}}

\\begin{{table*}}[ht]
\\centering
\\caption{{Shift diagnostics on $M$ and embeddings. KS sig.\\ $=$ meta features with
Bonferroni-significant two-sample KS; SMD $=$ standardized mean difference (b $-$ a);
AUC $=$ 5-fold logistic classifier separating the two groups.}}
\\label{{tab:amazon_shift_diag}}
\\resizebox{{\\textwidth}}{{!}}{{%
{shift_table()}%
}}
\\end{{table*}}

\\begin{{table}}[ht]
\\centering
\\caption{{Appendix: gender and bias, sentiment window, MiniLM-L6, $B{{=}}50$ (same RF backend).
$^\\dagger$DR from the earlier refit null (biased toward $p\\to1$ when $W$ is predictable);
residual-null rerun pending.
$^\\ddagger$meta-arm CF is whole-block LOCO (before the per-sub-block Bonferroni).}}
\\label{{tab:appendix_gender_bias}}
{appendix_table()}
\\end{{table}}

{reading}
\\end{{document}}
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    (OUT_DIR / "meta-vs-embed-manuscript.tex").write_text(tex, encoding="utf-8")
    print(f"Wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
