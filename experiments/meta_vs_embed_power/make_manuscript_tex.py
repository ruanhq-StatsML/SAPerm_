#!/usr/bin/env python3
"""Build the manuscript LaTeX (tables + reading) from merged Amazon / gender / bias results.

    python -m experiments.meta_vs_embed_power.make_manuscript_tex [OUT.tex]

The interpretive section is read verbatim from ``results/amazon/manuscript_reading.tex``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from experiments.meta_vs_embed_power.amazon_extract import OUT_DIR
from experiments.meta_vs_embed_power.run_amazon_b50 import B, merge

ENCODERS = (("MiniLM_L6", "MiniLM-L6"), ("qwen3-0.6b", "Qwen3-Emb-0.6B"))
TESTS = (("CFPerm", "CF"), ("RRPerm", "RR"), ("DRPerm", "DR"))
SCEN_LABEL = {
    "sentiment_meta": "Sentiment (VADER tail)",
    "syntax_meta": "Syntax (verb/noun tail)",
    "train_vs_heldout": "Time (early vs late)",
    "domain_shift": "Domain (Gift $\\to$ Magazine)",
}
FS = ("without_meta", "with_meta", "meta_only")
FS4 = ("without_meta", "with_meta", "with_spacy", "meta_only")
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
                p = {fs: bys[e].get((scen, fs), {}).get(t) for fs in FS4}
                cells += [
                    _p(p["without_meta"]), _p(p["with_meta"]), _d(p["with_meta"], p["without_meta"]),
                    _p(p["with_spacy"]), _d(p["with_spacy"], p["without_meta"]), _p(p["meta_only"]),
                ]
            rows.append(f"{label if first else ''} & {tl} & " + " & ".join(cells) + " \\\\")
            first = False
        rows.append("\\midrule")
    rows = rows[:-1]
    nulls = {}
    for e, _ in ENCODERS:
        for fs in FS4:
            for t, tl in TESTS:
                v = bys[e].get(("null_shift", fs), {}).get(t)
                if v is not None:
                    nulls.setdefault(tl, []).append(v)
    null_line = "; ".join(f"{tl} ${min(v):.2f}$--${max(v):.2f}$" for tl, v in nulls.items())
    head = " & ".join(f"\\multicolumn{{6}}{{c}}{{{lab}}}" for _, lab in ENCODERS)
    cmid = "\\cmidrule(lr){3-8}\\cmidrule(lr){9-14}"
    sub = " & ".join(["w/o & $M$ & $\\Delta_M$ & $S$ & $\\Delta_S$ & $M$ only"] * len(ENCODERS))
    body = "\n".join(rows)
    return (
        f"""\\begin{{tabular}}{{llcccccccccccc}}
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


RF_SCEN = (("null_shift", "Null (random split)"),) + tuple(SCEN_LABEL.items())


def rf_b150_table() -> str:
    pay = json.loads((OUT_DIR / "power_amazon_rf_b150.json").read_text(encoding="utf-8"))
    rnd, ordr = pay["splits"]["random"], pay["splits"]["ordered"]
    rows = []
    for scen, label in RF_SCEN:
        first = True
        for e, lab in ENCODERS:
            ds = f"amazon-cross_{e}" if scen == "domain_shift" else f"amazon_{e}"
            if scen == "null_shift":
                ds = f"amazon_{e}"
            g = lambda src, a: (src.get(ds, {}).get(f"{scen}|{a}") or {}).get("power")
            p = {a: g(rnd, a) for a in ("without_meta", "with_meta", "with_spacy", "meta_only")}
            o = {a: g(ordr, a) for a in ("without_meta", "with_meta", "with_spacy")}
            rows.append(
                f"{label if first else ''} & {lab} & {_p(p['without_meta'])} & {_p(p['with_meta'])} & "
                f"{_p(p['with_spacy'])} & {_p(p['meta_only'])} & {_d(p['with_meta'], p['without_meta'])} & "
                f"{_d(p['with_spacy'], p['without_meta'])} & {_d(o['with_meta'], o['without_meta'])} & "
                f"{_d(o['with_spacy'], o['without_meta'])} \\\\"
            )
            first = False
        rows.append("\\midrule")
    cross_null = []
    for e, lab in ENCODERS:
        c = rnd.get(f"amazon-cross_{e}", {})
        cross_null += [c[k]["power"] for k in c if k.startswith("null_shift|")]
    return f"""\\begin{{tabular}}{{llcccccccc}}
\\toprule
 & & \\multicolumn{{4}}{{c}}{{Power (random split)}} & \\multicolumn{{2}}{{c}}{{$\\Delta$ random}} & \\multicolumn{{2}}{{c}}{{$\\Delta$ ordered}} \\\\
\\cmidrule(lr){{3-6}}\\cmidrule(lr){{7-8}}\\cmidrule(lr){{9-10}}
Shift & Encoder & w/o & with $M$ & with $S$ & $M$ only & $M$ & $S$ & $M$ & $S$ \\\\
\\midrule
{chr(10).join(rows[:-1])}
\\bottomrule
\\end{{tabular}}""", (min(cross_null), max(cross_null))


DECORR_ARMS = (
    ("without_meta", "w/o", "Z"),
    ("with_meta", "with $M$", "Z+M"),
    ("with_meta_decorr", "$M_{0.3}$", "Z+Mk"),
    ("with_meta_decorr_t02", "$M_{0.2}$", "Z+Mk2"),
    ("with_meta_decorr_t02nv", "$M_{0.2}\\setminus\\{v,n\\}$", "Z+Mk2nv"),
    ("with_spacy_decorr", "$S_{0.3}$", "Z+Sk"),
    ("with_spacy_decorr_t02", "$S_{0.2}$", "Z+Sk2"),
)


def decorr_falsify_table() -> str | None:
    ds = "amazon_MiniLM_L6"
    tags = ("py", "decorr", "decorr_cf")
    hits: dict = {}
    for tag in tags:
        ck = OUT_DIR / f"checkpoint_amazon_{tag}_{ds}.npz"
        if not ck.is_file():
            continue
        for r in np.load(ck, allow_pickle=True)["rows"]:
            for t in ("DRPerm", "RRPerm", "CFPerm"):
                x = r.get(t)
                if x and not x.get("skipped") and x.get("p_value") is not None:
                    hits.setdefault((r["scenario"], r["feature_set"], t), {})[r["b"]] = x["p_value"] < 0.05
    if not any(k[1].endswith("decorr") for k in hits):
        return None
    ab = json.loads((OUT_DIR / "absorption_evidence_amazon.json").read_text(encoding="utf-8"))["windows"]
    rows = []
    for scen in ("sentiment_meta", "syntax_meta"):
        first = True
        for fs, lab, s in DECORR_ARMS:
            pw = [(lambda d: float(np.mean(list(d.values()))) if len(d) >= B else None)(hits.get((scen, fs, t), {}))
                  for t in ("DRPerm", "RRPerm", "CFPerm")]
            a = ab.get(scen, {}).get(s)
            diag = [f"${a['W_auc']:.2f}$", f"${a['var_resid_ratio_vs_Z']:.2f}$"] if a else ["--", "--"]
            rows.append(f"{SCEN_LABEL[scen] if first else ''} & {lab} & " + " & ".join(diag + [_p(v) for v in pw]) + " \\\\")
            first = False
        rows.append("\\midrule")
    return f"""\\begin{{tabular}}{{llccccc}}
\\toprule
Shift & Arm & AUC$_W$ & $\\mathrm{{Var}}(W-\\hat e)$ ratio & DR & RR & CF \\\\
\\midrule
{chr(10).join(rows[:-1])}
\\bottomrule
\\end{{tabular}}"""


GENDER_ROWS = (
    # scenario, arm, RF (B=150, upper tail, random split), DR, RR, CF (B=50)
    ("Sentiment", "w/o", .100, .88, 1.00, .92),
    ("", "with $M$", .033, .08, .90, .98),
    ("", "$M$ only", .027, .04, .94, 1.00),
    ("Syntax", "w/o", .267, .60, .74, .60),
    ("", "with $M$", .360, .12, .32, .68),
    ("", "$M$ only", .347, .06, .36, .46),
    ("Train/held-out", "w/o", .047, .02, .00, .02),
    ("", "with $M$", .033, .02, .00, .04),
    ("", "$M$ only", .027, .04, .00, .00),
    ("Null", "w/o", .040, .12, .06, .06),
    ("", "with $M$", .067, .08, .02, .00),
    ("", "$M$ only", .060, .04, .06, .06),
)


def gender_table() -> str:
    rows = []
    for i, (scen, arm, *vals) in enumerate(GENDER_ROWS):
        if i and scen:
            rows.append("\\midrule")
        rows.append(f"{scen} & {arm} & " + " & ".join(_p(v, bold_at=2) for v in vals) + " \\\\")
    return f"""\\begin{{tabular}}{{llcccc}}
\\toprule
Shift & Arm & RF & DR & RR & CF \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}"""


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    paired = os.environ.get("MANUSCRIPT_B50") == "1"
    if paired:
        t1, null_line = amazon_table(merge([e for e, _ in ENCODERS]))
    rf_tab, (cn_lo, cn_hi) = rf_b150_table()
    align = {s: json.loads((OUT_DIR / f).read_text(encoding="utf-8"))
             for s, f in (("gift", "alignment_report.json"), ("cross", "alignment_report_cross.json"))}
    reading = (OUT_DIR / "manuscript_reading.tex").read_text(encoding="utf-8")
    b50_block = (
        f"""\\begin{{table*}}[ht]
\\centering
\\caption{{Amazon DR / RR / CF power by arm ($B{{=}}{B}$, paired windows). $\\Delta_M$, $\\Delta_S$ $=$ arm $-$ w/o.
{('Null rejection rates: ' + null_line + '.') if null_line else ''}}}
\\label{{tab:amazon_power_b50}}
\\resizebox{{\\textwidth}}{{!}}{{%
{t1}%
}}
\\end{{table*}}

\\begin{{table}}[ht]
\\centering
\\caption{{DRPerm absorption diagnostic: mean $|W-\\hat e(X)|$ (cross-fitted).}}
\\label{{tab:amazon_dr_absorb}}
{dr_absorb_table()}
\\end{{table}}
"""
        if paired
        else "% Amazon DR / RR / CF paired table (B=50): added when run_amazon_chain.sh finishes.\n"
    )
    dtab = decorr_falsify_table()
    decorr_block = (
        f"""\\begin{{table}}[ht]
\\centering
\\caption{{Decorrelated arms (MiniLM-L6, Gift Cards, $B{{=}}{B}$; same paired windows as the main chain).
$M_\\tau$ and $S_\\tau$ remove features with full-frame $|\\rho_{{\\mathrm{{Spearman}}}}(f,y)|\\ge\\tau$.
This is a diagnostic ablation, since the filter uses $y$.
$M_{{0.2}}\\setminus\\{{v,n\\}}$ also removes \\texttt{{verb\\_ratio}} and \\texttt{{noun\\_ratio}}, the two components of
the syntax selection variable. AUC$_W$ and the residual-variance ratio are defined as in
Table~\\ref{{tab:amazon_absorption}}.}}
\\label{{tab:amazon_decorr}}
{dtab}
\\end{{table}}
"""
        if dtab
        else ""
    )
    tex = f"""\\documentclass[11pt]{{article}}
\\usepackage[margin=0.7in]{{geometry}}
\\usepackage{{booktabs,amsmath,graphicx}}
\\usepackage[T1]{{fontenc}}

\\title{{Text embedding $\\pm$ meta-features: shift-detection power\\\\
{{\\large Amazon Reviews 2023 (Gift Cards, Magazine Subscriptions) and gender}}}}
\\author{{SAPerm shift-detect}}
\\date{{}}

\\begin{{document}}
\\maketitle

\\begin{{abstract}}
\\noindent
Four feature arms per encoder: \\emph{{w/o}} $=$ PCA$_{{24}}$ of the sentence embedding $Z$;
\\emph{{with $M$}} $=[\\mathrm{{PCA}}_{{24}}(Z),\\,M]$, where $M$ is the 32-dimensional
\\texttt{{utils.text\\_meta}} vector (surface, syntax and VADER sentiment features);
\\emph{{with $S$}} $=[\\mathrm{{PCA}}_{{24}}(Z),\\,S]$, where $S$ is a 36-dimensional spaCy
(\\texttt{{en\\_core\\_web\\_sm}}) block with no VADER, sentiment lexicon or verb/noun ratio;
\\emph{{$M$ only}}. $n_{{\\mathrm{{doc}}}}{{=}}120$ per window, $N_{{\\mathrm{{perm}}}}{{=}}150$,
Random Forest outcome model, $\\alpha{{=}}0.05$; entries are empirical rejection rates, and all arms
of a replicate share the same windows.
RF $=$ RFPerm, the Python \\texttt{{RFPerm\\_Generalized.PermValTest}} port with $B{{=}}150$ and
the one-sided upper-tail $p$ of the official \\texttt{{RFPerm::permrf}}
(reject when the new window has higher held-out loss).
CF $=$ CFPerm LOCO on the meta block; RR $=$ RRPerm; DR $=$ DRPerm with residual-permutation null ($B{{=}}50$).
Source domain: 1{{,}}000 Gift Cards reviews (rating 3 dropped, 20--1000 characters,
text-deduplicated, $y=\\mathbb{{1}}[\\text{{rating}}\\ge4]$, 500/500); target domain:
1{{,}}000 Magazine Subscriptions reviews under the same filters, disjoint texts.
Encoders: all-MiniLM-L6-v2 (384-d) and Qwen3-Embedding-0.6B (1024-d, fp32).
Every artifact is keyed by row id and text SHA-1; alignment passed on both frames
(frame hashes \\texttt{{{align['gift']['frame_hash'][:8]}}}, \\texttt{{{align['cross']['frame_hash'][:8]}}}).
\\end{{abstract}}

\\begin{{table*}}[ht]
\\centering
\\caption{{Amazon RFPerm power, $B{{=}}150$, $N_{{\\mathrm{{perm}}}}{{=}}150$, upper-tail $p$.
Sentiment / syntax windows select the tails of a coordinate of $M$ (VADER compound,
verb/noun ratio); time compares early with late Gift Cards reviews; domain compares
Gift Cards with Magazine Subscriptions. $\\Delta=$ arm $-$ w/o.
Random: the official random 70/30 train/validation split of the first window;
ordered: the same rows in frame order, so the split is chronological on Gift Cards.
Cross-frame null rejection rates: ${cn_lo:.2f}$--${cn_hi:.2f}$.
Standard error of a cell $\\le 0.04$.}}
\\label{{tab:amazon_rf_b150}}
\\resizebox{{\\textwidth}}{{!}}{{%
{rf_tab}%
}}
\\end{{table*}}

{b50_block}
{decorr_block}
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
\\caption{{Gender, MiniLM-L6, $n{{=}}1000$ (balanced $y$). RF: $B{{=}}150$, upper-tail $p$,
random split. DR (residual null), RR, CF: $B{{=}}50$; $M$-only CF is per-sub-block LOCO
with Bonferroni over three blocks.}}
\\label{{tab:gender_power}}
{gender_table()}
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
