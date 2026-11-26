"""Summarize ``results/dim_scan_gender.jsonl`` (+ ``dim_scan_gender_diag.json``) into markdown tables.

    python3 -m experiments.meta_vs_embed_power.summarize_dim_scan  # writes dim_scan_gender_summary.{json,md}
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "results"
SCEN = ("null_shift", "sentiment_meta", "syntax_meta", "train_vs_heldout")
K_ORDER = ("24", "64", "128", "full")
EMBS = ("MiniLM_L6", "qwen3-0.6b")
KINDS = ("without", "with", "spacy", "with_decorr")
VADER_COLS = ("without", "with", "vader", "v_compound", "v_neg", "v_pos", "v_polar_gap", "with_decorr",
              "spacy", "vader_spacy")
TESTS = {"rf": ("RF_random", "RF_ordered"), "drc": ("DR", "RR", "CF")}
LABEL = {"without": "without_meta", "with": "with_meta", "spacy": "with_spacy", "with_decorr": "with_meta_decorr",
         "vader": "Z+VADER4", "v_compound": "Z+compound", "v_neg": "Z+neg", "v_pos": "Z+pos",
         "v_polar_gap": "Z+polar_gap", "vader_spacy": "Z+VADER4+S"}


def load() -> list[dict]:
    """Merge every ``dim_scan_gender*.jsonl`` (one per machine); rows are deduplicated by ``key``."""
    rows = {}
    for path in sorted(OUT.glob("dim_scan_gender*.jsonl")):
        for line in path.open():
            if line.strip():
                r = json.loads(line)
                rows[r["key"]] = r
    return list(rows.values())


def cells(rows: list[dict]) -> dict:
    g: dict = defaultdict(list)
    for r in rows:
        g[(r["tests"], r.get("emb", "MiniLM_L6"), r["kind"], r["k"], r["mtry"], r["scenario"])].append(r)
    out = {}
    for (tests, emb, kind, k, mtry, sc), rs in g.items():
        for t in TESTS[tests]:
            p = np.array([r[t]["p_value"] for r in rs])
            ent = {"B": len(rs), "power": float(np.mean(p < 0.05)), "mean_p": float(np.mean(p))}
            if "p_value_abs" in rs[0][t]:
                ent["power_abs"] = float(np.mean(np.array([r[t]["p_value_abs"] for r in rs]) < 0.05))
            if "null_sd" in rs[0][t]:
                z = [(r[t]["statistic"] - r[t]["null_mean"]) / max(r[t]["null_sd"], 1e-15) for r in rs]
                ent["z_obs_vs_null"] = float(np.mean(z))
                ent["null_mean"] = float(np.mean([r[t]["null_mean"] for r in rs]))
                ent["stat_mean"] = float(np.mean([r[t]["statistic"] for r in rs]))
            if "var_w_resid" in rs[0][t]:
                ent["var_w_resid"] = float(np.mean([r[t]["var_w_resid"] for r in rs]))
                ent["abs_w_resid"] = float(np.mean([r[t]["abs_w_resid_mean"] for r in rs]))
            out[f"{t}|{emb}|{kind}|{k}|{mtry}|{sc}"] = ent
    return out


def power_table(c: dict, test: str, emb: str, mtry=None, ks=K_ORDER) -> str:
    kinds = [kd for kd in KINDS if any(f"{test}|{emb}|{kd}|{k}|{mtry}|" in key for key in c for k in ks)]
    if not kinds:
        return ""
    head = "| 场景 | k | " + " | ".join(f"{LABEL[kd]}" for kd in kinds) + " | Δ with−without | Δ spacy−without | B |"
    lines = [head, "|---|---|" + "--:|" * (len(kinds) + 3)]
    for sc in SCEN:
        for k in ks:
            e = {kd: c.get(f"{test}|{emb}|{kd}|{k}|{mtry}|{sc}") for kd in kinds}
            if not any(e.values()):
                continue
            fmt = lambda x: f"{x['power']:.3f} ({x['mean_p']:.3f})" if x else "—"
            d = lambda a, b: f"{e[a]['power'] - e[b]['power']:+.3f}" if e.get(a) and e.get(b) else "—"
            bs = sorted({x["B"] for x in e.values() if x})
            lines.append(f"| {sc} | {k} | " + " | ".join(fmt(e[kd]) for kd in kinds)
                         + f" | {d('with', 'without')} | {d('spacy', 'without')} | {'/'.join(map(str, bs))} |")
    return "\n".join(lines)


def vader_table(c: dict, test: str, emb: str, ks=("24", "128", "full")) -> str:
    """Power of VADER sub-arms; additivity = Δ(VADER4+S) − [Δ(VADER4) + Δ(S)], Δ relative to without_meta."""
    cols = VADER_COLS
    head = ("| 场景 | k | " + " | ".join(LABEL[kd] for kd in cols)
            + " | Δ VADER4 | Δ M | Δ VADER4+S | 可加性残差 | B |")
    lines = [head, "|---|---|" + "--:|" * (len(cols) + 5)]
    for sc in SCEN:
        for k in ks:
            e = {kd: c.get(f"{test}|{emb}|{kd}|{k}|None|{sc}") for kd in cols}
            if not e.get("vader"):
                continue
            pw = lambda kd: e[kd]["power"] if e.get(kd) else None
            dlt = lambda kd: (pw(kd) - pw("without")) if pw(kd) is not None and pw("without") is not None else None
            f = lambda v: "—" if v is None else f"{v:+.3f}"
            add = None
            if None not in (dlt("vader_spacy"), dlt("vader"), dlt("spacy")):
                add = dlt("vader_spacy") - dlt("vader") - dlt("spacy")
            bs = sorted({x["B"] for x in e.values() if x})
            lines.append(f"| {sc} | {k} | " + " | ".join("—" if pw(kd) is None else f"{pw(kd):.3f}" for kd in cols)
                         + f" | {f(dlt('vader'))} | {f(dlt('with'))} | {f(dlt('vader_spacy'))} | {f(add)} | "
                         + "/".join(map(str, bs)) + " |")
    return "\n".join(lines)


def null_table(c: dict, emb: str) -> str:
    """DR/RR null behaviour per k: size, mean p and mean z of the observed statistic vs its null."""
    lines = ["| k | 臂 | DR size | DR 平均 p | DR z | RR size | RR 平均 p | RR z | Var(W−ê) (DR) | B |",
             "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for k in K_ORDER:
        for kd in KINDS:
            dr, rr = (c.get(f"{t}|{emb}|{kd}|{k}|None|null_shift") for t in ("DR", "RR"))
            if not dr:
                continue
            lines.append(f"| {k} | {LABEL[kd]} | {dr['power']:.2f} | {dr['mean_p']:.2f} | {dr['z_obs_vs_null']:+.2f} | "
                         f"{rr['power']:.2f} | {rr['mean_p']:.2f} | {rr['z_obs_vs_null']:+.2f} | "
                         f"{dr['var_w_resid']:.3f} | {dr['B']} |")
    return "\n".join(lines) if len(lines) > 2 else ""


def diag_tables(emb: str) -> str:
    p = OUT / "dim_scan_gender_diag.json"
    if not p.is_file() or emb not in json.loads(p.read_text()):
        return ""
    d = json.loads(p.read_text())[emb]
    ya, w = d["y_auc_5fold"], d["window"]
    kinds = [kd for kd in KINDS if f"{kd}|24" in ya]
    out = ["| k | " + " | ".join(f"y-AUC {LABEL[kd]}" for kd in kinds) + " |", "|---|" + "--:|" * len(kinds)]
    out += [f"| {k} | " + " | ".join(f"{ya[f'{kd}|{k}']:.3f}" for kd in kinds) + " |" for k in K_ORDER]
    out += ["", "| 场景 | k | AUC_W " + " / ".join(LABEL[kd] for kd in kinds) + " | Var(W−ê) "
            + " / ".join(LABEL[kd] for kd in kinds) + " | 比值 with/without | spacy/without | decorr/without |",
            "|---|---|---|---|--:|--:|--:|"]
    for sc in SCEN:
        for k in K_ORDER:
            v = {kd: w[f"{kd}|{k}|{sc}"] for kd in kinds}
            r = lambda a: f"{v[a]['var_w_resid'] / v['without']['var_w_resid']:.2f}" if a in v else "—"
            out.append(f"| {sc} | {k} | " + " / ".join(f"{v[kd]['auc_w']:.2f}" for kd in kinds) + " | "
                       + " / ".join(f"{v[kd]['var_w_resid']:.3f}" for kd in kinds)
                       + f" | {r('with')} | {r('spacy')} | {r('with_decorr')} |")
    return "\n".join(out)


def main() -> None:
    c = cells(load())
    (OUT / "dim_scan_gender_summary.json").write_text(json.dumps(c, indent=1, sort_keys=True))
    parts = []
    for emb in EMBS:
        for t in ("RF_random", "RF_ordered", "DR", "RR", "CF"):
            tab = power_table(c, t, emb)
            if tab:
                parts.append(f"#### {emb} · {t} 功效 (平均 p)\n\n{tab}")
        for t in ("RF_random", "RF_ordered", "DR", "RR", "CF"):
            if any(key.startswith(f"{t}|{emb}|vader|") for key in c):
                parts.append(f"#### {emb} · {t} · VADER 聚焦（功效）\n\n{vader_table(c, t, emb)}")
        for t in ("RF_random", "RF_ordered"):
            tab = power_table(c, t, emb, mtry=7, ks=("128", "full"))
            if tab:
                parts.append(f"#### {emb} · {t}，mtry 固定为 7\n\n{tab}")
        tab = null_table(c, emb)
        if tab:
            parts.append(f"#### {emb} · null_shift 下 DR/RR 的 size 与 null 行为\n\n{tab}")
        tab = diag_tables(emb)
        if tab:
            parts.append(f"#### {emb} · y-AUC、AUC_W、残差方差\n\n{tab}")
    md = "\n\n".join(parts) + "\n"
    (OUT / "dim_scan_gender_summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
