"""gender PCA-dimension scan: Z_k for k in {24, 64, 128, full} x arms, for MiniLM and Qwen embeddings.

Arms (``kind``): ``without`` = Z_k, ``with`` = [Z_k, M], ``spacy`` = [Z_k, S], and the diagnostic
``with_decorr`` / ``spacy_decorr`` = [Z_k, M_kept] / [Z_k, S_kept], where a column is dropped when
|spearman(column, y)| >= TAU on the 1000-row gender frame (same rule as ``decorr_features.py``).
Window selection always uses the full M (vader_compound, verb_noun_ratio).

Windows and seeds match ``worker.run_job`` (``2026 + 10007*(crc32(scenario)%997) + 17*b``), so every
arm, k and embedding sees the same window pair for a given (scenario, b). Rows are appended to
``results/dim_scan_gender.jsonl`` (resume-safe, keyed by [emb:]arm|scenario|b|tests).

``--tests rf``: RFPerm upper-tail p, random and ordered split in the same job.
``--tests drc``: DRPerm (residual null), RRPerm, CFPerm.
``--mtry N``: RF-only control arms with classifier mtry fixed to N.

Example::

    python3 -m experiments.meta_vs_embed_power.run_dim_scan --tests rf --ks full 128 --b0 0 --b1 150
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import spearmanr

from experiments.meta_vs_embed_power.data import load_gender, pca_embed
from experiments.meta_vs_embed_power.perm_lib import cfperm_loco_pvalue, drperm_pvalue, rrperm_pvalue
from experiments.meta_vs_embed_power.power import make_windows
from experiments.meta_vs_embed_power.rfperm_official import rfperm_permval_single

OUT = Path(__file__).resolve().parent / "results"
LOG = OUT / "dim_scan_gender.jsonl"
SCENARIOS = ("null_shift", "sentiment_meta", "syntax_meta", "train_vs_heldout")
KINDS = ("without", "with", "spacy", "with_decorr", "spacy_decorr")
SEED, N_DOC, N_EST, N_PERM, TAU = 2026, 120, 150, 150, 0.3
BASE_EMB = "MiniLM_L6"

_BASE: dict = {}
_Z: dict = {}


def load_base() -> dict:
    if _BASE:
        return _BASE
    _, y, texts = load_gender(1000, SEED)
    z = np.load(OUT / "meta_gender_n1000.npz", allow_pickle=True)
    M, names = np.asarray(z["M"], dtype=float), [str(n) for n in z["meta_names"]]
    sp_cache = OUT / "spacy_gender_n1000.npz"
    if sp_cache.is_file():
        zs = np.load(sp_cache, allow_pickle=True)
        S, s_names = zs["S"], [str(n) for n in zs["spacy_names"]]
    else:
        from experiments.meta_vs_embed_power.spacy_features import spacy_matrix

        S, s_names = spacy_matrix(texts)
        np.savez_compressed(sp_cache, S=S, spacy_names=np.array(s_names, dtype=object))
    S = np.asarray(S, dtype=float)

    def kept(block: np.ndarray) -> np.ndarray:
        rho = [0.0 if c.std() == 0 else abs(spearmanr(c, y).correlation) for c in block.T]
        return np.array([j for j, r in enumerate(rho) if r < TAU], dtype=int)

    _BASE.update(
        M=M, S=S, MS=np.hstack([M, S]), names=names, s_names=list(s_names), y=y,
        M_keep=kept(M), S_keep=kept(S),
    )
    return _BASE


def load_data(emb: str = BASE_EMB) -> dict:
    d = load_base()
    if emb not in _Z:
        Z = np.load(OUT / f"gender_{emb}_embedding.npy")
        assert len(Z) == len(d["y"]), (emb, Z.shape)
        _Z[emb] = {"full": Z}
    return d


def z_k(k: str, emb: str = BASE_EMB) -> np.ndarray:
    load_data(emb)
    zs = _Z[emb]
    if k not in zs:
        zs[k] = pca_embed(zs["full"], int(k), seed=SEED)
    return zs[k]


def rep_seed(scenario: str, b: int) -> int:
    return SEED + 10_007 * (zlib.crc32(scenario.encode()) % 997) + 17 * b


def arm_name(kind: str, k: str, mtry: int | None, emb: str = BASE_EMB) -> str:
    base = {"without": "without_meta", "with": "with_meta", "spacy": "with_spacy",
            "with_decorr": "with_meta_decorr", "spacy_decorr": "with_spacy_decorr"}[kind]
    pre = "" if emb == BASE_EMB else f"{emb}:"
    return f"{pre}{base}_k{k}" + (f"_mtry{mtry}" if mtry else "")


def build(kind: str, k: str, scenario: str, b: int, emb: str = BASE_EMB):
    d = load_data(emb)
    Zk = z_k(k, emb)
    seed = rep_seed(scenario, b)
    rng = np.random.default_rng(seed)
    # selection columns (vader_compound, verb_noun_ratio) sit in the first n_meta columns
    Ze, ye, Zn, yn, _, _, MSe, MSn = make_windows(scenario, Zk, d["MS"], d["y"], d["names"], N_DOC, rng)
    n_meta = d["M"].shape[1]
    blocks = {
        "with": (MSe[:, :n_meta], MSn[:, :n_meta]),
        "spacy": (MSe[:, n_meta:], MSn[:, n_meta:]),
        "with_decorr": (MSe[:, d["M_keep"]], MSn[:, d["M_keep"]]),
        "spacy_decorr": (MSe[:, n_meta + d["S_keep"]], MSn[:, n_meta + d["S_keep"]]),
    }
    dz = Zk.shape[1]
    if kind == "without":
        Xe, Xn = Ze, Zn
        test_idx = np.arange(dz - min(8, n_meta), dz)
    else:
        Be, Bn = blocks[kind]
        Xe, Xn = np.hstack([Ze, Be]), np.hstack([Zn, Bn])
        test_idx = np.arange(dz, dz + Be.shape[1])
    return Xe, ye, Xn, yn, test_idx, seed


def run_job(kind: str, k: str, mtry: int | None, scenario: str, b: int, tests: str,
            emb: str = BASE_EMB) -> dict:
    t0 = time.time()
    Xe, ye, Xn, yn, test_idx, seed = build(kind, k, scenario, b, emb)
    row = {
        "key": f"{arm_name(kind, k, mtry, emb)}|{scenario}|{b}|{tests}",
        "arm": arm_name(kind, k, mtry, emb), "emb": emb, "kind": kind, "k": k, "dim": int(Xe.shape[1]),
        "mtry": mtry, "scenario": scenario, "b": b, "seed": seed, "tests": tests,
    }
    keep = ("p_value", "p_value_upper", "p_value_lower", "p_value_abs", "statistic", "reject",
            "null_mean", "null_sd", "abs_w_resid_mean", "var_w_resid", "r_risk_full", "r_risk_minus")
    slim = lambda r: {key: (bool(v) if key == "reject" else float(v)) for key, v in r.items() if key in keep}
    if tests == "rf":
        for split in ("random", "ordered"):
            row[f"RF_{split}"] = slim(rfperm_permval_single(
                Xe, ye, Xn, yn, n_perm=N_PERM, seed=seed, n_estimators=N_EST, split=split, max_features=mtry,
            ))
    else:
        X = np.vstack([Xe, Xn])
        Y = np.concatenate([ye, yn])
        W = np.concatenate([np.zeros(len(ye), dtype=int), np.ones(len(yn), dtype=int)])
        row["DR"] = slim(drperm_pvalue(X, Y, W, n_perm=N_PERM, seed=seed + 3, n_estimators=N_EST,
                                       null_scheme="residual"))
        row["RR"] = slim(rrperm_pvalue(X, Y, W, n_perm=N_PERM, seed=seed + 11, n_estimators=N_EST))
        row["CF"] = slim(cfperm_loco_pvalue(X, Y, W, test_indices=test_idx, n_perm=N_PERM, seed=seed + 5,
                                            n_estimators=N_EST))
    row["sec"] = time.time() - t0
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tests", choices=("rf", "drc"), required=True)
    p.add_argument("--emb", default=BASE_EMB)
    p.add_argument("--ks", nargs="+", default=["full", "128"])
    p.add_argument("--kinds", nargs="+", default=["without", "with", "spacy"], choices=KINDS)
    p.add_argument("--mtry", type=int, default=None)
    p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    p.add_argument("--b0", type=int, default=0)
    p.add_argument("--b1", type=int, default=50)
    p.add_argument("--n-jobs", type=int, default=4)
    a = p.parse_args()

    load_data(a.emb)
    done = set()
    if LOG.is_file():
        done = {json.loads(line)["key"] for line in LOG.open() if line.strip()}
    jobs = [
        (kind, k, a.mtry, sc, b, a.tests, a.emb)
        for b in range(a.b0, a.b1)
        for k in a.ks
        for kind in a.kinds
        for sc in a.scenarios
        if f"{arm_name(kind, k, a.mtry, a.emb)}|{sc}|{b}|{a.tests}" not in done
    ]
    print(f"[dim_scan] {a.emb} {a.tests} ks={a.ks} kinds={a.kinds} mtry={a.mtry}: "
          f"{len(jobs)} jobs ({len(done)} rows on disk)", flush=True)
    t0 = time.time()
    with LOG.open("a") as fh:
        for i, row in enumerate(
            Parallel(n_jobs=a.n_jobs, backend="loky", return_as="generator_unordered")(
                delayed(run_job)(*j) for j in jobs
            ),
            1,
        ):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if i % 20 == 0 or i == len(jobs):
                print(f"[dim_scan] {i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    print("[dim_scan] done", file=sys.stderr)


if __name__ == "__main__":
    main()
