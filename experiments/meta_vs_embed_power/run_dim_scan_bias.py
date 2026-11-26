"""bias core-arm RFPerm scan (RF only): Z_k for k in {24, full} x core arms, MiniLM embedding.

Port of the RF part of ``run_dim_scan.py`` (gender branch) to the ``bias`` dataset (``train_bias2.csv``,
length-quartile pseudo-label). Same seeds, window construction, arm definitions and jsonl key convention:

    key = "{arm}|{scenario}|{b}|rf",  rep_seed = 2026 + 10007 * (crc32(scenario) % 997) + 17 * b

so at k=24 ``without``/``with`` reproduce ``run_power`` (RFPerm upper tail) exactly, and every arm shares
the same window pair for a given (scenario, b). Each row carries RF_random and RF_ordered (upper-tail p).

Arms (``kind``): ``without`` = Z_k, ``with`` = [Z_k, M], ``spacy`` = [Z_k, S], ``vader`` = [Z_k, VADER4]
(vader_neg, vader_pos, vader_compound, polar_gap; label ``with_vader``), ``with_decorr`` = [Z_k, M_kept]
where M_kept drops columns with |spearman(column, y)| >= 0.3 on the 1000-row bias frame.

    python3 -m experiments.meta_vs_embed_power.run_dim_scan_bias --ks 24 full --b1 150
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

from experiments.meta_vs_embed_power.data import load_train_bias_finder_tab, pca_embed
from experiments.meta_vs_embed_power.power import make_windows
from experiments.meta_vs_embed_power.rfperm_official import rfperm_permval_single

OUT = Path(__file__).resolve().parent / "results"
LOG = OUT / "dim_scan_bias.jsonl"
SCENARIOS = ("sentiment_meta", "syntax_meta", "train_vs_heldout")
VADER = ("vader_neg", "vader_pos", "vader_compound", "polar_gap")
KINDS = ("without", "with", "spacy", "vader", "with_decorr")
LABEL = {"without": "without_meta", "with": "with_meta", "spacy": "with_spacy",
         "vader": "with_vader", "with_decorr": "with_meta_decorr"}
SEED, N_DOC, N_EST, N_PERM, TAU = 2026, 120, 150, 150, 0.3
EMB = "MiniLM_L6"

_BASE: dict = {}
_Z: dict = {}


def load_base() -> dict:
    if _BASE:
        return _BASE
    _, y, texts = load_train_bias_finder_tab(1000, SEED)
    z = np.load(OUT / "meta_bias_n1000.npz", allow_pickle=True)
    M, names = np.asarray(z["M"], dtype=float), [str(n) for n in z["meta_names"]]
    sp_cache = OUT / "spacy_bias_n1000.npz"
    if sp_cache.is_file():
        zs = np.load(sp_cache, allow_pickle=True)
        S, s_names = zs["S"], [str(n) for n in zs["spacy_names"]]
    else:
        from experiments.meta_vs_embed_power.spacy_features import spacy_matrix

        S, s_names = spacy_matrix(texts)
        np.savez_compressed(sp_cache, S=S, spacy_names=np.array(s_names, dtype=object))
    S = np.asarray(S, dtype=float)
    assert len(y) == len(M) == len(S) == 1000, (len(y), M.shape, S.shape)
    rho = [0.0 if c.std() == 0 else abs(spearmanr(c, y).correlation) for c in M.T]
    keep = np.array([j for j, r in enumerate(rho) if r < TAU], dtype=int)
    _BASE.update(M=M, S=S, MS=np.hstack([M, S]), names=names, s_names=list(s_names), y=y,
                 M_keep=keep, rho=dict(zip(names, rho)))
    return _BASE


def z_k(k: str) -> np.ndarray:
    d = load_base()
    if "full" not in _Z:
        Z = np.load(OUT / f"bias_{EMB}_embedding.npy")
        assert len(Z) == len(d["y"]), Z.shape
        _Z["full"] = Z
    if k not in _Z:
        _Z[k] = pca_embed(_Z["full"], int(k), seed=SEED)
    return _Z[k]


def rep_seed(scenario: str, b: int) -> int:
    return SEED + 10_007 * (zlib.crc32(scenario.encode()) % 997) + 17 * b


def build(kind: str, k: str, scenario: str, b: int):
    d = load_base()
    Zk = z_k(k)
    seed = rep_seed(scenario, b)
    rng = np.random.default_rng(seed)
    Ze, ye, Zn, yn, _, _, MSe, MSn = make_windows(scenario, Zk, d["MS"], d["y"], d["names"], N_DOC, rng)
    n_meta = d["M"].shape[1]
    if kind == "without":
        return Ze, ye, Zn, yn, seed
    if kind == "with":
        idx = np.arange(n_meta)
    elif kind == "spacy":
        idx = n_meta + np.arange(d["S"].shape[1])
    elif kind == "vader":
        idx = np.array([d["names"].index(c) for c in VADER], dtype=int)
    else:
        idx = d["M_keep"]
    return np.hstack([Ze, MSe[:, idx]]), ye, np.hstack([Zn, MSn[:, idx]]), yn, seed


def run_job(kind: str, k: str, scenario: str, b: int) -> dict:
    t0 = time.time()
    Xe, ye, Xn, yn, seed = build(kind, k, scenario, b)
    keep = ("p_value", "p_value_upper", "p_value_lower", "p_value_abs", "statistic", "reject")
    slim = lambda r: {key: (bool(v) if key == "reject" else float(v)) for key, v in r.items() if key in keep}
    row = {"key": f"{LABEL[kind]}_k{k}|{scenario}|{b}|rf", "arm": f"{LABEL[kind]}_k{k}", "emb": EMB,
           "kind": kind, "k": k, "dim": int(Xe.shape[1]), "scenario": scenario, "b": b, "seed": seed,
           "tests": "rf"}
    for split in ("random", "ordered"):
        row[f"RF_{split}"] = slim(rfperm_permval_single(
            Xe, ye, Xn, yn, n_perm=N_PERM, seed=seed, n_estimators=N_EST, split=split))
    row["sec"] = time.time() - t0
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ks", nargs="+", default=["24", "full"])
    p.add_argument("--kinds", nargs="+", default=list(KINDS), choices=KINDS)
    p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    p.add_argument("--b0", type=int, default=0)
    p.add_argument("--b1", type=int, default=150)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--log", type=Path, default=LOG)
    a = p.parse_args()
    d = load_base()
    dropped = [n for n, r in d["rho"].items() if r >= TAU]
    print(f"[dim_scan_bias] decorr drops {len(dropped)}/{len(d['names'])} M columns: {dropped}", flush=True)
    done = set()
    if a.log.is_file():
        done = {json.loads(line)["key"] for line in a.log.open() if line.strip()}
    jobs = [
        (kind, k, sc, b)
        for b in range(a.b0, a.b1) for k in a.ks for kind in a.kinds for sc in a.scenarios
        if f"{LABEL[kind]}_k{k}|{sc}|{b}|rf" not in done
    ]
    print(f"[dim_scan_bias] {len(jobs)} jobs ({len(done)} rows on disk)", flush=True)
    t0 = time.time()
    with a.log.open("a") as fh:
        for i, row in enumerate(
            Parallel(n_jobs=a.n_jobs, backend="loky", return_as="generator_unordered")(
                delayed(run_job)(*j) for j in jobs), 1):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if i % 50 == 0 or i == len(jobs):
                print(f"[dim_scan_bias] {i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    print("[dim_scan_bias] done", file=sys.stderr)


if __name__ == "__main__":
    main()
