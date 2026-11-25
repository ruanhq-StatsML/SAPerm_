#!/usr/bin/env python3
"""BiasFinder BERT + GEN-sarc — one dataset digestion script.

  read → preprocess → embed(PCA) → meta(spaCy+VADER) → schema

  python -m datasets.biasfinder_gen --n-max 2000 --write results/digest_biasfinder.npz
"""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path

import numpy as np

from ._common import (
    OUT,
    REPO,
    Digested,
    cached_text_meta,
    pca_embed,
    stratified_cap,
)

EMB = REPO / "datasets" / "train_biasfinder_bert_embedding.csv"
ZIP = REPO / "datasets" / "dataset.zip"


def load(n_max: int | None = None, seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Raw Z (n,768), y, texts — shared by digestion and power_*.py."""
    texts, classes_txt = [], []
    with zipfile.ZipFile(ZIP) as zf:
        with zf.open("GEN-sarc-notsarc.csv") as raw:
            for row in csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8")):
                texts.append(row["text"])
                classes_txt.append(row["class"])

    Z_list, classes_emb = [], []
    with EMB.open(encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        start = 1 if header[0] in ("", "Unnamed: 0") or header[0].lower() == "unnamed: 0" else 0
        for row in reader:
            classes_emb.append(row[-1])
            feats = row[start:-1]
            if len(feats) == 769:
                feats = feats[1:]
            Z_list.append([float(v) for v in feats])

    if classes_txt != classes_emb:
        raise RuntimeError("GEN text not aligned with BiasFinder embedding rows")

    Z = np.asarray(Z_list, dtype=float)
    y = np.asarray([1 if c == "sarc" else 0 for c in classes_emb], dtype=float)
    take = stratified_cap(y, n_max, seed=seed)
    return Z[take], y[take], [texts[i] for i in take]


def digest(n_max: int | None = 2000, pca_k: int = 24, seed: int = 0) -> Digested:
    from datasets.feature_schema import schema

    Z, y, texts = load(n_max, seed=seed)
    M, names = cached_text_meta(texts, OUT / f"meta_cache_n{len(texts)}.npz")
    Zpca = pca_embed(Z, pca_k, seed=seed)
    return Digested(
        dataset="biasfinder_gen",
        Z=Zpca,
        M=M,
        y=y,
        texts=texts,
        meta_names=names,
        schema=schema(pca_k),
        extras={"p_embed_raw": int(Z.shape[1]), "pca_k": pca_k},
    )


SCENARIOS = ("null_shift", "sentiment_meta", "syntax_meta", "train_vs_heldout")


def make_windows(
    scenario: str,
    d: Digested,
    n_doc: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    from datasets.text_meta import FEATURE_NAMES

    M, y = d.M, d.y
    n = len(y)
    if scenario == "null_shift":
        idx = rng.permutation(n)
        return idx[:n_doc], idx[n_doc : 2 * n_doc]
    if scenario == "sentiment_meta":
        score = M[:, FEATURE_NAMES.index("vader_compound")]
        order = np.argsort(score)
        neg = order[: max(n_doc * 3, n_doc)]
        mid = order[len(order) // 3 : 2 * len(order) // 3]
        return (
            rng.choice(mid, n_doc, replace=len(mid) < n_doc),
            rng.choice(neg, n_doc, replace=len(neg) < n_doc),
        )
    if scenario == "syntax_meta":
        score = M[:, FEATURE_NAMES.index("verb_noun_ratio")]
        order = np.argsort(score)
        lo, hi = order[: max(n_doc * 3, n_doc)], order[-max(n_doc * 3, n_doc) :]
        return (
            rng.choice(lo, n_doc, replace=len(lo) < n_doc),
            rng.choice(hi, n_doc, replace=len(hi) < n_doc),
        )
    if scenario == "train_vs_heldout":
        half = n // 2
        a = rng.choice(np.arange(half), n_doc, replace=half < n_doc)
        b = rng.choice(np.arange(half, n), n_doc, replace=(n - half) < n_doc)
        return a, b
    raise ValueError(scenario)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-max", type=int, default=2000)
    p.add_argument("--pca-k", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--write", type=Path, default=None)
    args = p.parse_args(argv)
    d = digest(args.n_max, args.pca_k, args.seed)
    print(f"[biasfinder_gen] n={d.n} Z={d.Z.shape} M={d.M.shape} p={d.schema['p_total']}")
    if args.write:
        d.save(args.write)
        print(f"  wrote {args.write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
