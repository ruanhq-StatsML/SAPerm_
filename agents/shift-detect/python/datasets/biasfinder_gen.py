#!/usr/bin/env python3
"""Dataset digestion: BiasFinder BERT + GEN-sarc text.

Pipeline (one script, one dataset)
----------------------------------
  1. read        : datasets/train_biasfinder_bert_embedding.csv
                   + datasets/dataset.zip::GEN-sarc-notsarc.csv
  2. preprocess  : drop index col, align text↔embed 1:1, stratify cap
  3. embed       : Z already provided (768-d BiasFinder BERT)
  4. meta        : spaCy + VADER via extract_features
  5. schema      : feature_schema list-indices for X=[Z_PCA, M]

Usage
-----
  python -m datasets.biasfinder_gen --n-max 2000 --write results/digest_biasfinder.npz
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# allow `python datasets/biasfinder_gen.py` and `python -m datasets.biasfinder_gen`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from extract_features import FEATURE_NAMES, build_matrix  # noqa: E402
from feature_schema import schema  # noqa: E402

try:
    from ._common import Digested  # noqa: E402
except ImportError:
    from datasets._common import Digested  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
EMB = REPO / "datasets" / "train_biasfinder_bert_embedding.csv"
ZIP = REPO / "datasets" / "dataset.zip"
OUT = _ROOT / "results"


def read_raw() -> tuple[list[str], list[str], list[list[float]], list[str]]:
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
    return texts, classes_txt, Z_list, classes_emb


def preprocess(
    texts: list[str],
    classes_txt: list[str],
    Z_list: list[list[float]],
    classes_emb: list[str],
    n_max: int | None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if classes_txt != classes_emb:
        raise RuntimeError("GEN text not aligned with BiasFinder embedding rows")
    Z = np.asarray(Z_list, dtype=float)
    y = np.asarray([1 if c == "sarc" else 0 for c in classes_emb], dtype=float)
    if n_max is not None and n_max < len(y):
        rng = np.random.default_rng(seed)
        i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
        k0, k1 = n_max // 2, n_max - n_max // 2
        take = np.concatenate(
            [rng.choice(i0, min(k0, len(i0)), False), rng.choice(i1, min(k1, len(i1)), False)]
        )
        take.sort()
        Z, y = Z[take], y[take]
        texts = [texts[i] for i in take]
    return Z, y, texts


def extract_meta(texts: list[str], cache: Path | None) -> tuple[np.ndarray, list[str]]:
    if cache is not None and cache.exists():
        data = np.load(cache, allow_pickle=True)
        names = list(data["names"])
        if names == FEATURE_NAMES:
            return data["M"], names
    M_list, names = build_matrix(texts, "meta")
    M = np.asarray(M_list, dtype=float)
    # reorder to FEATURE_NAMES
    j = {n: i for i, n in enumerate(names)}
    M = M[:, [j[n] for n in FEATURE_NAMES]]
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, M=M, names=np.array(FEATURE_NAMES, dtype=object))
    return M, list(FEATURE_NAMES)


def digest(n_max: int | None = 2000, pca_k: int = 24, seed: int = 0) -> Digested:
    texts, ct, Z_list, ce = read_raw()
    Z, y, texts = preprocess(texts, ct, Z_list, ce, n_max, seed=seed)
    cache = OUT / f"meta_cache_n{len(texts)}.npz"
    M, names = extract_meta(texts, cache)
    Zpca = PCA(pca_k, random_state=seed).fit_transform(StandardScaler().fit_transform(Z))
    return Digested(
        dataset="biasfinder_gen",
        Z=Zpca,  # store PCA'd embed for downstream tests
        M=M,
        y=y,
        texts=texts,
        meta_names=names,
        schema=schema(pca_k),
        extras={"p_embed_raw": int(Z.shape[1]), "pca_k": pca_k},
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-max", type=int, default=2000)
    p.add_argument("--pca-k", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--write", type=Path, default=None)
    args = p.parse_args(argv)
    d = digest(args.n_max, args.pca_k, args.seed)
    print(
        f"[biasfinder_gen] n={d.n} Z={None if d.Z is None else d.Z.shape} "
        f"M={d.M.shape} schema_p={d.schema['p_total']}"
    )
    print(f"  meta blocks: {list(d.schema['meta_block_indices'])}")
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.write,
            Z=d.Z, M=d.M, y=d.y,
            meta_names=np.array(d.meta_names, dtype=object),
            texts=np.array(d.texts, dtype=object),
        )
        print(f"  wrote {args.write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
