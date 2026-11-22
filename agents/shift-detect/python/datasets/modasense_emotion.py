#!/usr/bin/env python3
"""ModaSense/emotion — one dataset digestion script.

  read → preprocess → (no embed) → meta(text+image) → schema

  python -m datasets.modasense_emotion --n-max 500
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ._common import OUT, PY_ROOT, Digested

PARQUET = PY_ROOT / "data" / "modasense" / "data" / "test-00000-of-00001.parquet"
_NEG = frozenset({"fear", "sad", "angry", "disgust"})


def load(n_max: int | None = None) -> tuple[list[str], list, np.ndarray]:
    """texts, images, y from parquet."""
    if not PARQUET.exists():
        raise FileNotFoundError(
            f"Missing {PARQUET}. "
            "huggingface-cli download ModaSense/emotion --repo-type dataset "
            "--include 'data/*.parquet' --local-dir data/modasense"
        )
    df = pd.read_parquet(PARQUET)
    if n_max:
        df = df.iloc[:n_max].reset_index(drop=True)
    texts = df["text_propositional"].astype(str).tolist()
    images = df["image_perceptual"].tolist()
    y = np.array(
        [1 if str(l).lower() in _NEG else 0 for l in df["text_label"]], dtype=float
    )
    return texts, images, y


def digest(n_max: int | None = 500) -> Digested:
    from extract_features import FEATURE_NAMES as TEXT_NAMES
    from extract_mm_features import IMAGE_NAMES, build_mm_matrix, schema_mm

    cache = OUT / f"mm_cache_modasense_n{n_max or 'all'}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return Digested(
            dataset="modasense_emotion",
            Z=None,
            M=z["M"],
            y=z["y"],
            texts=list(z["texts"]),
            meta_names=TEXT_NAMES + IMAGE_NAMES,
            schema=schema_mm(),
            extras={"modalities": ["text", "image"]},
        )

    texts, images, y = load(n_max)
    M, names = build_mm_matrix(texts, images)
    assert names == TEXT_NAMES + IMAGE_NAMES
    d = Digested(
        dataset="modasense_emotion",
        Z=None,
        M=M,
        y=y,
        texts=texts,
        meta_names=names,
        schema=schema_mm(),
        extras={"modalities": ["text", "image"]},
    )
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache, M=M, y=y, texts=np.array(texts, dtype=object),
        names=np.array(names, dtype=object),
    )
    return d


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-max", type=int, default=500)
    args = p.parse_args(argv)
    d = digest(args.n_max)
    print(f"[modasense] n={d.n} M={d.M.shape} layout={d.schema['layout']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
