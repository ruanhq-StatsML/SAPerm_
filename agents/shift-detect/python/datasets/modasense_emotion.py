#!/usr/bin/env python3
"""ModaSense/emotion digestion: read → meta(text+image[+audio]) → schema.

  python -m datasets.modasense_emotion --n-max 500
  python -m datasets.modasense_emotion --n-max 500 --audio
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ._common import OUT, PY_ROOT, Digested

PARQUET = PY_ROOT / "data" / "modasense" / "data" / "test-00000-of-00001.parquet"
_NEG = frozenset({"fear", "sad", "angry", "disgust"})


def load_df(n_max: int | None = None) -> pd.DataFrame:
    if not PARQUET.exists():
        raise FileNotFoundError(
            f"Missing {PARQUET}. huggingface-cli download ModaSense/emotion "
            "--repo-type dataset --include 'data/*.parquet' --local-dir data/modasense"
        )
    df = pd.read_parquet(PARQUET)
    return df.iloc[:n_max].reset_index(drop=True) if n_max else df


def digest(n_max: int | None = 500, *, with_audio: bool = False) -> Digested:
    from extract_features import FEATURE_NAMES as TEXT_NAMES
    from extract_mm_features import AUDIO_NAMES, IMAGE_NAMES, build_mm_matrix, schema_mm

    df = load_df(n_max)
    tag = "mm3" if with_audio else "mm"
    cache = OUT / f"{tag}_cache_modasense_n{len(df)}.npz"
    names = list(TEXT_NAMES) + IMAGE_NAMES + (AUDIO_NAMES if with_audio else [])
    texts = df["text_propositional"].astype(str).tolist()
    y = np.array([1 if str(l).lower() in _NEG else 0 for l in df["text_label"]], dtype=float)

    if cache.exists():
        M = np.load(cache, allow_pickle=True)["M"]
    else:
        M, got = build_mm_matrix(
            texts,
            df["image_perceptual"].tolist(),
            df["audio_perceptual"].tolist() if with_audio else None,
        )
        assert list(got) == names
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache, M=M, y=y, texts=np.array(texts, dtype=object),
            names=np.array(names, dtype=object),
        )

    return Digested(
        dataset="modasense_emotion",
        Z=None,
        M=M,
        y=y,
        texts=texts,
        meta_names=names,
        schema=schema_mm(with_audio=with_audio),
        extras={"modalities": ["text", "image"] + (["audio"] if with_audio else [])},
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-max", type=int, default=500)
    p.add_argument("--audio", action="store_true")
    args = p.parse_args(argv)
    d = digest(args.n_max, with_audio=args.audio)
    print(f"[modasense] n={d.n} M={d.M.shape} {d.schema['layout']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
