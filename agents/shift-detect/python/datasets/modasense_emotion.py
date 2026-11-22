#!/usr/bin/env python3
"""Dataset digestion: ModaSense/emotion (text + image).

Pipeline
--------
  1. read        : HF parquet ModaSense/emotion
  2. preprocess  : pull text_propositional + image_perceptual bytes; y from text_label
  3. embed       : none (meta-only multimodal); optional later
  4. meta        : spaCy+VADER text + OpenCV image (extract_mm_features)
  5. schema      : text_meta | image_meta indices

Usage
-----
  python -m datasets.modasense_emotion --n-max 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from extract_features import FEATURE_NAMES as TEXT_NAMES  # noqa: E402
from extract_mm_features import IMAGE_NAMES, build_mm_matrix, schema_mm  # noqa: E402

try:
    from ._common import Digested  # noqa: E402
except ImportError:
    from datasets._common import Digested  # noqa: E402

PARQUET = _ROOT / "data" / "modasense" / "data" / "test-00000-of-00001.parquet"
OUT = _ROOT / "results"


def read_raw(path: Path = PARQUET) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download: huggingface-cli download ModaSense/emotion "
            "--repo-type dataset --include 'data/*.parquet' --local-dir data/modasense"
        )
    return pd.read_parquet(path)


def preprocess(df: pd.DataFrame, n_max: int | None) -> pd.DataFrame:
    if n_max:
        df = df.iloc[:n_max].reset_index(drop=True)
    return df


def digest(n_max: int | None = 500) -> Digested:
    df = preprocess(read_raw(), n_max)
    cache = OUT / f"mm_cache_modasense_n{len(df)}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        M, y = z["M"], z["y"]
        texts = list(z["texts"]) if "texts" in z.files else df["text_propositional"].astype(str).tolist()
    else:
        texts = df["text_propositional"].astype(str).tolist()
        images = df["image_perceptual"].tolist()
        M, names = build_mm_matrix(texts, images)
        assert names == TEXT_NAMES + IMAGE_NAMES
        neg = {"fear", "sad", "angry", "disgust"}
        y = np.array([1 if str(l).lower() in neg else 0 for l in df["text_label"]], dtype=float)
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
        meta_names=TEXT_NAMES + IMAGE_NAMES,
        schema=schema_mm(),
        extras={"modalities": ["text", "image"]},
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-max", type=int, default=500)
    args = p.parse_args(argv)
    d = digest(args.n_max)
    print(f"[modasense] n={d.n} M={d.M.shape} schema={d.schema['layout']}")
    print(f"  blocks: {list(d.schema['blocks'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
