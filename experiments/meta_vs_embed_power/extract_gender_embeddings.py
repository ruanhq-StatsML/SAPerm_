"""Embed the 1000-row gender frame (``load_gender(1000, 2026)``) with one encoder from ``MODEL_CONFIGS``.

Writes ``results/gender_<model>_embedding.npy`` (row order = the order ``run_dim_scan`` uses).

    python3 -m experiments.meta_vs_embed_power.extract_gender_embeddings qwen3-0.6b
"""

from __future__ import annotations

import hashlib
import sys
import time

import numpy as np

from experiments.meta_vs_embed_power.data import load_gender
from experiments.meta_vs_embed_power.extract import extract_embeddings
from experiments.meta_vs_embed_power.run_dim_scan import OUT, SEED


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen3-0.6b"
    _, y, texts = load_gender(1000, SEED)
    t0 = time.time()
    Z = extract_embeddings(texts, dataset="gender", model=model, out_dir=OUT)
    digest = hashlib.sha256(np.ascontiguousarray(Z, dtype=np.float32).tobytes()).hexdigest()[:16]
    print(f"{model}: {Z.shape} in {time.time() - t0:.0f}s, sha256[:16]={digest}")


if __name__ == "__main__":
    main()
