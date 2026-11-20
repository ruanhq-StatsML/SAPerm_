"""Precompute LCO-Embedding-Omni-3B caches for CMU-MOSEI and MELD.

```bash
python -m experiments.multimodal_attribution.prepare_emotion_embeddings \\
  --model-key LCO_Omni_3B --max-rows 700
```
"""

from __future__ import annotations

import argparse
import time

from experiments.multimodal_attribution.emotion_datasets import (
    DEFAULT_MODEL,
    prepare_meld_embeddings,
    prepare_mosei_embeddings,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-key", default=DEFAULT_MODEL)
    p.add_argument("--max-rows", type=int, default=700)
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="LCO Omni encode batch size (CPU: 8 is usually ok; drop to 4 if OOM)",
    )
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    t0 = time.time()
    print(f"[prepare] MOSEI model={args.model_key} max_rows={args.max_rows}", flush=True)
    mosei = prepare_mosei_embeddings(
        model_key=args.model_key,
        max_rows=args.max_rows,
        force=args.force,
        batch_size=args.batch_size,
    )
    print(
        f"[prepare] MOSEI done ref={len(mosei['ref']['y'])} new={len(mosei['new']['y'])} "
        f"dim={mosei['ref']['text'].shape[1]} elapsed={time.time()-t0:.0f}s",
        flush=True,
    )
    t1 = time.time()
    print(f"[prepare] MELD model={args.model_key} max_rows={args.max_rows}", flush=True)
    meld = prepare_meld_embeddings(
        model_key=args.model_key,
        max_rows=args.max_rows,
        force=args.force,
        batch_size=args.batch_size,
    )
    print(
        f"[prepare] MELD done ref={len(meld['ref']['y'])} new={len(meld['new']['y'])} "
        f"dim={meld['ref']['text'].shape[1]} elapsed={time.time()-t1:.0f}s",
        flush=True,
    )
    print(f"[prepare] all done in {(time.time()-t0)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
