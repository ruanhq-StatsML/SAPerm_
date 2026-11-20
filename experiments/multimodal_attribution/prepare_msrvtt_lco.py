"""MSR-VTT → LCO-Embedding-Omni-3B per modality (no CLIP/WavLM frame mean-pool).

Replaces the old per-video mean of CLIP frames / WavLM windows. Each modality is
encoded **once** by Omni in a shared space, then cached for PO-risk block leave-out:

  text  → caption string
  audio → soundtrack path (ffmpeg → wav) or ``[audio]`` proxy if missing
  video → mp4 path (Omni video tower; not frame-mean CLIP)

```bash
python -m experiments.multimodal_attribution.prepare_msrvtt_lco \\
  --max-rows 700 --batch-size 4 \\
  --out-dir experiments/multimodal_attribution/results/msrvtt_lco_cache
```
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.multimodal_attribution.embeddings import (  # noqa: E402
    LCO_OMNI_MODEL_ID,
    extract_lco_omni_embeddings,
)
from experiments.multimodal_attribution.msrvtt import load_msrvtt_captions  # noqa: E402

DEFAULT_OUT = Path("experiments/multimodal_attribution/results/msrvtt_lco_cache")


def _resolve_video(video_root: Path, video_file: str) -> Optional[Path]:
    for c in (
        video_root / video_file,
        video_root / "TrainValVideo" / video_file,
        video_root / "TestVideo" / video_file,
        video_root / "Videos" / video_file,
    ):
        if c.is_file():
            return c
    return None


def _extract_wav(video_path: Path, wav_path: Path, sr: int = 16000) -> Optional[Path]:
    if wav_path.is_file() and wav_path.stat().st_size > 0:
        return wav_path
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(sr),
        str(wav_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
    except FileNotFoundError:
        return None
    if proc.returncode != 0 or not wav_path.is_file():
        return None
    return wav_path


def _cache_path(out_dir: Path, split: str, modality: str) -> Path:
    return out_dir / f"msrvtt_{split}_{modality}_LCO_Omni_3B.npz"


def _load_or_encode(
    inputs: Sequence,
    *,
    path: Path,
    batch_size: int,
    force: bool,
) -> np.ndarray:
    if path.exists() and not force:
        emb = np.asarray(np.load(path)["emb"], dtype=float)
        if emb.shape[0] == len(inputs):
            print(f"[cache hit] {path.name} shape={emb.shape}", flush=True)
            return emb
    emb = extract_lco_omni_embeddings(inputs, batch_size=batch_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, emb=np.asarray(emb, dtype=np.float32))
    print(f"[cache save] {path.name} shape={emb.shape}", flush=True)
    return np.asarray(emb, dtype=float)


def prepare_split(
    *,
    split: str,
    config: str,
    max_rows: int,
    video_root: Path,
    out_dir: Path,
    batch_size: int,
    force: bool,
) -> Dict[str, np.ndarray]:
    hf_split = "train" if split.startswith("train") else "test"
    ds = load_msrvtt_captions(config=config, split=hf_split, max_rows=None)
    # Unique video_ids (multi-caption collapse).
    rows: List[dict] = []
    seen = set()
    for ex in ds:
        vid = ex["video_id"]
        if vid in seen:
            continue
        seen.add(vid)
        rows.append(
            {
                "video_id": vid,
                "video_file": ex.get("video") or f"{vid}.mp4",
                "caption": ex["caption"],
                "category": int(ex.get("category", -1)),
            }
        )
        if len(rows) >= max_rows:
            break

    captions = [str(r["caption"]) for r in rows]
    text_emb = _load_or_encode(
        captions,
        path=_cache_path(out_dir, split, "text"),
        batch_size=batch_size,
        force=force,
    )

    audio_inputs = []
    video_inputs = []
    wav_dir = out_dir / "wav"
    meta = []
    for r in rows:
        vpath = _resolve_video(video_root, r["video_file"])
        wav = None
        if vpath is not None:
            wav = _extract_wav(vpath, wav_dir / f"{r['video_id']}.wav")
        if wav is not None:
            audio_inputs.append({"audio": str(wav)})
        else:
            audio_inputs.append(f"[audio] Soundtrack for: {r['caption']}")
        if vpath is not None:
            video_inputs.append({"video": str(vpath), "text": r["caption"]})
        else:
            video_inputs.append(f"[video] Scene: {r['caption']}")
        meta.append({**r, "video_path": str(vpath) if vpath else "", "wav": str(wav) if wav else ""})

    audio_emb = _load_or_encode(
        audio_inputs,
        path=_cache_path(out_dir, split, "audio"),
        batch_size=max(1, batch_size // 2),
        force=force,
    )
    video_emb = _load_or_encode(
        video_inputs,
        path=_cache_path(out_dir, split, "video"),
        batch_size=max(1, batch_size // 2),
        force=force,
    )

    pack_path = out_dir / f"msrvtt_{split}_LCO_Omni_3B.npz"
    y = np.asarray([r["category"] for r in rows], dtype=float)
    np.savez(
        pack_path,
        text=text_emb,
        audio=audio_emb,
        video=video_emb,
        y=y,
        video_id=np.asarray([r["video_id"] for r in rows], dtype=object),
    )
    (out_dir / f"msrvtt_{split}_meta.jsonl").write_text(
        "\n".join(json.dumps(m) for m in meta) + "\n"
    )
    print(f"[pack] {pack_path} n={len(rows)} dim={text_emb.shape[1]}", flush=True)
    return {"text": text_emb, "audio": audio_emb, "video": video_emb, "y": y}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-rows", type=int, default=700)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--force", action="store_true")
    p.add_argument("--video-root", type=Path, default=Path("experiments/multimodal_attribution/data/MSRVTT"))
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--train-config", default="train_9k")
    p.add_argument("--test-config", default="test_1k")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[prepare] MSR-VTT LCO Omni ({LCO_OMNI_MODEL_ID}) max_rows={args.max_rows}", flush=True)
    prepare_split(
        split="train",
        config=args.train_config,
        max_rows=args.max_rows,
        video_root=args.video_root,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        force=args.force,
    )
    prepare_split(
        split="test",
        config=args.test_config,
        max_rows=args.max_rows,
        video_root=args.video_root,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        force=args.force,
    )
    print(f"[prepare] done in {(time.time()-t0)/3600:.2f}h → {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
