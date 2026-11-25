"""MSR-VTT: one row per video; caption, waveform, silent video -> benchmark."""

import argparse
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from utils import audio_meta, build_X, text_meta, video_meta, extract_lco_omni_embeddings

def encode_modalities(inputs, batch_size=8):
    return {name: np.asarray(extract_lco_omni_embeddings(values, batch_size=batch_size), dtype=float)
            for name, values in inputs.items()}

def run_msrvtt(video_root, n_pc=30, n_B=150, add_meta_features=True,
               config="train_9k", hf_split="train", n_max=500,
               batch_size=8, positive_category=None, split_col=None, target_value=None):
    from datasets import load_dataset
    from benchmark_all_method import benchmark_all_method

    ds = load_dataset("friedrichor/MSR-VTT", config, split=hf_split)
    files = {p.name: p for p in Path(video_root).rglob("*.mp4")}
    rows, seen = [], set()
    texts, audios, videos, audio_rows, video_rows = [], [], [], [], []
    skipped = 0
    with TemporaryDirectory() as tmp:
        for ex in ds:
            if n_max is not None and len(rows) >= n_max:
                break
            vid = str(ex["video_id"])
            if vid in seen:
                continue
            seen.add(vid)
            path = files.get(Path(ex.get("video") or f"{vid}.mp4").name)
            if path is None or ex.get("category") is None or ex.get("caption") is None:
                skipped += 1
                continue

            wav = Path(tmp) / f"{len(rows)}.wav"
            silent = Path(tmp) / f"{len(rows)}.mp4"
            cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(path)]
            try:
                subprocess.run(cmd + ["-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
                                      "-c:a", "pcm_s16le", str(wav)], check=True, capture_output=True)
                subprocess.run(cmd + ["-map", "0:v:0", "-an", "-c:v", "copy", str(silent)],
                               check=True, capture_output=True)
            except subprocess.CalledProcessError:
                skipped += 1
                continue  # No caption proxy for an absent modality.

            texts.append(str(ex["caption"]))
            audios.append({"audio": str(wav)})
            videos.append({"video": str(silent)})
            rows.append({"video_id": vid, "caption": texts[-1], "category": int(ex["category"])})
            if add_meta_features:
                audio_rows.append(audio_meta(wav))
                video_rows.append(video_meta(silent))

        if not rows:
            raise ValueError("No complete MSR-VTT videos found")
        print(f"MSR-VTT: kept {len(rows)} / {len(seen)} unique attempted videos; skipped {skipped}")
        embeddings = encode_modalities({"text": texts, "audio": audios, "video": videos},
                                       batch_size) if n_pc else {}
        meta = {"text": text_meta(texts), "audio": pd.DataFrame(audio_rows),
                "video": pd.DataFrame(video_rows)} if add_meta_features else {}

    df = pd.DataFrame(rows).set_index("video_id", drop=False)
    y = df["category"].to_numpy()
    if positive_category is not None:
        y = (y == positive_category).astype(int)
    X, indices = build_X(embeddings, meta, n_pc)
    X.index = df.index
    X["Y"] = y
    if split_col:
        target = df[split_col].astype(str) == str(target_value)
        df1, df2 = X.loc[~target], X.loc[target]
    else:
        X = X.sample(frac=1, random_state=0)
        df1, df2 = X.iloc[:len(X)//2], X.iloc[len(X)//2:]
    if df1.empty or df2.empty:
        raise ValueError("Both batches must be nonempty")
    return df1, df2, indices, benchmark_all_method(df1, df2, B=n_B)


def main(argv=None):
    p = argparse.ArgumentParser(description="RUN MSRVTT Dataset")
    p.add_argument("--video-root", required=True)
    p.add_argument("--config", default="train_9k")
    p.add_argument("--hf-split", default="train")
    p.add_argument("--n-pc", type=int, default=30)
    p.add_argument("--n-b", type=int, default=150)
    p.add_argument("--add-meta-features", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--n-max", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--positive-category", type=int)
    p.add_argument("--split-col")
    p.add_argument("--target-value")
    p.add_argument("--out")
    args = p.parse_args(argv)
    if (args.split_col is None) != (args.target_value is None):
        p.error("--split-col and --target-value must be supplied together")
    _, _, _, result = run_msrvtt(args.video_root, args.n_pc, args.n_b, args.add_meta_features,
                                  args.config, args.hf_split, args.n_max, args.batch_size,
                                  args.positive_category, args.split_col, args.target_value)
    print(result.to_string(index=False))
    if args.out:
        result.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
