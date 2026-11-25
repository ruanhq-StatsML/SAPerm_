"""ModaSense: parquet -> independent text/image/audio features -> benchmark."""

import argparse
import io
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from utils import audio_meta, build_X, text_meta, video_meta, extract_lco_omni_embeddings

def encode_modalities(inputs, batch_size=8):
    return {name: np.asarray(extract_lco_omni_embeddings(values, batch_size=batch_size), dtype=float)
            for name, values in inputs.items()}


def run_modasense(path, n_pc=30, n_B=150, add_meta_features=True,
                  n_max=500, batch_size=8, split_col=None, target_value=None):
    from PIL import Image
    from benchmark_all_method import benchmark_all_method

    df = pd.read_parquet(path).iloc[:n_max].reset_index(drop=True)
    if df[["text_propositional", "text_label", "image_perceptual", "audio_perceptual"]].isna().any().any():
        raise ValueError("ModaSense contains missing fields; select complete rows explicitly")
    texts = df["text_propositional"].astype(str).tolist()
    y = df["text_label"].str.lower().isin({"fear", "sad", "angry", "disgust"}).astype(int).to_numpy()

    with TemporaryDirectory() as tmp:
        images, audios, image_rows, audio_rows = [], [], [], []
        for i, (image, audio) in enumerate(zip(df["image_perceptual"], df["audio_perceptual"])):
            im = Image.open(io.BytesIO(media_bytes(image))).convert("RGB")
            img = Path(tmp) / f"{i}.png"
            wav = Path(tmp) / f"{i}.wav"
            im.save(img)
            wav.write_bytes(media_bytes(audio))
            images.append({"image": str(img)})
            audios.append({"audio": str(wav)})
            if add_meta_features:
                image_rows.append(image_meta(np.asarray(im)[:, :, ::-1].copy()))
                audio_rows.append(audio_meta(wav))

        embeddings = encode_modalities({"text": texts, "image": images, "audio": audios},
                                       batch_size) if n_pc else {}
        meta = {"text": text_meta(texts), "image": pd.DataFrame(image_rows),
                "audio": pd.DataFrame(audio_rows)} if add_meta_features else {}

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
    p = argparse.ArgumentParser(description="RUN ModaSense Dataset")
    p.add_argument("--path", default="data/modasense/data/test-00000-of-00001.parquet")
    p.add_argument("--n-pc", type=int, default=30)
    p.add_argument("--n-b", type=int, default=150)
    p.add_argument("--add-meta-features", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--n-max", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--split-col")
    p.add_argument("--target-value")
    p.add_argument("--out")
    args = p.parse_args(argv)
    if (args.split_col is None) != (args.target_value is None):
        p.error("--split-col and --target-value must be supplied together")
    _, _, _, result = run_modasense(args.path, args.n_pc, args.n_b, args.add_meta_features,
                                     args.n_max, args.batch_size, args.split_col, args.target_value)
    print(result.to_string(index=False))
    if args.out:
        result.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
