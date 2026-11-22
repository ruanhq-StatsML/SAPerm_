#!/usr/bin/env python3
"""ModaSense/emotion digestion: read → meta(text+image[+audio]) → schema.

Pipeline (meta-only multimodal; no embedding PCA):

  parquet → text/image[/audio] meta → X=[M_text, M_image, M_audio] → schema

  python -m datasets.modasense_emotion --n-max 500 --audio
  python -m datasets.modasense_emotion --n-max 500 --audio --write
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    from datasets.text_meta import FEATURE_NAMES as TEXT_NAMES
    from datasets.mm_meta import AUDIO_NAMES, IMAGE_NAMES, build_mm_matrix, schema_mm

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
        extras={
            "modalities": ["text", "image"] + (["audio"] if with_audio else []),
            "cache": str(cache),
            "label": "text_label → binary (neg={fear,sad,angry,disgust})",
            "source": "ModaSense/emotion parquet",
        },
    )


def write_digest_report(d: Digested, path: Path | None = None) -> Path:
    """Write a small JSON/MD summary of the digested matrix + schema."""
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "mm3" if "audio" in d.extras.get("modalities", []) else "mm"
    stem = path or (OUT / f"digest_modasense_{tag}_n{d.n}")
    stem = Path(stem)
    if stem.suffix:
        stem = stem.with_suffix("")

    modality = d.schema.get("modality", {})
    blocks = d.schema.get("blocks", {})
    summary = {
        "dataset": d.dataset,
        "n": d.n,
        "p": int(d.M.shape[1]),
        "layout": d.schema.get("layout"),
        "modalities": d.extras.get("modalities"),
        "y_rate": float(np.mean(d.y)),
        "modality_dims": {m: len(ix) for m, ix in modality.items()},
        "blocks": {
            name: {
                "package": meta.get("package"),
                "n_feat": len(meta.get("indices", [])),
                "indices": meta.get("indices"),
            }
            for name, meta in blocks.items()
        },
        "meta_names": list(d.meta_names),
        "extras": {k: v for k, v in d.extras.items() if k != "modalities"},
    }
    stem.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# ModaSense digestion",
        "",
        f"- dataset: `{d.dataset}`",
        f"- n={d.n}, p={d.M.shape[1]}, y_rate={summary['y_rate']:.3f}",
        f"- layout: `{summary['layout']}`",
        f"- modalities: {', '.join(summary['modalities'] or [])}",
        "",
        "## Modality dims",
        "",
        "| modality | dim |",
        "| --- | ---: |",
    ]
    for m, dim in summary["modality_dims"].items():
        lines.append(f"| {m} | {dim} |")
    lines += ["", "## Blocks", ""]
    for name, meta in summary["blocks"].items():
        lines.append(f"- `{name}` ({meta['package']}): {meta['n_feat']} feats")
    lines += [
        "",
        "## Downstream",
        "",
        "```bash",
        "python power_mm_modasense.py --B 80",
        "python case_modasense_business.py",
        "python run_mmd_two_datasets.py",
        "```",
        "",
        "Scientific read: `results/modasense_scientific_case.md`.",
    ]
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return stem.with_suffix(".md")


SCENARIOS = (
    "null_shift",
    "text_sentiment_shift",
    "image_brightness_shift",
    "audio_energy_shift",
)


def make_windows(
    scenario: str,
    d: Digested,
    n_doc: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Dataset-specific batch indices (exist, new)."""
    from datasets.text_meta import FEATURE_NAMES as TEXT_NAMES
    from datasets.mm_meta import AUDIO_NAMES, IMAGE_NAMES

    M, y = d.M, d.y
    n = len(y)
    if scenario == "null_shift":
        idx = rng.permutation(n)
        return idx[:n_doc], idx[n_doc : 2 * n_doc]

    if scenario == "text_sentiment_shift":
        score = M[:, TEXT_NAMES.index("vader_compound")]
    elif scenario == "image_brightness_shift":
        score = M[:, len(TEXT_NAMES) + IMAGE_NAMES.index("img_brightness")]
    elif scenario == "audio_energy_shift":
        if "audio" not in d.schema.get("modality", {}):
            raise ValueError("audio_energy_shift needs digest(..., with_audio=True)")
        off = len(TEXT_NAMES) + len(IMAGE_NAMES)
        score = M[:, off + AUDIO_NAMES.index("aud_rms_energy")]
    else:
        raise ValueError(scenario)
    order = np.argsort(score)
    lo, hi = order[: max(n_doc * 2, n_doc)], order[-max(n_doc * 2, n_doc) :]
    a = rng.choice(lo, n_doc, replace=len(lo) < n_doc)
    b = rng.choice(hi, n_doc, replace=len(hi) < n_doc)
    return a, b


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-max", type=int, default=500)
    p.add_argument("--audio", action="store_true", help="include audio meta (X 3-modal)")
    p.add_argument("--write", action="store_true", help="write digest_modasense_*.{json,md}")
    args = p.parse_args(argv)
    d = digest(args.n_max, with_audio=args.audio)
    print(f"[modasense] n={d.n} M={d.M.shape} {d.schema['layout']}")
    if args.write:
        path = write_digest_report(d)
        print(f"[modasense] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
