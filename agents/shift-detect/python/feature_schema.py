#!/usr/bin/env python3
"""Index registry for X = [Z_embed, M_meta]: modality + named blocks.

Used after digestion / localization / interaction so CFPerm LOCO and
reporting always know which columns are embeddings vs which meta block.
"""

from __future__ import annotations

from dataclasses import dataclass

from extract_features import BLOCKS, FEATURE_NAMES


@dataclass(frozen=True)
class BlockSlice:
    name: str
    modality: str  # "embed" | "meta"
    package: str
    start: int
    end: int  # exclusive
    names: tuple[str, ...]

    @property
    def indices(self) -> list[int]:
        return list(range(self.start, self.end))


# Text meta packages (portable, encapsulated).
META_PACKAGES = {
    "spacy_style": "spacy:en_core_web_sm",
    "spacy_syntax": "spacy:en_core_web_sm",
    "spacy_ner": "spacy:en_core_web_sm",
    "vader_sent": "vaderSentiment",
    "discourse": "surface+regex",
    "structure_cues": "spacy+regex",
}

# Optional video meta packages (out of scope for this run; listed for portability).
VIDEO_META_PACKAGES = {
    "motion": ["opencv-python", "scenedetect"],
    "scene_cuts": ["scenedetect", "ffmpeg-python"],
    "audio": ["librosa", "soundfile"],
    "asr_subtitle": ["whisper", "faster-whisper"],
    "objects": ["ultralytics", "torchvision"],
}


def schema(pca_k: int) -> dict:
    """Build list-indices for X=[Z_PCA(0:pca_k), M(pca_k:pca_k+p_meta)]."""
    blocks: list[BlockSlice] = [
        BlockSlice(
            name="embed_pca",
            modality="embed",
            package="transformers/BiasFinder-BERT+sklearn.PCA",
            start=0,
            end=pca_k,
            names=tuple(f"pca_{i}" for i in range(pca_k)),
        )
    ]
    offset = pca_k
    for bname, feats in BLOCKS.items():
        n = len(feats)
        blocks.append(
            BlockSlice(
                name=bname,
                modality="meta",
                package=META_PACKAGES.get(bname, "meta"),
                start=offset,
                end=offset + n,
                names=tuple(feats),
            )
        )
        offset += n

    meta_start = pca_k
    meta_end = pca_k + len(FEATURE_NAMES)
    return {
        "layout": "X=[Z_PCA, M_meta]",
        "pca_k": pca_k,
        "p_meta": len(FEATURE_NAMES),
        "p_total": meta_end,
        "modality": {
            "embed": list(range(0, pca_k)),
            "meta": list(range(meta_start, meta_end)),
        },
        "meta_block_indices": {
            b.name: b.indices for b in blocks if b.modality == "meta"
        },
        "blocks": [
            {
                "name": b.name,
                "modality": b.modality,
                "package": b.package,
                "start": b.start,
                "end": b.end,
                "indices": b.indices,
                "names": list(b.names),
            }
            for b in blocks
        ],
        "feature_names_meta": list(FEATURE_NAMES),
        "packages_text": sorted(set(META_PACKAGES.values())),
        "packages_video_optional": VIDEO_META_PACKAGES,
    }


def meta_indices(pca_k: int) -> list[int]:
    return list(range(pca_k, pca_k + len(FEATURE_NAMES)))


def block_indices(pca_k: int, block: str) -> list[int]:
    s = schema(pca_k)
    if block == "meta":
        return s["modality"]["meta"]
    if block == "embed" or block == "embed_pca":
        return s["modality"]["embed"]
    return s["meta_block_indices"][block]


if __name__ == "__main__":
    import json
    import sys

    k = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    json.dump(schema(k), sys.stdout, indent=2)
    sys.stdout.write("\n")
