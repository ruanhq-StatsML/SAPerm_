"""CMU-MOSEI and MELD → LCO-Embedding-Omni-3B → PCA band PC 31–100.

Pipeline matches the MSR-VTT-style digestion: per-modality embeddings,
``pc30_100`` / ``pc31_100`` (0-indexed [30,100) ≡ PC 31–100), pooled PCA
concat, batch size 500.

- MOSEI: ``vintp/CMU-Mosei-text`` (sentiment ∈ [-3, 3], train vs test).
- MELD: utterance CSVs (sentiment → {-1,0,1}, train vs test).

When raw audio/video paths are unavailable, audio/video channels are encoded
with LCO Omni using modality-tagged text (ASR for MOSEI audio).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from experiments.multimodal_attribution.concatenate import (
    concatenate_modalities_pooled,
    stack_outcome,
)
from experiments.multimodal_attribution.embeddings import (
    extract_lco_omni_embeddings,
    extract_text_embeddings,
)

CACHE_DIR = Path("experiments/multimodal_attribution/results/emotion_cache")
MELD_CSV_DIR = Path(os.environ.get("MELD_CSV_DIR", "/tmp/meld_csv"))
DEFAULT_BAND = "pc31_100"
DEFAULT_BATCH = 500
DEFAULT_MODEL = "LCO_Omni_3B"
LCO_MODEL_ID = "LCO-Embedding/LCO-Embedding-Omni-3B"


def _sentiment_meld_to_float(s: str) -> float:
    m = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
    return float(m.get(str(s).strip().lower(), 0.0))


def load_mosei_frames(max_rows: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train / test frames from ``vintp/CMU-Mosei-text`` (~23k utterances)."""
    from datasets import load_dataset

    train = load_dataset("vintp/CMU-Mosei-text", split="train")
    test = load_dataset("vintp/CMU-Mosei-text", split="test")
    df_tr = train.to_pandas()
    df_te = test.to_pandas()
    if max_rows is not None:
        df_tr = df_tr.iloc[: int(max_rows)].copy()
        df_te = df_te.iloc[: int(max_rows)].copy()
    return df_tr, df_te


def load_meld_frames(
    csv_dir: Path = MELD_CSV_DIR, max_rows: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train / test MELD utterance CSVs (~13k rows)."""
    csv_dir = Path(csv_dir)
    train = pd.read_csv(csv_dir / "train_sent_emo.csv")
    test = pd.read_csv(csv_dir / "test_sent_emo.csv")
    if max_rows is not None:
        train = train.iloc[: int(max_rows)].copy()
        test = test.iloc[: int(max_rows)].copy()
    return train, test


def _cache_path(dataset: str, split: str, model_key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = model_key.replace("/", "_")
    return CACHE_DIR / f"{dataset}_{split}_{safe}.npz"


DEFAULT_LCO_BATCH = 8


def _embed_lco_texts(texts: Sequence[str], batch_size: int = DEFAULT_LCO_BATCH) -> np.ndarray:
    return extract_lco_omni_embeddings(list(map(str, texts)), batch_size=batch_size)


def _embed_channel(
    texts: Sequence[str], model_key: str, batch_size: int = DEFAULT_LCO_BATCH
) -> np.ndarray:
    if model_key in ("LCO_Omni_3B", LCO_MODEL_ID) or "LCO-Embedding" in model_key:
        return _embed_lco_texts(texts, batch_size=batch_size)
    return extract_text_embeddings(list(map(str, texts)), model_key=model_key)


def _mod_cache_path(dataset: str, split: str, modality: str, model_key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = model_key.replace("/", "_")
    return CACHE_DIR / f"{dataset}_{split}_{modality}_{safe}.npz"


def _load_or_embed_channel(
    texts: Sequence[str],
    *,
    dataset: str,
    split: str,
    modality: str,
    model_key: str,
    batch_size: int,
    force: bool = False,
) -> np.ndarray:
    """Per-modality checkpoint so a crash / restart does not redo finished channels."""
    path = _mod_cache_path(dataset, split, modality, model_key)
    if path.exists() and not force:
        data = np.load(path, allow_pickle=False)
        emb = np.asarray(data["emb"], dtype=float)
        if emb.shape[0] == len(texts):
            print(f"[cache hit] {path.name} shape={emb.shape}", flush=True)
            return emb
    emb = _embed_channel(texts, model_key=model_key, batch_size=batch_size)
    np.savez(path, emb=emb)
    print(f"[cache save] {path.name} shape={emb.shape}", flush=True)
    return emb


def build_modality_block(
    texts: Sequence[str],
    *,
    asr: Optional[Sequence[str]] = None,
    model_key: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_LCO_BATCH,
    dataset: str = "tmp",
    split: str = "tmp",
    force: bool = False,
) -> Dict[str, np.ndarray]:
    """LCO Omni embeddings for text / audio / video channels (checkpointed)."""
    text_emb = _load_or_embed_channel(
        texts,
        dataset=dataset,
        split=split,
        modality="text",
        model_key=model_key,
        batch_size=batch_size,
        force=force,
    )
    if asr is None:
        audio_inputs = [f"[audio] Spoken utterance: {t}" for t in texts]
    else:
        audio_inputs = [
            f"[audio] {a}" if str(a).strip() else f"[audio] Spoken utterance: {t}"
            for a, t in zip(asr, texts)
        ]
    audio_emb = _load_or_embed_channel(
        audio_inputs,
        dataset=dataset,
        split=split,
        modality="audio",
        model_key=model_key,
        batch_size=batch_size,
        force=force,
    )
    video_inputs = [f"[video] Scene dialogue: {t}" for t in texts]
    video_emb = _load_or_embed_channel(
        video_inputs,
        dataset=dataset,
        split=split,
        modality="video",
        model_key=model_key,
        batch_size=batch_size,
        force=force,
    )
    return {"text": text_emb, "audio": audio_emb, "video": video_emb}


def _save_split(path: Path, mods: Dict[str, np.ndarray], y: np.ndarray) -> None:
    np.savez(path, text=mods["text"], audio=mods["audio"], video=mods["video"], y=y)


def prepare_mosei_embeddings(
    model_key: str = DEFAULT_MODEL,
    max_rows: Optional[int] = 2000,
    force: bool = False,
    batch_size: int = DEFAULT_LCO_BATCH,
) -> Dict[str, Dict[str, np.ndarray]]:
    """``{ref,new}`` modality dicts + outcomes for MOSEI train vs test."""
    cache_ref = _cache_path("mosei", "train", model_key)
    cache_new = _cache_path("mosei", "test", model_key)
    if cache_ref.exists() and cache_new.exists() and not force:
        ref = dict(np.load(cache_ref, allow_pickle=False))
        new = dict(np.load(cache_new, allow_pickle=False))
        return {"ref": ref, "new": new}

    df_tr, df_te = load_mosei_frames(max_rows=max_rows)
    y_ref = np.asarray(df_tr["sentiment"], dtype=float)
    y_new = np.asarray(df_te["sentiment"], dtype=float)

    if cache_ref.exists() and not force:
        ref_mods = dict(np.load(cache_ref, allow_pickle=False))
        print(f"[cache hit] {cache_ref.name}", flush=True)
    else:
        ref_mods = build_modality_block(
            df_tr["text"].fillna("").tolist(),
            asr=df_tr["ASR"].fillna("").tolist() if "ASR" in df_tr.columns else None,
            model_key=model_key,
            batch_size=batch_size,
            dataset="mosei",
            split="train",
            force=force,
        )
        _save_split(cache_ref, ref_mods, y_ref)
        ref_mods["y"] = y_ref
        print(f"[cache save] {cache_ref.name}", flush=True)

    if cache_new.exists() and not force:
        new_mods = dict(np.load(cache_new, allow_pickle=False))
        print(f"[cache hit] {cache_new.name}", flush=True)
    else:
        new_mods = build_modality_block(
            df_te["text"].fillna("").tolist(),
            asr=df_te["ASR"].fillna("").tolist() if "ASR" in df_te.columns else None,
            model_key=model_key,
            batch_size=batch_size,
            dataset="mosei",
            split="test",
            force=force,
        )
        _save_split(cache_new, new_mods, y_new)
        new_mods["y"] = y_new
        print(f"[cache save] {cache_new.name}", flush=True)

    if "y" not in ref_mods:
        ref_mods["y"] = y_ref
    if "y" not in new_mods:
        new_mods["y"] = y_new
    return {"ref": ref_mods, "new": new_mods}


def prepare_meld_embeddings(
    model_key: str = DEFAULT_MODEL,
    max_rows: Optional[int] = 2000,
    force: bool = False,
    csv_dir: Path = MELD_CSV_DIR,
    batch_size: int = DEFAULT_LCO_BATCH,
) -> Dict[str, Dict[str, np.ndarray]]:
    cache_ref = _cache_path("meld", "train", model_key)
    cache_new = _cache_path("meld", "test", model_key)
    if cache_ref.exists() and cache_new.exists() and not force:
        ref = dict(np.load(cache_ref, allow_pickle=False))
        new = dict(np.load(cache_new, allow_pickle=False))
        return {"ref": ref, "new": new}

    df_tr, df_te = load_meld_frames(csv_dir=csv_dir, max_rows=max_rows)
    y_ref = np.asarray([_sentiment_meld_to_float(s) for s in df_tr["Sentiment"]], dtype=float)
    y_new = np.asarray([_sentiment_meld_to_float(s) for s in df_te["Sentiment"]], dtype=float)

    if cache_ref.exists() and not force:
        ref_mods = dict(np.load(cache_ref, allow_pickle=False))
        print(f"[cache hit] {cache_ref.name}", flush=True)
    else:
        ref_mods = build_modality_block(
            df_tr["Utterance"].fillna("").tolist(),
            model_key=model_key,
            batch_size=batch_size,
            dataset="meld",
            split="train",
            force=force,
        )
        _save_split(cache_ref, ref_mods, y_ref)
        ref_mods["y"] = y_ref
        print(f"[cache save] {cache_ref.name}", flush=True)

    if cache_new.exists() and not force:
        new_mods = dict(np.load(cache_new, allow_pickle=False))
        print(f"[cache hit] {cache_new.name}", flush=True)
    else:
        new_mods = build_modality_block(
            df_te["Utterance"].fillna("").tolist(),
            model_key=model_key,
            batch_size=batch_size,
            dataset="meld",
            split="test",
            force=force,
        )
        _save_split(cache_new, new_mods, y_new)
        new_mods["y"] = y_new
        print(f"[cache save] {cache_new.name}", flush=True)

    if "y" not in ref_mods:
        ref_mods["y"] = y_ref
    if "y" not in new_mods:
        new_mods["y"] = y_new
    return {"ref": ref_mods, "new": new_mods}


def sample_emotion_batch(
    pack: Dict[str, Dict[str, np.ndarray]],
    *,
    n: int = DEFAULT_BATCH,
    band: str = DEFAULT_BAND,
    seed: int = 0,
    combo: Sequence[str] = ("text", "audio", "video"),
) -> dict:
    """Draw n/domain rows, PCA-band concat (default PC 31–100), return OOD batch."""
    rng = np.random.default_rng(seed)
    ref_all = pack["ref"]
    new_all = pack["new"]
    n_ref = min(n, len(ref_all["y"]))
    n_new = min(n, len(new_all["y"]))
    idx_r = rng.choice(len(ref_all["y"]), size=n_ref, replace=False)
    idx_n = rng.choice(len(new_all["y"]), size=n_new, replace=False)
    ref_mods = {k: np.asarray(ref_all[k])[idx_r] for k in combo}
    new_mods = {k: np.asarray(new_all[k])[idx_n] for k in combo}
    y_ref = np.asarray(ref_all["y"], dtype=float)[idx_r]
    y_new = np.asarray(new_all["y"], dtype=float)[idx_n]
    X_ref, X_new, indices, _ = concatenate_modalities_pooled(
        ref_mods, new_mods, band=band, seed=seed
    )
    return {
        "X": np.vstack([X_ref, X_new]),
        "Y": np.concatenate([y_ref, y_new]),
        "T": np.concatenate([np.zeros(n_ref), np.ones(n_new)]).astype(int),
        "df_ref": stack_outcome(X_ref, y_ref),
        "df_new": stack_outcome(X_new, y_new),
        "modality_indices": indices,
        "band": band,
        "n_per_batch": int(n_ref),
        "seed": seed,
        "combo": list(combo),
    }
