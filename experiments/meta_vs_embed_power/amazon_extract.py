"""Amazon Reviews 2023 (Gift Cards) — standalone, row-aligned extraction step.

One DataFrame is built once (text / y / shift columns). Every downstream
artifact — text meta ``M`` and each encoder's embedding ``Z`` — is derived from
that same ``df["text"]`` list and stored with ``row_id`` and ``text_sha1`` so
alignment can be re-verified on every load.

Usage::

    python -m experiments.meta_vs_embed_power.amazon_extract [--encoders MiniLM_L6 qwen3-0.6b]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.meta_vs_embed_power.data import REPO, pca_embed, stratified_cap

OUT_DIR = Path(__file__).resolve().parent / "results" / "amazon"
SRC = REPO / "datasets" / "Gift_Cards.jsonl"
N_ROWS = 1000
SEED = 2026
MIN_CHARS, MAX_CHARS = 20, 1000
ENCODERS = ("MiniLM_L6", "qwen3-0.6b")
N_SPOT = 20


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def frame_hash(row_id: np.ndarray, text_sha1: np.ndarray) -> str:
    joined = "\n".join(f"{int(r)}:{h}" for r, h in zip(row_id, text_sha1))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def frame_path() -> Path:
    return OUT_DIR / f"frame_n{N_ROWS}.jsonl"


def meta_path() -> Path:
    return OUT_DIR / f"meta_n{N_ROWS}.npz"


def emb_path(encoder: str) -> Path:
    return OUT_DIR / f"emb_{encoder}_n{N_ROWS}.npz"


def build_frame() -> pd.DataFrame:
    """Binary sentiment y = rating >= 4 (rating 3 dropped); 500/500 stratified; ordered by time."""
    raw = pd.read_json(SRC, lines=True)
    raw["src_line"] = np.arange(len(raw))
    raw["text"] = raw["text"].astype(str)
    n_chars = raw["text"].str.len()
    keep = (raw["rating"] != 3) & (n_chars >= MIN_CHARS) & (n_chars <= MAX_CHARS)
    df = raw.loc[keep, ["src_line", "text", "rating", "timestamp"]]
    df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
    df["y"] = (df["rating"] >= 4).astype(int)

    take = stratified_cap(df["y"].to_numpy(dtype=float), N_ROWS, SEED)
    df = df.iloc[take].sort_values(["timestamp", "src_line"]).reset_index(drop=True)
    # train_vs_heldout uses first half vs second half of row order -> early vs late reviews.
    df["shift_time"] = (np.arange(len(df)) >= len(df) // 2).astype(int)
    df.insert(0, "row_id", np.arange(len(df)))
    df["text_sha1"] = [sha1(t) for t in df["text"]]
    return df


def load_frame() -> pd.DataFrame:
    df = pd.read_json(frame_path(), lines=True, dtype={"text": str, "text_sha1": str})
    recomputed = np.array([sha1(t) for t in df["text"]])
    if not np.array_equal(recomputed, df["text_sha1"].to_numpy()):
        raise AssertionError("frame text_sha1 does not match stored text")
    return df


def extract_meta(df: pd.DataFrame) -> None:
    from experiments.meta_vs_embed_power.features import meta_matrix

    M, names = meta_matrix(df["text"].tolist(), cache=None)
    np.savez_compressed(
        meta_path(),
        M=M,
        meta_names=np.array(names, dtype=object),
        row_id=df["row_id"].to_numpy(),
        text_sha1=df["text_sha1"].to_numpy(dtype=object),
        frame_hash=frame_hash(df["row_id"].to_numpy(), df["text_sha1"].to_numpy()),
    )


def extract_embedding(df: pd.DataFrame, encoder: str) -> None:
    from experiments.meta_vs_embed_power.model_configs import get_embedding

    Z = get_embedding(df["text"].tolist(), encoder)
    np.savez_compressed(
        emb_path(encoder),
        Z=Z,
        encoder=encoder,
        row_id=df["row_id"].to_numpy(),
        text_sha1=df["text_sha1"].to_numpy(dtype=object),
        frame_hash=frame_hash(df["row_id"].to_numpy(), df["text_sha1"].to_numpy()),
    )


def _check_keys(npz, df: pd.DataFrame, fh: str, label: str) -> list[str]:
    errs = []
    n = len(df)
    arr = npz["Z"] if "Z" in npz.files else npz["M"]
    if arr.shape[0] != n:
        errs.append(f"{label}: {arr.shape[0]} rows vs frame {n}")
    if not np.array_equal(npz["row_id"], df["row_id"].to_numpy()):
        errs.append(f"{label}: row_id mismatch")
    if not np.array_equal(npz["text_sha1"].astype(str), df["text_sha1"].to_numpy(dtype=str)):
        errs.append(f"{label}: text_sha1 mismatch")
    if str(npz["frame_hash"]) != fh:
        errs.append(f"{label}: frame_hash mismatch")
    return errs


def check_alignment(df: pd.DataFrame, encoders: list[str], *, spot: bool = True) -> dict:
    """Full-row key/length checks + spot re-derivation of M and Z from single texts."""
    from experiments.meta_vs_embed_power.features import meta_matrix
    from experiments.meta_vs_embed_power.model_configs import get_embedding

    fh = frame_hash(df["row_id"].to_numpy(), df["text_sha1"].to_numpy())
    texts = df["text"].tolist()
    report: dict = {"n_rows": len(df), "frame_hash": fh, "errors": []}

    meta = np.load(meta_path(), allow_pickle=True)
    report["errors"] += _check_keys(meta, df, fh, "meta")
    M, names = meta["M"], list(meta["meta_names"])
    char_ok = np.array_equal(M[:, names.index("char_len")].astype(int), df["text"].str.len().to_numpy())
    word_ok = np.array_equal(
        M[:, names.index("word_count")].astype(int), np.array([len(t.split()) for t in texts])
    )
    report["meta_char_len_all_rows_equal"] = bool(char_ok)
    report["meta_word_count_all_rows_equal"] = bool(word_ok)
    if not (char_ok and word_ok):
        report["errors"].append("meta: char_len/word_count do not match frame text")

    vc = M[:, names.index("vader_compound")]
    report["corr_vader_compound_y"] = float(np.corrcoef(vc, df["y"])[0, 1])
    report["y_balance"] = {int(k): int(v) for k, v in df["y"].value_counts().items()}
    report["y_equals_rating_ge_4"] = bool(np.array_equal(df["y"], (df["rating"] >= 4).astype(int)))

    rng = np.random.default_rng(SEED + 7)
    idx = np.sort(rng.choice(len(df), N_SPOT, replace=False))
    report["spot_rows"] = idx.tolist()
    if spot:
        M_spot, _ = meta_matrix([texts[i] for i in idx], cache=None)
        report["meta_spot_max_abs_diff"] = float(np.max(np.abs(M_spot - M[idx])))
        if not np.allclose(M_spot, M[idx], atol=1e-9):
            report["errors"].append("meta: spot re-derivation differs")

    report["embeddings"] = {}
    for enc in encoders:
        emb = np.load(emb_path(enc), allow_pickle=True)
        report["errors"] += _check_keys(emb, df, fh, f"emb[{enc}]")
        Z = emb["Z"]
        r = {"shape": list(Z.shape)}
        if spot:
            Zs = get_embedding([texts[i] for i in idx], enc, batch_size=1)
            Zs /= np.linalg.norm(Zs, axis=1, keepdims=True)
            Zc = Z[idx] / np.linalg.norm(Z[idx], axis=1, keepdims=True)
            cos_self = np.sum(Zs * Zc, axis=1)
            cos_shift = np.sum(Zs * np.roll(Zc, 1, axis=0), axis=1)
            r["spot_cos_self_min"] = float(cos_self.min())
            r["spot_cos_neighbor_max"] = float(cos_shift.max())
            r["spot_self_is_argmax"] = bool(np.all(np.argmax(Zs @ Zc.T, axis=1) == np.arange(len(idx))))
            if cos_self.min() < 0.999 or not r["spot_self_is_argmax"]:
                report["errors"].append(f"emb[{enc}]: spot re-embedding does not match stored row")
        report["embeddings"][enc] = r
    report["passed"] = not report["errors"]
    return report


def available_encoders(requested: list[str]) -> list[str]:
    ok = []
    for enc in requested:
        try:
            from transformers import AutoConfig

            from experiments.meta_vs_embed_power.model_configs import MODEL_CONFIGS

            cfg = MODEL_CONFIGS[enc]
            AutoConfig.from_pretrained(cfg.name, trust_remote_code=cfg.trust_remote_code)
            ok.append(enc)
        except Exception as exc:  # noqa: BLE001
            print(f"[amazon_extract] encoder {enc} unavailable: {exc}", file=sys.stderr)
    return ok


def run(encoders: list[str], *, force: bool = False, max_attempts: int = 2) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    encoders = available_encoders(encoders)
    report: dict = {}
    for attempt in range(1, max_attempts + 1):
        if force or attempt > 1 or not frame_path().exists():
            df = build_frame()
            df.to_json(frame_path(), orient="records", lines=True, force_ascii=False)
            for p in [meta_path(), *[emb_path(e) for e in encoders]]:
                p.unlink(missing_ok=True)
        df = load_frame()
        if not meta_path().exists():
            extract_meta(df)
        for enc in encoders:
            if not emb_path(enc).exists():
                extract_embedding(df, enc)
        report = check_alignment(df, encoders)
        report["attempt"] = attempt
        if report["passed"]:
            break
        print(f"[amazon_extract] alignment failed (attempt {attempt}): {report['errors']}", file=sys.stderr)
    report["encoders"] = encoders
    report["source"] = str(SRC.relative_to(REPO))
    report["filters"] = {
        "rating_dropped": 3,
        "chars": [MIN_CHARS, MAX_CHARS],
        "dedupe": "text",
        "y": "rating >= 4",
        "order": "timestamp",
    }
    (OUT_DIR / "alignment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_cache_embedding(encoder: str) -> np.ndarray:
    df = load_frame()
    emb = np.load(emb_path(encoder), allow_pickle=True)
    errs = _check_keys(emb, df, frame_hash(df["row_id"].to_numpy(), df["text_sha1"].to_numpy()), f"emb[{encoder}]")
    if errs:
        raise AssertionError(f"Amazon embedding misaligned: {errs}")
    return emb["Z"]


def load_aligned(encoder: str, pca_k: int, seed: int = SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return (Zpca, M, y, meta_names) after re-verifying keys against the frame."""
    df = load_frame()
    fh = frame_hash(df["row_id"].to_numpy(), df["text_sha1"].to_numpy())
    meta = np.load(meta_path(), allow_pickle=True)
    emb = np.load(emb_path(encoder), allow_pickle=True)
    errs = _check_keys(meta, df, fh, "meta") + _check_keys(emb, df, fh, f"emb[{encoder}]")
    if errs:
        raise AssertionError(f"Amazon artifacts misaligned: {errs}")
    return pca_embed(emb["Z"], pca_k, seed=seed), meta["M"], df["y"].to_numpy(dtype=float), list(meta["meta_names"])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoders", nargs="*", default=list(ENCODERS))
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    report = run(args.encoders, force=args.force)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())