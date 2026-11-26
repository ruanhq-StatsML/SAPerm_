"""Amazon Reviews 2023 — standalone, row-aligned extraction step.

One DataFrame is built once (text / y / shift columns). Every downstream
artifact — text meta ``M`` and each encoder's embedding ``Z`` — is derived from
that same ``df["text"]`` list and stored with ``row_id`` and ``text_sha1`` so
alignment can be re-verified on every load.

Specs:
  gift   Gift_Cards only, 500/500 by y, ordered by time (train_vs_heldout = early vs late).
  cross  Gift_Cards (source, the exact ``gift`` rows) then Magazine_Subscriptions
         (target, 500/500 by y); rows [0, n/2) are source, [n/2, n) target, so the
         half split used by ``domain_shift`` is exactly the domain split.

Usage::

    python -m experiments.meta_vs_embed_power.amazon_extract [--spec gift|cross] [--encoders MiniLM_L6 qwen3-0.6b]
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
TARGET_SRC = REPO / "datasets" / "magazineSubscription_df.csv"
N_ROWS = 1000
SEED = 2026
MIN_CHARS, MAX_CHARS = 20, 1000
ENCODERS = ("MiniLM_L6", "qwen3-0.6b")
N_SPOT = 20
SPECS = ("gift", "cross")


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def frame_hash(row_id: np.ndarray, text_sha1: np.ndarray) -> str:
    joined = "\n".join(f"{int(r)}:{h}" for r, h in zip(row_id, text_sha1))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def _stem(spec: str) -> str:
    return f"n{N_ROWS}" if spec == "gift" else f"cross_n{2 * N_ROWS}"


def frame_path(spec: str = "gift") -> Path:
    return OUT_DIR / f"frame_{_stem(spec)}.jsonl"


def meta_path(spec: str = "gift") -> Path:
    return OUT_DIR / f"meta_{_stem(spec)}.npz"


def emb_path(encoder: str, spec: str = "gift") -> Path:
    if spec == "gift":
        return OUT_DIR / f"emb_{encoder}_n{N_ROWS}.npz"
    return OUT_DIR / f"emb_cross_{encoder}_n{2 * N_ROWS}.npz"


def report_path(spec: str = "gift") -> Path:
    return OUT_DIR / ("alignment_report.json" if spec == "gift" else "alignment_report_cross.json")


def _filter(raw: pd.DataFrame) -> pd.DataFrame:
    """Shared filters: drop rating 3, 20-1000 chars, dedupe by text; y = rating >= 4."""
    raw = raw.copy()
    raw["src_line"] = np.arange(len(raw))
    raw["text"] = raw["text"].fillna("").astype(str)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], unit="ms") if np.issubdtype(
        raw["timestamp"].dtype, np.number
    ) else pd.to_datetime(raw["timestamp"])
    n_chars = raw["text"].str.len()
    keep = (raw["rating"] != 3) & (n_chars >= MIN_CHARS) & (n_chars <= MAX_CHARS)
    df = raw.loc[keep, ["src_line", "text", "rating", "timestamp"]]
    df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
    df["y"] = (df["rating"] >= 4).astype(int)
    return df


def _sample(df: pd.DataFrame) -> pd.DataFrame:
    take = stratified_cap(df["y"].to_numpy(dtype=float), N_ROWS, SEED)
    return df.iloc[take].sort_values(["timestamp", "src_line"]).reset_index(drop=True)


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    df.insert(0, "row_id", np.arange(len(df)))
    df["text_sha1"] = [sha1(t) for t in df["text"]]
    return df


def build_frame(spec: str = "gift") -> pd.DataFrame:
    if spec == "gift":
        df = _sample(_filter(pd.read_json(SRC, lines=True)))
        # train_vs_heldout uses first half vs second half of row order -> early vs late reviews.
        df["shift_time"] = (np.arange(len(df)) >= len(df) // 2).astype(int)
        return _finalize(df)

    src = load_frame("gift")
    src = src.rename(columns={"row_id": "gift_row_id"})[
        ["gift_row_id", "src_line", "text", "rating", "timestamp", "y"]
    ]
    src["domain"] = "Gift_Cards"
    tgt = _filter(pd.read_csv(TARGET_SRC))
    tgt = tgt[~tgt["text"].isin(set(src["text"]))]
    tgt = _sample(tgt)
    tgt["gift_row_id"] = -1
    tgt["domain"] = "Magazine_Subscriptions"
    df = pd.concat([src, tgt[src.columns]], ignore_index=True)
    df["shift_domain"] = (df["domain"] != "Gift_Cards").astype(int)
    return _finalize(df)


def load_frame(spec: str = "gift") -> pd.DataFrame:
    df = pd.read_json(frame_path(spec), lines=True, dtype={"text": str, "text_sha1": str})
    recomputed = np.array([sha1(t) for t in df["text"]])
    if not np.array_equal(recomputed, df["text_sha1"].to_numpy()):
        raise AssertionError(f"{spec} frame text_sha1 does not match stored text")
    n = len(df)
    half = (np.arange(n) >= n // 2).astype(int)
    shift_col = "shift_domain" if spec == "cross" else "shift_time"
    if not np.array_equal(df[shift_col].to_numpy(), half):
        raise AssertionError(f"{spec} frame: {shift_col} is not the half split of row order")
    if not np.array_equal(df["row_id"].to_numpy(), np.arange(n)):
        raise AssertionError(f"{spec} frame: row_id is not 0..n-1 in file order")
    if not np.array_equal(df["y"].to_numpy(), (df["rating"] >= 4).astype(int).to_numpy()):
        raise AssertionError(f"{spec} frame: y != rating >= 4")
    return df


def _keys(df: pd.DataFrame) -> dict:
    return {
        "row_id": df["row_id"].to_numpy(),
        "text_sha1": df["text_sha1"].to_numpy(dtype=object),
        "frame_hash": frame_hash(df["row_id"].to_numpy(), df["text_sha1"].to_numpy()),
    }


def extract_meta(df: pd.DataFrame, spec: str = "gift") -> None:
    from experiments.meta_vs_embed_power.features import meta_matrix

    M, names = meta_matrix(df["text"].tolist(), cache=None)
    np.savez_compressed(meta_path(spec), M=M, meta_names=np.array(names, dtype=object), **_keys(df))


def extract_embedding(df: pd.DataFrame, encoder: str, spec: str = "gift") -> None:
    from experiments.meta_vs_embed_power.model_configs import get_embedding

    Z = get_embedding(df["text"].tolist(), encoder)
    np.savez_compressed(emb_path(encoder, spec), Z=Z, encoder=encoder, **_keys(df))


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


def _spot_rows(n: int, spec: str) -> np.ndarray:
    rng = np.random.default_rng(SEED + 7)
    if spec == "gift":
        return np.sort(rng.choice(n, N_SPOT, replace=False))
    half = n // 2
    return np.sort(np.concatenate([
        rng.choice(half, N_SPOT // 2, replace=False),
        half + rng.choice(n - half, N_SPOT // 2, replace=False),
    ]))


def check_alignment(df: pd.DataFrame, encoders: list[str], *, spec: str = "gift", spot: bool = True) -> dict:
    """Full-row key/length checks + spot re-derivation of M and Z from single texts."""
    from experiments.meta_vs_embed_power.features import meta_matrix
    from experiments.meta_vs_embed_power.model_configs import MODEL_CONFIGS, get_embedding

    fh = frame_hash(df["row_id"].to_numpy(), df["text_sha1"].to_numpy())
    texts = df["text"].tolist()
    report: dict = {"spec": spec, "n_rows": len(df), "frame_hash": fh, "errors": []}

    meta = np.load(meta_path(spec), allow_pickle=True)
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

    if spec == "cross":
        gift = load_frame("gift")
        src = df[df["shift_domain"] == 0]
        same_src = np.array_equal(src["text_sha1"].to_numpy(), gift["text_sha1"].to_numpy()) and np.array_equal(
            src["gift_row_id"].to_numpy(), gift["row_id"].to_numpy()
        )
        report["source_rows_identical_to_gift_frame"] = bool(same_src)
        report["domain_text_overlap"] = int(
            len(set(src["text"]) & set(df.loc[df["shift_domain"] == 1, "text"]))
        )
        report["y_balance_by_domain"] = {
            d: {int(k): int(v) for k, v in g["y"].value_counts().items()} for d, g in df.groupby("domain")
        }
        report["corr_vader_compound_y_by_domain"] = {
            d: float(np.corrcoef(vc[g.index], g["y"])[0, 1]) for d, g in df.groupby("domain")
        }
        if not same_src:
            report["errors"].append("cross: source rows differ from gift frame")
        if report["domain_text_overlap"]:
            report["errors"].append("cross: texts shared across domains")

    idx = _spot_rows(len(df), spec)
    report["spot_rows"] = idx.tolist()
    if spot:
        M_spot, _ = meta_matrix([texts[i] for i in idx], cache=None)
        report["meta_spot_max_abs_diff"] = float(np.max(np.abs(M_spot - M[idx])))
        if not np.allclose(M_spot, M[idx], atol=1e-9):
            report["errors"].append("meta: spot re-derivation differs")

    report["embeddings"] = {}
    for enc in encoders:
        emb = np.load(emb_path(enc, spec), allow_pickle=True)
        report["errors"] += _check_keys(emb, df, fh, f"emb[{enc}]")
        Z = emb["Z"]
        r = {"shape": list(Z.shape)}
        if spot:
            Zc = Z[idx] / np.linalg.norm(Z[idx], axis=1, keepdims=True)

            # Strict: re-encode each spot row inside its original batch (same padding as the cache).
            bs = MODEL_CONFIGS[enc].batch_size
            Zb = np.empty_like(Zc)
            for b in np.unique(idx // bs):
                lo, hi = b * bs, min((b + 1) * bs, len(texts))
                Zbatch = get_embedding(texts[lo:hi], enc)
                sel = (idx >= lo) & (idx < hi)
                Zb[sel] = Zbatch[idx[sel] - lo]
            Zb /= np.linalg.norm(Zb, axis=1, keepdims=True)
            cos_b = np.sum(Zb * Zc, axis=1)
            r["spot_inbatch_cos_self_min"] = float(cos_b.min())
            r["spot_inbatch_self_is_argmax"] = bool(np.all(np.argmax(Zb @ Zc.T, axis=1) == np.arange(len(idx))))
            if cos_b.min() < 0.99999 or not r["spot_inbatch_self_is_argmax"]:
                report["errors"].append(f"emb[{enc}]: in-batch re-embedding does not reproduce stored row")

            # Diagnostic: single-text re-encode (no padding); padding sensitivity, must still be argmax.
            Zs = get_embedding([texts[i] for i in idx], enc, batch_size=1)
            Zs /= np.linalg.norm(Zs, axis=1, keepdims=True)
            cos_self = np.sum(Zs * Zc, axis=1)
            cos_shift = np.sum(Zs * np.roll(Zc, 1, axis=0), axis=1)
            r["spot_single_cos_self_min"] = float(cos_self.min())
            r["spot_single_cos_neighbor_max"] = float(cos_shift.max())
            r["spot_single_self_is_argmax"] = bool(np.all(np.argmax(Zs @ Zc.T, axis=1) == np.arange(len(idx))))
            if cos_self.min() < 0.99 or not r["spot_single_self_is_argmax"]:
                report["errors"].append(f"emb[{enc}]: single-text re-embedding not closest to stored row")
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


def run(encoders: list[str], *, spec: str = "gift", force: bool = False, max_attempts: int = 2) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    encoders = available_encoders(encoders)
    report: dict = {}
    for attempt in range(1, max_attempts + 1):
        if force or attempt > 1 or not frame_path(spec).exists():
            df = build_frame(spec)
            df.to_json(frame_path(spec), orient="records", lines=True, force_ascii=False, date_format="epoch")
            for p in [meta_path(spec), *[emb_path(e, spec) for e in encoders]]:
                p.unlink(missing_ok=True)
        df = load_frame(spec)
        if not meta_path(spec).exists():
            extract_meta(df, spec)
        for enc in encoders:
            if not emb_path(enc, spec).exists():
                extract_embedding(df, enc, spec)
        report = check_alignment(df, encoders, spec=spec)
        report["attempt"] = attempt
        if report["passed"]:
            break
        print(f"[amazon_extract] alignment failed (attempt {attempt}): {report['errors']}", file=sys.stderr)
    report["encoders"] = encoders
    report["source"] = str(SRC.relative_to(REPO))
    if spec == "cross":
        report["target"] = str(TARGET_SRC.relative_to(REPO))
    report["filters"] = {
        "rating_dropped": 3,
        "chars": [MIN_CHARS, MAX_CHARS],
        "dedupe": "text" if spec == "gift" else "text, within domain and across domains",
        "y": "rating >= 4",
        "order": "timestamp" if spec == "gift" else "domain (source then target), timestamp within domain",
    }
    report_path(spec).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _verified(encoder: str, spec: str):
    df = load_frame(spec)
    fh = frame_hash(df["row_id"].to_numpy(), df["text_sha1"].to_numpy())
    meta = np.load(meta_path(spec), allow_pickle=True)
    emb = np.load(emb_path(encoder, spec), allow_pickle=True)
    errs = _check_keys(meta, df, fh, "meta") + _check_keys(emb, df, fh, f"emb[{encoder}]")
    if str(emb["encoder"]) != encoder:
        errs.append(f"emb file encoder {emb['encoder']} != {encoder}")
    names = list(meta["meta_names"])
    if not np.array_equal(meta["M"][:, names.index("char_len")].astype(int), df["text"].str.len().to_numpy()):
        errs.append("meta char_len != len(frame text)")
    if errs:
        raise AssertionError(f"Amazon {spec} artifacts misaligned: {errs}")
    print(f"[aligned] {spec}/{encoder} n={len(df)} frame_hash={fh[:10]} meta+emb keys OK", file=sys.stderr)
    return df, meta, emb


def load_cache_embedding(encoder: str, spec: str = "gift") -> np.ndarray:
    return _verified(encoder, spec)[2]["Z"]


def load_aligned(
    encoder: str, pca_k: int, seed: int = SEED, spec: str = "gift"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return (Zpca, M, y, meta_names) after re-verifying keys against the frame."""
    df, meta, emb = _verified(encoder, spec)
    return pca_embed(emb["Z"], pca_k, seed=seed), meta["M"], df["y"].to_numpy(dtype=float), list(meta["meta_names"])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", choices=SPECS, default="gift")
    p.add_argument("--encoders", nargs="*", default=list(ENCODERS))
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    report = run(args.encoders, spec=args.spec, force=args.force)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
