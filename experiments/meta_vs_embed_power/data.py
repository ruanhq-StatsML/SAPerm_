"""Load gender / bias / BiasFinder tables + text for ``utils.text_meta``."""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]


def pca_embed(Z: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    k = min(k, Z.shape[0] - 1, Z.shape[1])
    return PCA(n_components=k, random_state=seed).fit_transform(
        StandardScaler().fit_transform(Z)
    )


def stratified_cap(y: np.ndarray, n_max: int | None, seed: int) -> np.ndarray:
    if n_max is None or len(y) <= n_max:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    half = n_max // 2
    pick = np.concatenate(
        [
            rng.choice(idx0, min(half, len(idx0)), replace=len(idx0) < half),
            rng.choice(idx1, n_max - half, replace=len(idx1) < half - len(idx0)),
        ]
    )
    return rng.permutation(pick)


def load_biasfinder(n_max: int | None = 1000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[str]]:
    emb_path = REPO / "datasets" / "train_biasfinder_bert_embedding.csv"
    zip_path = REPO / "datasets" / "dataset.zip"
    # File is class-sorted: truncating to the first n_max rows yields a single class.
    cap = 10**9
    texts, classes_txt = [], []
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("GEN-sarc-notsarc.csv") as raw:
            for i, row in enumerate(csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"))):
                if i >= cap:
                    break
                texts.append(row["text"])
                classes_txt.append(row["class"])

    Z_list, classes_emb = [], []
    with emb_path.open(encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        start = 1 if header[0] in ("", "Unnamed: 0") or header[0].lower().startswith("unnamed") else 0
        for i, row in enumerate(reader):
            if i >= cap:
                break
            classes_emb.append(row[-1])
            feats = row[start:-1]
            if len(feats) == 769:
                feats = feats[1:]
            Z_list.append([float(v) for v in feats])

    if classes_txt != classes_emb:
        raise RuntimeError("GEN text not aligned with BiasFinder embedding rows")
    Z = np.asarray(Z_list, dtype=float)
    y = np.asarray([1 if c == "sarc" else 0 for c in classes_emb], dtype=float)
    take = stratified_cap(y, n_max, seed)
    return Z[take], y[take], [texts[i] for i in take]


def load_gender(n_max: int | None = 1000, seed: int = 0) -> tuple[np.ndarray | None, np.ndarray, list[str]]:
    path = REPO / "datasets" / "explainability_gender_bias_data.csv"
    df = pd.read_csv(path, header=None).iloc[1:]
    texts = df.iloc[:, 1].astype(str).tolist()
    y = np.asarray([1 if s.strip().lower() == "positive" else 0 for s in df.iloc[:, 2]], dtype=float)
    take = stratified_cap(y, n_max, seed)
    return None, y[take], [texts[i] for i in take]


def load_train_bias_finder_tab(n_max: int | None = 1000, seed: int = 0) -> tuple[np.ndarray | None, np.ndarray, list[str]]:
    """``train_bias2`` (BiasFinder-style toxicity); pseudo-label from length quartile for smoke power."""
    path = REPO / "datasets" / "train_bias2.csv"
    df = pd.read_csv(path, delimiter="\t", header=None, on_bad_lines="skip")
    texts = df.iloc[:, 1].astype(str).tolist()
    lens = np.asarray([len(t) for t in texts], dtype=float)
    y = (lens >= np.quantile(lens, 0.75)).astype(float)
    take = stratified_cap(y, n_max, seed)
    return None, y[take], [texts[i] for i in take]


def load_equity_gender(n_max: int | None = 1200, seed: int = 0) -> tuple[np.ndarray | None, np.ndarray, list[str]]:
    path = REPO / "datasets" / "Equity-Evaluation-Corpus.csv"
    df = pd.read_csv(path)
    texts = df["Sentence"].astype(str).tolist()
    y = np.asarray([1 if g == "male" else 0 for g in df["Gender"]], dtype=float)
    take = stratified_cap(y, n_max, seed)
    return None, y[take], [texts[i] for i in take]

