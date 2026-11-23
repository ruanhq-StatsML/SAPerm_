"""Load real-data windows for PO-risk Shapley.

Each loader returns ``(X_ref, y_ref, X_new, y_new, names, meta)``.
``meta['drivers']`` is an optional index list used only for recovery rates
when the window was built to move a known feature family.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

REPO = Path(__file__).resolve().parents[2]
BIAS_TRAIN = REPO / "datasets" / "train_biasfinder_bert_embedding.csv"
BIAS_TEST = REPO / "datasets" / "test_biasfinder_bert_embedding.csv"
MODA_NPZ = REPO / "agents" / "shift-detect" / "python" / "results" / "mm3_cache_modasense_n500.npz"
MODA_PARQUET = (
    REPO / "agents" / "shift-detect" / "python" / "data" / "modasense" / "data" / "test-00000-of-00001.parquet"
)

_NEG = frozenset({"fear", "sad", "angry", "disgust"})


def _as_binary(y) -> np.ndarray:
    y = np.asarray(y)
    if y.dtype.kind in "iufb":
        vals = np.unique(y)
        if set(vals.tolist()) <= {0, 1}:
            return y.astype(float)
        return (y == vals.max()).astype(float)
    s = pd.Series(y).astype(str).str.lower()
    if set(s.unique()) <= {"0", "1"}:
        return s.astype(float).to_numpy()
    # common adult / bank encodings
    pos = {"1", "yes", ">50k", "good", "sarc", "true"}
    return s.isin(pos).astype(float).to_numpy()


def _sample_pair(X, y, n_ref: int, n_new: int, rng: np.random.Generator):
    n = len(y)
    take = min(n, n_ref + n_new)
    idx = rng.choice(n, size=take, replace=False)
    if take < n_ref + n_new:
        # with replacement only if the pool is tiny
        idx = rng.choice(n, size=n_ref + n_new, replace=True)
    a, b = idx[:n_ref], idx[n_ref : n_ref + n_new]
    return X[a], y[a], X[b], y[b]


def biasfinder_pca(
    *,
    scenario: str = "train_vs_test",
    n_ref: int = 120,
    n_new: int = 120,
    pca_k: int = 24,
    seed: int = 0,
):
    """BiasFinder BERT embeddings (768-d → PCA-k) with train/test or class mix."""
    rng = np.random.default_rng(seed)
    train = pd.read_csv(BIAS_TRAIN)
    test = pd.read_csv(BIAS_TEST)
    feat_cols = [c for c in train.columns if c not in ("Unnamed: 0", "class", "sentiment")]
    Xt = train[feat_cols].to_numpy(dtype=float)
    Xs = test[feat_cols].to_numpy(dtype=float)
    yt = _as_binary(train["class"].map({"notsarc": 0, "sarc": 1}))
    ys = _as_binary(test["sentiment"])

    scaler = StandardScaler().fit(Xt)
    pca = PCA(n_components=pca_k, random_state=seed)
    Zt = pca.fit_transform(scaler.transform(Xt))
    Zs = pca.transform(scaler.transform(Xs))
    names = [f"pca_{i}" for i in range(pca_k)]

    if scenario == "train_vs_test":
        i0 = rng.choice(len(Zt), size=n_ref, replace=False)
        i1 = rng.choice(len(Zs), size=n_new, replace=False)
        return Zt[i0], yt[i0], Zs[i1], ys[i1], names, {"dataset": "biasfinder", "scenario": scenario, "drivers": []}

    if scenario == "random_split":
        X = Zt
        y = yt
        X0, X1, y0, y1 = train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y)
        i0 = rng.choice(len(X0), size=min(n_ref, len(X0)), replace=False)
        i1 = rng.choice(len(X1), size=min(n_new, len(X1)), replace=False)
        return X0[i0], y0[i0], X1[i1], y1[i1], names, {"dataset": "biasfinder", "scenario": scenario, "drivers": []}

    if scenario == "sarc_enrich":
        # ref: mostly notsarc; new: mostly sarc — label mix shift with same P(X) family
        i_neg = np.flatnonzero(yt == 0)
        i_pos = np.flatnonzero(yt == 1)
        r_neg = int(0.8 * n_ref)
        r_pos = n_ref - r_neg
        n_pos = int(0.8 * n_new)
        n_neg = n_new - n_pos
        ref = np.concatenate(
            [rng.choice(i_neg, r_neg, replace=False), rng.choice(i_pos, r_pos, replace=False)]
        )
        new = np.concatenate(
            [rng.choice(i_pos, n_pos, replace=False), rng.choice(i_neg, n_neg, replace=False)]
        )
        return Zt[ref], yt[ref], Zt[new], yt[new], names, {"dataset": "biasfinder", "scenario": scenario, "drivers": []}

    raise ValueError(scenario)


def modasense(
    *,
    scenario: str = "text_polarity",
    n_ref: int = 80,
    n_new: int = 80,
    seed: int = 0,
):
    """ModaSense/emotion meta matrix with designed modality windows."""
    rng = np.random.default_rng(seed)
    z = np.load(MODA_NPZ, allow_pickle=True)
    X = np.asarray(z["M"], dtype=float)
    names = [str(n) for n in z["names"].tolist()]
    df = pd.read_parquet(MODA_PARQUET).iloc[: len(X)].reset_index(drop=True)
    y_text = np.array([1.0 if str(l).lower() in _NEG else 0.0 for l in df["text_label"]], dtype=float)

    name_to_i = {n: i for i, n in enumerate(names)}
    vader = [name_to_i[n] for n in names if n.startswith("vader_")]
    img = [name_to_i[n] for n in names if n.startswith("img_")]
    aud = [name_to_i[n] for n in names if n.startswith("aud_")]

    if scenario == "random_split":
        X0, X1, y0, y1 = train_test_split(X, y_text, test_size=0.5, random_state=seed, stratify=y_text)
        i0 = rng.choice(len(X0), size=min(n_ref, len(X0)), replace=False)
        i1 = rng.choice(len(X1), size=min(n_new, len(X1)), replace=False)
        return (
            X0[i0],
            y0[i0],
            X1[i1],
            y1[i1],
            names,
            {"dataset": "modasense", "scenario": scenario, "drivers": []},
        )

    if scenario == "text_polarity":
        # ref: non-neg text labels; new: neg text labels
        i0 = np.flatnonzero(y_text == 0)
        i1 = np.flatnonzero(y_text == 1)
        a = rng.choice(i0, size=min(n_ref, len(i0)), replace=False)
        b = rng.choice(i1, size=min(n_new, len(i1)), replace=False)
        return X[a], y_text[a], X[b], y_text[b], names, {
            "dataset": "modasense",
            "scenario": scenario,
            "drivers": vader,
        }

    if scenario == "img_brightness":
        j = name_to_i["img_brightness"]
        med = float(np.median(X[:, j]))
        i0 = np.flatnonzero(X[:, j] <= med)
        i1 = np.flatnonzero(X[:, j] > med)
        a = rng.choice(i0, size=min(n_ref, len(i0)), replace=False)
        b = rng.choice(i1, size=min(n_new, len(i1)), replace=False)
        return X[a], y_text[a], X[b], y_text[b], names, {
            "dataset": "modasense",
            "scenario": scenario,
            "drivers": img,
        }

    if scenario == "audio_energy":
        j = name_to_i["aud_rms_energy"]
        med = float(np.median(X[:, j]))
        i0 = np.flatnonzero(X[:, j] <= med)
        i1 = np.flatnonzero(X[:, j] > med)
        a = rng.choice(i0, size=min(n_ref, len(i0)), replace=False)
        b = rng.choice(i1, size=min(n_new, len(i1)), replace=False)
        return X[a], y_text[a], X[b], y_text[b], names, {
            "dataset": "modasense",
            "scenario": scenario,
            "drivers": aud,
        }

    raise ValueError(scenario)


def _openml_xy(data_id: int | None = None, name: str | None = None, version: int | None = None):
    from sklearn.datasets import fetch_openml

    kwargs = {"as_frame": True, "parser": "liac-arff"}
    if data_id is not None:
        kwargs["data_id"] = data_id
    else:
        kwargs["name"] = name
        if version is not None:
            kwargs["version"] = version
    bunch = fetch_openml(**kwargs)
    X = bunch.data.copy()
    y = _as_binary(bunch.target)
    # encode categoricals
    cat = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    parts = []
    names = []
    if num:
        parts.append(X[num].to_numpy(dtype=float))
        names.extend(num)
    if cat:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        parts.append(enc.fit_transform(X[cat].astype(str)))
        names.extend(cat)
    Xn = np.hstack(parts) if parts else np.zeros((len(X), 0))
    Xn = np.nan_to_num(Xn, nan=0.0)
    return Xn, y, names, X


def openml_split(
    *,
    dataset: str,
    scenario: str = "random_split",
    n_ref: int = 150,
    n_new: int = 150,
    seed: int = 0,
):
    """Adult / bank-marketing / credit-g with random or protected-attribute windows."""
    rng = np.random.default_rng(seed)
    if dataset == "adult":
        X, y, names, raw = _openml_xy(data_id=1590)
        split_col = "sex"
    elif dataset == "bank":
        X, y, names, raw = _openml_xy(data_id=1461)
        # age median split (no clean protected attr required)
        split_col = "age" if "age" in raw.columns else None
    elif dataset == "credit_g":
        X, y, names, raw = _openml_xy(data_id=31)
        split_col = "age" if "age" in raw.columns else None
    else:
        raise ValueError(dataset)

    name_to_i = {n: i for i, n in enumerate(names)}

    if scenario == "random_split":
        X0, X1, y0, y1 = train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y if len(np.unique(y)) > 1 else None)
        i0 = rng.choice(len(X0), size=min(n_ref, len(X0)), replace=False)
        i1 = rng.choice(len(X1), size=min(n_new, len(X1)), replace=False)
        return X0[i0], y0[i0], X1[i1], y1[i1], names, {"dataset": dataset, "scenario": scenario, "drivers": []}

    if scenario == "attr_split":
        if split_col is None or split_col not in raw.columns:
            raise ValueError(f"no split column for {dataset}")
        col = raw[split_col]
        # drop the split attribute from X when present
        keep = [i for i, n in enumerate(names) if n != split_col]
        Xk = X[:, keep]
        names_k = [names[i] for i in keep]
        if col.dtype.kind in "iufc" or pd.api.types.is_numeric_dtype(col):
            med = float(np.median(pd.to_numeric(col, errors="coerce")))
            mask0 = pd.to_numeric(col, errors="coerce").to_numpy() <= med
        else:
            levels = pd.Series(col.astype(str)).unique().tolist()
            mask0 = col.astype(str).to_numpy() == levels[0]
        i0 = np.flatnonzero(mask0)
        i1 = np.flatnonzero(~mask0)
        a = rng.choice(i0, size=min(n_ref, len(i0)), replace=False)
        b = rng.choice(i1, size=min(n_new, len(i1)), replace=False)
        # no guaranteed driver set after dropping the attribute
        return Xk[a], y[a], Xk[b], y[b], names_k, {
            "dataset": dataset,
            "scenario": scenario,
            "split_col": split_col,
            "drivers": [],
        }

    raise ValueError(scenario)


SCENARIOS = [
    ("biasfinder", "random_split"),
    ("biasfinder", "train_vs_test"),
    ("biasfinder", "sarc_enrich"),
    ("modasense", "random_split"),
    ("modasense", "text_polarity"),
    ("modasense", "img_brightness"),
    ("modasense", "audio_energy"),
    ("adult", "random_split"),
    ("adult", "attr_split"),
    ("bank", "random_split"),
    ("bank", "attr_split"),
    ("credit_g", "random_split"),
    ("credit_g", "attr_split"),
]


def load_window(dataset: str, scenario: str, *, seed: int = 0, n_ref: int = 120, n_new: int = 120):
    if dataset == "biasfinder":
        return biasfinder_pca(scenario=scenario, n_ref=n_ref, n_new=n_new, seed=seed)
    if dataset == "modasense":
        return modasense(scenario=scenario, n_ref=min(n_ref, 100), n_new=min(n_new, 100), seed=seed)
    if dataset in {"adult", "bank", "credit_g"}:
        return openml_split(dataset=dataset, scenario=scenario, n_ref=n_ref, n_new=n_new, seed=seed)
    raise ValueError(dataset)
