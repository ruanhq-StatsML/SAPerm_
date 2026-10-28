"""Load a tabular panel and name the feature blocks."""

from __future__ import annotations

from pathlib import Path

import re

import numpy as np
import pandas as pd


def load_movielens(
    data_dir: str = "data/movielens",
    min_ratings: int = 20,
    seed: int = 0,
    span: str = "calendar",
    n_batches: int = 8,
    per_batch: int | None = None,
):
    """MovieLens-1M as user, item, and context blocks.

    Ground truth is planted for the smoke test: from batch 3 onward the user
    block is marked drifted. The raw ratings file has no block label.
    """
    del seed
    root = Path(data_dir) / "ml-1m"
    ratings = pd.read_csv(
        root / "ratings.dat",
        sep="::", engine="python", header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    users = pd.read_csv(
        root / "users.dat",
        sep="::", engine="python", header=None,
        names=["user_id", "gender", "age", "occupation", "zip"],
    )
    movies = pd.read_csv(
        root / "movies.dat",
        sep="::", engine="python", header=None,
        names=["item_id", "title", "genres"], encoding="latin-1",
    )
    user_counts = ratings["user_id"].value_counts()
    item_counts = ratings["item_id"].value_counts()
    ratings = ratings[
        ratings["user_id"].isin(user_counts[user_counts >= min_ratings].index)
        & ratings["item_id"].isin(item_counts[item_counts >= min_ratings].index)
    ].copy()
    ratings = ratings.merge(users, on="user_id", how="left")
    ratings = ratings.merge(movies, on="item_id", how="left")

    ratings["gender_num"] = (ratings["gender"] == "M").astype(np.float32)
    ratings["age_num"] = ratings["age"].astype(np.float32)
    ratings["occupation_num"] = ratings["occupation"].astype(np.float32)
    ratings["zip_num"] = pd.to_numeric(ratings["zip"].astype(str).str[:1], errors="coerce").fillna(0)

    genre_dummies = ratings["genres"].fillna("(no genres listed)").str.get_dummies(sep="|")
    top_genres = genre_dummies.sum().nlargest(10).index.tolist()
    for genre in top_genres:
        ratings[f"genre_{genre}"] = genre_dummies[genre].astype(np.float32)

    clock = pd.to_datetime(ratings["timestamp"], unit="s")
    ratings["hour"] = clock.dt.hour.astype(np.float32)
    ratings["day_of_week"] = clock.dt.dayofweek.astype(np.float32)

    user_cols = ["gender_num", "age_num", "occupation_num", "zip_num"]
    item_cols = [f"genre_{genre}" for genre in top_genres]
    context_cols = ["rating", "hour", "day_of_week"]
    col_names = user_cols + item_cols + context_cols
    values = ratings[col_names].to_numpy(dtype=np.float32)
    blocks = {
        "user": list(range(0, len(user_cols))),
        "item": list(range(len(user_cols), len(user_cols) + len(item_cols))),
        "context": list(range(len(user_cols) + len(item_cols), len(col_names))),
    }
    if span == "calendar":
        day = (ratings["timestamp"] - ratings["timestamp"].min()) / (60 * 60 * 24)
        n_keep = min(int(day.max() // 30) + 1, n_batches)
        batched = []
        for t in range(n_keep):
            mask = (day >= t * 30) & (day < (t + 1) * 30)
            if int(mask.sum()) < 200:
                continue
            chunk = values[mask.to_numpy()]
            if per_batch is not None and len(chunk) > per_batch:
                take = np.linspace(0, len(chunk) - 1, per_batch).astype(int)
                chunk = chunk[take]
            batched.append(chunk)
        gt = {t: ({"user"} if t >= 3 else set()) for t in range(1, len(batched))}
    elif span == "equal":
        order = np.argsort(ratings["timestamp"].to_numpy(), kind="mergesort")
        batched = _batches_from_order(values[order], n_batches, min_rows=200, per_batch=per_batch)
        gt = {t: set() for t in range(len(batched))}
    else:
        raise ValueError(span)
    return {
        "X_batched": batched,
        "blocks": blocks,
        "gt": gt,
        "col_names": col_names,
        "F": len(col_names),
    }


def load_synthetic_panel(N=800, T=8, F=100, seed=0):
    """Gaussian panel. From batch 3, block_1 is shifted by 1.5."""
    rng = np.random.RandomState(seed)
    X = rng.randn(N, T, F).astype(np.float32)
    blocks = {
        "block_0": list(range(0, 25)),
        "block_1": list(range(25, 50)),
        "block_2": list(range(50, 75)),
        "block_3": list(range(75, 100)),
    }
    gt = {t: set() for t in range(T)}
    for t in range(3, T):
        X[:, t, blocks["block_1"]] += 1.5
        gt[t] |= {"block_1"}
    return {
        "X_batched": [X[:, t] for t in range(T)],
        "blocks": blocks,
        "gt": gt,
        "col_names": [f"f{j}" for j in range(F)],
        "F": F,
    }


def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.select_dtypes(include=["number"]).copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.fillna(out.median(numeric_only=True))
    keep = [c for c in out.columns if out[c].std(ddof=0) > 0]
    return out[keep].astype(np.float32)


def _batches_from_order(values: np.ndarray, n_batches: int, min_rows: int, per_batch: int | None = None):
    """Split an ordered table into equal-count batches.

    `per_batch` keeps evenly spaced rows inside each batch so a long panel
    can be scored without fitting the generator on every row.
    """
    n_batches = max(2, n_batches)
    edges = np.linspace(0, len(values), n_batches + 1).astype(int)
    batched = []
    for start, stop in zip(edges[:-1], edges[1:]):
        if stop - start < min_rows:
            continue
        if per_batch is None or stop - start <= per_batch:
            batched.append(values[start:stop])
            continue
        take = np.linspace(start, stop - 1, per_batch).astype(int)
        batched.append(values[take])
    return batched


def _blocks_from_groups(columns, groups):
    """`groups` is (block name, predicate). The first matching block owns a column."""
    names = list(columns)
    blocks = {}
    used = set()
    for block, predicate in groups:
        cols = [i for i, name in enumerate(names) if i not in used and predicate(name)]
        if cols:
            blocks[block] = cols
            used.update(cols)
    rest = [i for i in range(len(names)) if i not in used]
    if rest:
        blocks["other"] = rest
    if len(blocks) < 2:
        raise ValueError("need at least two feature blocks")
    return blocks


def load_ieee_cis(
    data_dir: str = "data/ieee_cis",
    max_rows: int | None = None,
    seed: int = 0,
    n_batches: int = 8,
    per_batch: int | None = None,
    drop_clock: bool = False,
):
    """IEEE-CIS fraud transactions, blocked by column family.

    Expects train_transaction.csv and, when present, train_identity.csv.
    TransactionID and isFraud are dropped. Vesta columns V1, V2, ... form
    their own block. There is no planted block label, so gt is empty.
    """
    del seed
    root = Path(data_dir)
    transaction_path = root / "train_transaction.csv"
    if not transaction_path.exists():
        raise FileNotFoundError(
            f"missing {transaction_path}. Place the Kaggle IEEE-CIS train files in {root}."
        )
    transaction = pd.read_csv(transaction_path)
    identity_path = root / "train_identity.csv"
    if identity_path.exists():
        identity = pd.read_csv(identity_path)
        frame = transaction.merge(identity, on="TransactionID", how="left")
    else:
        frame = transaction
    if "TransactionDT" in frame.columns:
        frame = frame.sort_values("TransactionDT")
        frame["hour"] = (frame["TransactionDT"] % (24 * 60 * 60)) / 3600.0
        frame["day"] = frame["TransactionDT"] / (24 * 60 * 60)
    if max_rows is not None:
        frame = frame.iloc[:max_rows]
    drop = ["TransactionID", "isFraud"]
    if drop_clock:
        drop.extend(["TransactionDT", "day"])
    frame = frame.drop(columns=[c for c in drop if c in frame.columns])
    numeric = _numeric_frame(frame)
    groups = [
        ("time", lambda name: name in {"TransactionDT", "hour", "day"}),
        ("amount", lambda name: name == "TransactionAmt"),
        ("card", lambda name: name.startswith("card")),
        ("address", lambda name: name.startswith("addr") or name.startswith("dist")),
        ("counts", lambda name: re.fullmatch(r"C\d+", name) is not None),
        ("timedelta", lambda name: re.fullmatch(r"D\d+", name) is not None),
        ("match", lambda name: re.fullmatch(r"M\d+", name) is not None),
        ("vesta", lambda name: re.fullmatch(r"V\d+", name) is not None),
        ("identity", lambda name: name.startswith("id_") or name.startswith("Device")),
    ]
    blocks = _blocks_from_groups(numeric.columns, groups)
    values = numeric.to_numpy(dtype=np.float32)
    batched = _batches_from_order(values, n_batches=n_batches, min_rows=200, per_batch=per_batch)
    return {
        "X_batched": batched,
        "blocks": blocks,
        "gt": {t: set() for t in range(len(batched))},
        "col_names": list(numeric.columns),
        "F": numeric.shape[1],
    }


def load_home_credit(
    data_dir: str = "data/home_credit",
    max_rows: int | None = None,
    seed: int = 0,
    n_batches: int = 8,
    per_batch: int | None = None,
):
    """Home Credit application_train.csv, blocked by column family.

    The file has no event time. Rows are split in SK_ID_CURR order. gt is empty.
    """
    del seed
    path = Path(data_dir) / "application_train.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}. Place the Kaggle Home Credit application file in {data_dir}."
        )
    frame = pd.read_csv(path)
    if "SK_ID_CURR" in frame.columns:
        frame = frame.sort_values("SK_ID_CURR")
    if max_rows is not None:
        frame = frame.iloc[:max_rows]
    numeric = _numeric_frame(frame.drop(columns=["SK_ID_CURR", "TARGET"], errors="ignore"))
    groups = [
        ("external", lambda name: name.startswith("EXT_SOURCE")),
        ("amount", lambda name: name.startswith("AMT_")),
        ("personal", lambda name: name in {
            "DAYS_BIRTH", "DAYS_EMPLOYED", "CNT_CHILDREN", "CNT_FAM_MEMBERS",
        } or name.startswith("FLAG_OWN")),
        ("region", lambda name: name.startswith("REGION") or name.startswith("DAYS_")),
    ]
    blocks = _blocks_from_groups(numeric.columns, groups)
    values = numeric.to_numpy(dtype=np.float32)
    batched = _batches_from_order(values, n_batches=n_batches, min_rows=200, per_batch=per_batch)
    return {
        "X_batched": batched,
        "blocks": blocks,
        "gt": {t: set() for t in range(len(batched))},
        "col_names": list(numeric.columns),
        "F": numeric.shape[1],
    }
