"""HC3 human/ChatGPT pairs, loaded from the parquet conversion.

The original dataset script no longer loads on current ``datasets``.
The Hub still publishes the converted parquet shards.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parent / "results" / "cache"
PARQUET = {
    "finance": "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/refs%2Fconvert%2Fparquet/finance/train/0000.parquet",
    "open_qa": "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/refs%2Fconvert%2Fparquet/open_qa/train/0000.parquet",
    "medicine": "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/refs%2Fconvert%2Fparquet/medicine/train/0000.parquet",
    "wiki_csai": "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/refs%2Fconvert%2Fparquet/wiki_csai/train/0000.parquet",
}


def _first_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple, np.ndarray)):
        for item in value:
            text = _first_text(item)
            if text:
                return text
    return str(value).strip()


def _word_count(text: str) -> int:
    return len(text.split())


def _download(url: str, name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.parquet"
    if path.exists() and path.stat().st_size > 0:
        return path
    urllib.request.urlretrieve(url, path)
    return path


def load_pairs(configs: tuple[str, ...] = ("finance", "open_qa", "medicine", "wiki_csai"), max_per_config: int = 800) -> list[dict[str, str]]:
    """One human answer and one ChatGPT answer from each kept row."""
    rows: list[dict[str, str]] = []
    for config in configs:
        frame = pd.read_parquet(_download(PARQUET[config], config))
        kept = 0
        for record in frame.itertuples(index=False):
            human = _first_text(getattr(record, "human_answers"))
            machine = _first_text(getattr(record, "chatgpt_answers"))
            if _word_count(human) < 30 or _word_count(machine) < 30:
                continue
            human = " ".join(human.split()[:160])
            machine = " ".join(machine.split()[:160])
            rows.append({"text": human, "label": 0, "source": config})
            rows.append({"text": machine, "label": 1, "source": config})
            kept += 1
            if kept >= max_per_config:
                break
    return rows


def split_rows(
    rows: list[dict[str, str]],
    n_train: int,
    n_cal: int,
    n_test: int,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Balanced splits. Labels are 0 human, 1 AI. Each size must be even."""
    rng = np.random.default_rng(seed)
    pools = {0: [], 1: []}
    for row in rows:
        pools[int(row["label"])].append(row["text"])
    for label in (0, 1):
        order = rng.permutation(len(pools[label]))
        pools[label] = [pools[label][int(i)] for i in order]

    def take(n: int) -> dict[str, np.ndarray]:
        if n % 2 != 0:
            raise ValueError("split size must be even")
        texts: list[str] = []
        labels: list[int] = []
        for label in (0, 1):
            if len(pools[label]) < n // 2:
                raise RuntimeError(f"need {n // 2} texts for label {label}, found {len(pools[label])}")
            chunk = pools[label][: n // 2]
            pools[label] = pools[label][n // 2 :]
            texts.extend(chunk)
            labels.extend([label] * (n // 2))
        order = rng.permutation(n)
        return {
            "text": np.asarray(texts, dtype=object)[order],
            "label": np.asarray(labels, dtype=int)[order],
        }

    return {"train": take(n_train), "cal": take(n_cal), "test": take(n_test)}
