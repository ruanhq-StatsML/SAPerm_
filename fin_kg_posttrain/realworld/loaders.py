"""Load real corpora from datasets/dataset.zip (or an unpacked directory)."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ZIP = REPO_ROOT / "datasets" / "dataset.zip"


@dataclass
class RealCorpus:
    name: str
    rows: List[Dict[str, str]]
    text_col: str
    label_cols: List[str]


def list_corpora() -> List[str]:
    return [
        "equity_evaluation",
        "biasfinder",
        "sarcasm",
        "amazon_magazine",
        "amazon_gift_cards",
        "semeval_sarc",
    ]


def _open_zip_member(zpath: Path, member: str) -> io.TextIOWrapper:
    zf = zipfile.ZipFile(zpath)
    return io.TextIOWrapper(zf.open(member), encoding="utf-8")


def _read_csv_dict(f, **kwargs) -> List[Dict[str, str]]:
    reader = csv.DictReader(f, **kwargs)
    return [{k: (v if v is not None else "") for k, v in row.items()} for row in reader]


def load_equity_evaluation(zpath: Path = DEFAULT_ZIP) -> RealCorpus:
    with _open_zip_member(zpath, "Equity-Evaluation-Corpus.csv") as f:
        rows = _read_csv_dict(f)
    # normalize text column name used by FT wrappers
    for r in rows:
        r["text"] = r.get("Sentence", "")
    return RealCorpus(
        name="equity_evaluation",
        rows=rows,
        text_col="text",
        label_cols=["Gender", "Race", "Emotion", "Person"],
    )


def load_biasfinder(zpath: Path = DEFAULT_ZIP) -> RealCorpus:
    rows: List[Dict[str, str]] = []
    for member, split in (("train_bias2.csv", "train"), ("test_bias2.csv", "test")):
        with _open_zip_member(zpath, member) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                # TSV: sentiment \t text
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    continue
                sent, text = parts
                rows.append(
                    {
                        "text": text,
                        "sentiment": sent,
                        "class": "pos" if sent.strip() == "1" else "neg",
                        "split": split,
                    }
                )
    return RealCorpus(
        name="biasfinder",
        rows=rows,
        text_col="text",
        label_cols=["sentiment", "class", "split"],
    )


def load_sarcasm(zpath: Path = DEFAULT_ZIP) -> RealCorpus:
    rows: List[Dict[str, str]] = []
    for member in ("GEN-sarc-notsarc.csv", "HYP-sarc-notsarc.csv", "RQ-sarc-notsarc.csv"):
        src = member.split("-")[0]
        with _open_zip_member(zpath, member) as f:
            for r in _read_csv_dict(f):
                r["source"] = src
                r["text"] = r.get("text", "")
                rows.append(r)
    return RealCorpus(
        name="sarcasm",
        rows=rows,
        text_col="text",
        label_cols=["class", "source"],
    )


def load_semeval_sarc(zpath: Path = DEFAULT_ZIP) -> RealCorpus:
    # zip ships "train.En copy.csv"
    with _open_zip_member(zpath, "train.En copy.csv") as f:
        raw = _read_csv_dict(f)
    rows = []
    for r in raw:
        text = r.get("tweet") or r.get("text") or ""
        label = r.get("sarcastic") or r.get("label") or ""
        rows.append(
            {
                "text": text,
                "label": label,
                "class": "sarc" if str(label).strip() in {"1", "sarc"} else "notsarc",
            }
        )
    return RealCorpus(
        name="semeval_sarc",
        rows=rows,
        text_col="text",
        label_cols=["label", "class"],
    )


def load_amazon_magazine(zpath: Path = DEFAULT_ZIP) -> RealCorpus:
    with _open_zip_member(zpath, "magazineSubscription_df.csv") as f:
        raw = _read_csv_dict(f)
    rows = []
    for r in raw:
        rows.append(
            {
                "text": r.get("text", ""),
                "title": r.get("title", ""),
                "rating": r.get("rating", ""),
                "asin": r.get("asin", ""),
                "user_id": r.get("user_id", ""),
                "verified_purchase": r.get("verified_purchase", ""),
            }
        )
    return RealCorpus(
        name="amazon_magazine",
        rows=rows,
        text_col="text",
        label_cols=["rating", "asin"],
    )


def load_amazon_gift_cards(zpath: Path = DEFAULT_ZIP, max_rows: Optional[int] = None) -> RealCorpus:
    rows: List[Dict[str, str]] = []
    with _open_zip_member(zpath, "Gift_Cards.jsonl") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            obj = json.loads(line)
            rows.append(
                {
                    "text": str(obj.get("text", "")),
                    "title": str(obj.get("title", "")),
                    "rating": str(obj.get("rating", "")),
                    "asin": str(obj.get("asin", "")),
                    "user_id": str(obj.get("user_id", "")),
                    "verified_purchase": str(obj.get("verified_purchase", "")),
                }
            )
    return RealCorpus(
        name="amazon_gift_cards",
        rows=rows,
        text_col="text",
        label_cols=["rating", "asin"],
    )


_LOADERS = {
    "equity_evaluation": load_equity_evaluation,
    "biasfinder": load_biasfinder,
    "sarcasm": load_sarcasm,
    "semeval_sarc": load_semeval_sarc,
    "amazon_magazine": load_amazon_magazine,
    "amazon_gift_cards": load_amazon_gift_cards,
}


def load_corpus(name: str, zpath: Path | str = DEFAULT_ZIP, **kwargs) -> RealCorpus:
    if name not in _LOADERS:
        raise KeyError(f"unknown corpus {name}; choose from {list_corpora()}")
    return _LOADERS[name](Path(zpath), **kwargs)
