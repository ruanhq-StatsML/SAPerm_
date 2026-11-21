"""Shared checkpoint helpers: save-as-you-go + stackable Monte Carlo.

Rules
-----
- Append every finished (dataset, combo, scenario, method, rep) row immediately.
- Resume skips keys already present (first write wins).
- Raising ``n_rep`` only fills *new* rep indices; it does not redo finished cells.
- Deduplicate before summarizing so accidental re-appends do not inflate power.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

KEY_COLS = ("dataset", "combo", "scenario", "method", "rep")


def done_key(dataset: str, combo: str, scenario: str, method: str, rep: int) -> str:
    return f"{dataset}|{combo}|{scenario}|{method}|{rep}"


def read_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    if "scenario" in df.columns:
        df["scenario"] = df["scenario"].replace({"null": "no_shift", "nan": "no_shift"})
    if "rep" in df.columns:
        df["rep"] = pd.to_numeric(df["rep"], errors="coerce").astype("Int64")
    if "pvalue" in df.columns:
        df["pvalue"] = pd.to_numeric(df["pvalue"], errors="coerce")
    return df


def dedupe_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the first row for each (dataset, combo, scenario, method, rep)."""
    if df.empty:
        return df
    cols = [c for c in KEY_COLS if c in df.columns]
    if len(cols) < len(KEY_COLS):
        return df
    return df.drop_duplicates(subset=list(KEY_COLS), keep="first").reset_index(drop=True)


def compact_raw_csv(path: Path) -> Tuple[int, int]:
    """Rewrite ``power_raw.csv`` deduped in place. Returns (before, after)."""
    if not path.exists():
        return 0, 0
    raw = read_raw(path)
    before = len(raw)
    clean = dedupe_raw(raw)
    clean.to_csv(path, index=False)
    return before, len(clean)


def load_done(path: Path) -> Tuple[pd.DataFrame, Set[str]]:
    df = dedupe_raw(read_raw(path))
    if df.empty:
        return df, set()
    done = {
        done_key(str(r.dataset), str(r.combo), str(r.scenario), str(r.method), int(r.rep))
        for r in df.itertuples()
    }
    return df, done


def append_rows(path: Path, rows: List[dict]) -> None:
    """Append rows with an exclusive flock so parallel jobs can share one CSV."""
    if not rows:
        return
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    chunk = pd.DataFrame(rows)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            write_header = not path.exists() or path.stat().st_size == 0
            chunk.to_csv(path, mode="a", header=write_header, index=False)
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def rep_methods_done(
    done: Set[str],
    *,
    dataset: str,
    combo: str,
    scenario: str,
    methods: Iterable[str],
    rep: int,
) -> bool:
    """True iff every method in ``methods`` already has a saved row for this rep."""
    return all(done_key(dataset, combo, scenario, m, rep) in done for m in methods)


def summarize_power(
    raw: pd.DataFrame,
    *,
    alpha: float,
    out_dir: Path,
    title: str,
    highlight: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    raw = dedupe_raw(raw)
    if raw.empty:
        return raw
    summary = (
        raw.groupby(["dataset", "combo", "scenario", "method"], as_index=False)
        .agg(
            power=("pvalue", lambda s: float(np.mean(np.asarray(s, float) < alpha))),
            mean_p=("pvalue", "mean"),
            n_rep=("rep", "nunique"),
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "power_summary.csv", index=False)

    hi = set(highlight or ())
    lines = [f"# {title}", "", f"_rows={len(raw)} after dedupe; α={alpha}_", ""]
    for (ds, combo, scen), sub in summary.groupby(["dataset", "combo", "scenario"]):
        lines.append(f"## `{ds}` / `{combo}` / `{scen}`")
        lines.append("")
        lines.append("| method | power | mean p | n_rep |")
        lines.append("|---|---:|---:|---:|")
        for r in sub.sort_values("power", ascending=False).itertuples():
            name = f"**{r.method}**" if r.method in hi else r.method
            lines.append(f"| {name} | {r.power:.3f} | {r.mean_p:.3f} | {int(r.n_rep)} |")
        lines.append("")
    (out_dir / "POWER_BOARD.md").write_text("\n".join(lines))
    return summary
