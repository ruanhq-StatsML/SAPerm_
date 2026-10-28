"""Real covariate tables used as the shared P(X) of the 10-site benchmark.

ACS Income and ACS Public Coverage are two WhyShift tasks built on the 2018
ACS 1-year person file for California. Covariates are the official folktables
feature lists. The income label is the usual $50k indicator. The public-coverage
label is public health-insurance coverage. Income components are not added on
top of the income task, so the base outcome is not a deterministic function of
an included total.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parent / "cache"

TASKS = {
    "acs_income": {
        "roles": {"x1": "WKHP", "x2": "SCHL", "x3": "AGEP"},
        "description": "WhyShift ACS Income, California 2018, official 10 covariates",
    },
    "acs_pubcov": {
        "roles": {"x1": "AGEP", "x2": "SCHL", "x3": "PINCP"},
        "description": "WhyShift ACS Public Coverage, California 2018, official 19 covariates",
    },
}


def _load_acs_frame(task: str):
    from folktables import ACSDataSource, ACSIncome, ACSPublicCoverage

    problem = {"acs_income": ACSIncome, "acs_pubcov": ACSPublicCoverage}[task]
    source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    raw = source.get_data(states=["CA"], download=True)
    X, y, _ = problem.df_to_pandas(raw)
    frame = X.copy()
    frame["__y__"] = y.iloc[:, 0].astype(int).to_numpy()
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    return frame


def load_task(task: str) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    if task not in TASKS:
        raise KeyError(task)
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{task}.npz"
    meta = TASKS[task]
    if path.exists():
        blob = np.load(path, allow_pickle=True)
        names = [str(c) for c in blob["feature_names"].tolist()]
        return blob["X"], blob["y"].astype(int), names, meta
    frame = _load_acs_frame(task)
    names = [c for c in frame.columns if c != "__y__"]
    X = frame[names].to_numpy(dtype=float)
    y = frame["__y__"].to_numpy(dtype=int)
    np.savez_compressed(path, X=X, y=y, feature_names=np.array(names, dtype=object))
    return X, y, names, meta
