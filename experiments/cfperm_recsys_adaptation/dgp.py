"""Synthetic three-block logistic benchmark from the adaptation spec.

Each block has eight independent standard-normal coordinates. Source logit:

    1.2*T0 + 1.0*A0 + 1.0*V0 + 0.6*T0*V0

Target scenarios change one mechanism at a time. Rows are independent, so a
row split is the exchangeability design.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BLOCK = 8
BLOCKS = ("text", "audio", "visual")
BLOCK_SLICE = {
    "text": slice(0, BLOCK),
    "audio": slice(BLOCK, 2 * BLOCK),
    "visual": slice(2 * BLOCK, 3 * BLOCK),
}
P = 3 * BLOCK
SCENARIOS = (
    "null",
    "benign_covariate",
    "modality_concept",
    "cross_modal",
    "localized",
    "mixed",
)
# Planted changing block, used only to score selection after the fact.
PLANTED_BLOCK = {
    "null": "",
    "benign_covariate": "visual",
    "modality_concept": "visual",
    "cross_modal": "text_visual",
    "localized": "audio",
    "mixed": "visual+text_visual",
}
SPLITS = (
    "source_train",
    "source_diagnostic",
    "target_diagnostic",
    "target_adaptation",
    "target_validation",
    "source_test",
    "target_test",
)


def logit_parts(text: np.ndarray, audio: np.ndarray, visual: np.ndarray, scenario: str, domain: int) -> np.ndarray:
    """Return the scenario logit. ``domain`` 0 is always the source mechanism."""
    t0 = text[:, 0]
    a0 = audio[:, 0]
    v0 = visual[:, 0]
    t1 = text[:, 1]
    if domain == 0 or scenario in ("null", "benign_covariate"):
        return 1.2 * t0 + 1.0 * a0 + 1.0 * v0 + 0.6 * t0 * v0
    if scenario == "modality_concept":
        return 1.2 * t0 + 1.0 * a0 - 1.5 * v0 + 0.6 * t0 * v0
    if scenario == "cross_modal":
        return 1.2 * t0 + 1.0 * a0 + 1.0 * v0 - 1.4 * t0 * v0
    if scenario == "localized":
        return 1.2 * t0 + 1.0 * a0 + 1.0 * v0 + 0.6 * t0 * v0 - 2.0 * a0 * (t1 > 0.8)
    if scenario == "mixed":
        return 1.2 * t0 + 1.0 * a0 + 1.0 * v0 - 1.4 * t0 * v0
    raise ValueError(f"unknown scenario: {scenario}")


def _shift_covariate(visual: np.ndarray, scenario: str, domain: int) -> np.ndarray:
    if domain == 1 and scenario in ("benign_covariate", "mixed"):
        out = visual.copy()
        out[:, 7] = out[:, 7] + 1.5
        return out
    return visual


@dataclass
class Split:
    name: str
    domain: int
    x: np.ndarray
    y: np.ndarray
    sample_id: np.ndarray

    @property
    def n(self) -> int:
        return int(len(self.y))

    @property
    def label_rate(self) -> float:
        return float(np.mean(self.y)) if len(self.y) else float("nan")


def _draw_split(n: int, scenario: str, domain: int, rng: np.random.Generator, id_start: int) -> Split:
    text = rng.normal(size=(n, BLOCK))
    audio = rng.normal(size=(n, BLOCK))
    visual = rng.normal(size=(n, BLOCK))
    visual = _shift_covariate(visual, scenario, domain)
    eta = logit_parts(text, audio, visual, scenario, domain)
    prob = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, prob).astype(float)
    x = np.concatenate([text, audio, visual], axis=1)
    sample_id = np.arange(id_start, id_start + n, dtype=int)
    name = "source" if domain == 0 else "target"
    return Split(name=name, domain=domain, x=x, y=y, sample_id=sample_id)


def generate_scenario(scenario: str, sizes: dict, seed: int) -> dict[str, Split]:
    """Draw independent splits. Scaler fitting is left to the caller."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")
    spec = {
        "source_train": ("source_train", 0),
        "source_diagnostic": ("source_diagnostic", 0),
        "target_diagnostic": ("target_diagnostic", 1),
        "target_adaptation": ("target_adaptation", 1),
        "target_validation": ("target_validation", 1),
        "source_test": ("source_test", 0),
        "target_test": ("target_test", 1),
    }
    out = {}
    cursor = 0
    for offset, name in enumerate(SPLITS):
        n = int(sizes[name])
        # Stable, scenario-specific stream. Python's hash() is process-salted.
        scenario_code = SCENARIOS.index(scenario) + 1
        rng = np.random.default_rng(seed * 100_003 + scenario_code * 100 + offset + 1)
        domain = spec[name][1]
        out[name] = _draw_split(n, scenario, domain, rng, cursor)
        out[name].name = name
        cursor += n
    return out


def fit_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd


def apply_scaler(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (x - mu) / sd


def scale_splits(splits: dict[str, Split]) -> dict[str, Split]:
    """Fit the scaler on source training rows and apply it to every split."""
    mu, sd = fit_scaler(splits["source_train"].x)
    scaled = {}
    for name, split in splits.items():
        scaled[name] = Split(
            name=split.name,
            domain=split.domain,
            x=apply_scaler(split.x, mu, sd),
            y=split.y.copy(),
            sample_id=split.sample_id.copy(),
        )
    return scaled


# Starting sizes from the spec. `--quick` keeps these sizes and a single label
# budget; it shortens training. The later grid adds seeds, permutations, and
# the budget sweep rather than a different sample size.
QUICK_SIZES = {
    "source_train": 2000,
    "source_diagnostic": 500,
    "target_diagnostic": 500,
    "target_adaptation": 500,
    "target_validation": 250,
    "source_test": 1000,
    "target_test": 1000,
}

FULL_SIZES = {
    "source_train": 2000,
    "source_diagnostic": 500,
    "target_diagnostic": 500,
    "target_adaptation": 500,
    "target_validation": 250,
    "source_test": 1000,
    "target_test": 1000,
}
