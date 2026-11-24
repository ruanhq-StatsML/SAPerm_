"""Modular predictor with separate adapter, fusion, and head parameter groups.

An adapter update here is a linear map on a frozen embedding block, not
pretrained-encoder fine-tuning. Source replay is off for every arm.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from experiments.cfperm_recsys_adaptation.dgp import BLOCK, BLOCK_SLICE

ARM_GROUPS = {
    "no_update": (),
    "head_only": ("head",),
    "text_only": ("text_adapter",),
    "audio_only": ("audio_adapter",),
    "visual_only": ("visual_adapter",),
    "fusion_only": ("fusion",),
    "pair_TA": ("text_adapter", "audio_adapter"),
    "pair_TV": ("text_adapter", "visual_adapter"),
    "pair_AV": ("audio_adapter", "visual_adapter"),
    "full": ("text_adapter", "audio_adapter", "visual_adapter", "fusion", "head"),
}


class ModularPredictor(nn.Module):
    """Block adapters, a bilinear fusion, and a logit head.

    Products of adapter outputs let the fusion represent the planted T0*V0 term.
    The product itself has no parameters; those sit in ``fusion``.
    """

    def __init__(self, hidden: int = 8):
        super().__init__()
        self.hidden = int(hidden)
        self.text_adapter = nn.Linear(BLOCK, hidden)
        self.audio_adapter = nn.Linear(BLOCK, hidden)
        self.visual_adapter = nn.Linear(BLOCK, hidden)
        self.fusion = nn.Linear(hidden * 6, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.text_adapter(x[:, BLOCK_SLICE["text"]])
        a = self.audio_adapter(x[:, BLOCK_SLICE["audio"]])
        v = self.visual_adapter(x[:, BLOCK_SLICE["visual"]])
        fused = torch.cat([t, a, v, t * v, t * a, a * v], dim=1)
        hidden = torch.tanh(self.fusion(fused))
        return self.head(hidden).squeeze(-1)


def _tensor(x: np.ndarray, y: np.ndarray | None = None):
    xt = torch.as_tensor(np.asarray(x, dtype=np.float32))
    if y is None:
        return xt
    yt = torch.as_tensor(np.asarray(y, dtype=np.float32))
    return xt, yt


def predict_proba(model: ModularPredictor, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(_tensor(x))
        return torch.sigmoid(logits).cpu().numpy().astype(float)


def brier(model: ModularPredictor, x: np.ndarray, y: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    prob = predict_proba(model, x)
    return float(np.mean((prob - np.asarray(y, dtype=float)) ** 2))


def _set_trainable(model: ModularPredictor, groups: tuple[str, ...]) -> None:
    for name, param in model.named_parameters():
        prefix = name.split(".", 1)[0]
        param.requires_grad = prefix in groups


def _frozen_snapshot(model: ModularPredictor) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if not param.requires_grad
    }


def _frozen_unchanged(model: ModularPredictor, before: dict[str, torch.Tensor]) -> bool:
    for name, param in model.named_parameters():
        if name not in before:
            continue
        if not torch.equal(param.detach().cpu(), before[name]):
            return False
    return True


def trainable_parameter_count(model: ModularPredictor) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


@dataclass
class TrainResult:
    wall_seconds: float
    n_steps: int
    trainable_params: int
    update_cost: float
    frozen_unchanged: bool
    n_labels: int


def fit_model(
    model: ModularPredictor,
    x: np.ndarray,
    y: np.ndarray,
    *,
    groups: tuple[str, ...],
    epochs: int,
    lr: float,
    weight_decay: float = 1e-4,
) -> TrainResult:
    """Full-batch Adam on the named groups. Frozen tensors are checked afterwards."""
    _set_trainable(model, groups)
    before = _frozen_snapshot(model)
    n_labels = int(len(y))
    n_params = trainable_parameter_count(model)
    if n_labels == 0 or n_params == 0 or epochs <= 0:
        model.eval()
        return TrainResult(0.0, 0, 0, 0.0, True, 0)

    xt, yt = _tensor(x, y)
    opt = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        weight_decay=weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    start = time.perf_counter()
    for _ in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()
    wall = time.perf_counter() - start
    model.eval()
    unchanged = _frozen_unchanged(model, before)
    steps = int(epochs)
    return TrainResult(
        wall_seconds=float(wall),
        n_steps=steps,
        trainable_params=n_params,
        update_cost=float(steps * n_params),
        frozen_unchanged=bool(unchanged),
        n_labels=n_labels,
    )


def clone_model(model: ModularPredictor) -> ModularPredictor:
    copied = copy.deepcopy(model)
    copied.eval()
    return copied


def initial_source_model(x: np.ndarray, y: np.ndarray, *, hidden: int, epochs: int, lr: float, seed: int) -> ModularPredictor:
    torch.manual_seed(int(seed))
    model = ModularPredictor(hidden=hidden)
    fit_model(
        model,
        x,
        y,
        groups=ARM_GROUPS["full"],
        epochs=epochs,
        lr=lr,
    )
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model
