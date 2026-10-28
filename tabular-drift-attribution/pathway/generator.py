"""Class-conditional TabDDPM, with RePaint imputation of one feature block.

The diffusion class is `GaussianMultinomialDiffusion` from
yandex-research/tab-ddpm (commit b476257). Numerical columns use their
Gaussian head. The reverse chain is their `gaussian_p_sample`. Known columns
are reset to the anchor noising process at every step, which is the RePaint
conditioning step. The label passed to the denoiser is drawn from the anchor
label distribution, not from the current day's outcome.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_CANDIDATES = [
    Path("/tmp/tab-ddpm"),
    Path(__file__).resolve().parents[1] / "third_party" / "tab-ddpm",
]
for candidate in _CANDIDATES:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion  # noqa: E402
from tab_ddpm.modules import MLPDiffusion  # noqa: E402


class TabDDPMRepaint:
    def __init__(
        self,
        n_features: int,
        steps: int = 1000,
        timesteps: int = 80,
        batch_size: int = 256,
        lr: float = 1e-3,
        seed: int = 0,
        device: str = "cpu",
    ):
        self.n_features = n_features
        self.steps = steps
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.device = torch.device(device)
        self.y_dist = None
        self.loss_last = None
        torch.manual_seed(seed)
        denoise = MLPDiffusion(
            d_in=n_features,
            num_classes=2,
            is_y_cond=True,
            rtdl_params={"d_layers": [256, 256], "dropout": 0.0},
            dim_t=64,
        )
        self.diffusion = GaussianMultinomialDiffusion(
            num_classes=np.array([0]),
            num_numerical_features=n_features,
            denoise_fn=denoise,
            num_timesteps=timesteps,
            gaussian_loss_type="mse",
            gaussian_parametrization="eps",
            multinomial_loss_type="vb_stochastic",
            parametrization="x0",
            scheduler="cosine",
            device=self.device,
        ).to(self.device).float()

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self.y_dist = np.array([1.0 - y.mean(), y.mean()], dtype=np.float64)
        self.y_dist = np.clip(self.y_dist, 1e-3, None)
        self.y_dist = self.y_dist / self.y_dist.sum()
        opt = torch.optim.AdamW(self.diffusion.parameters(), lr=self.lr, weight_decay=1e-4)
        n = X.shape[0]
        rng = np.random.default_rng(self.seed)
        self.diffusion.train()
        last = 0.0
        for step in range(self.steps):
            take = rng.choice(n, size=min(self.batch_size, n), replace=n < self.batch_size)
            xb = torch.tensor(X[take], device=self.device)
            yb = torch.tensor(y[take], device=self.device)
            opt.zero_grad()
            loss_multi, loss_gauss = self.diffusion.mixed_loss(xb, {"y": yb})
            loss = loss_multi + loss_gauss
            loss.backward()
            opt.step()
            frac = step / max(self.steps - 1, 1)
            for group in opt.param_groups:
                group["lr"] = self.lr * (1.0 - frac)
            last = float(loss.detach())
            if (step + 1) % 250 == 0 or step == 0:
                print(f"  tabddpm step {step + 1}/{self.steps} loss {last:.4f}", flush=True)
        self.loss_last = last
        self.diffusion.eval()
        return self

    def _draw_labels(self, n: int) -> torch.Tensor:
        probs = torch.tensor(self.y_dist, dtype=torch.float32, device=self.device)
        return torch.multinomial(probs, num_samples=n, replacement=True)

    @torch.no_grad()
    def sample(self, n: int) -> np.ndarray:
        """Unconditional class-mixture sample, using their reverse chain."""
        self.diffusion.eval()
        b = n
        device = self.device
        z = torch.randn((b, self.n_features), device=device, dtype=torch.float32)
        y = self._draw_labels(b)
        for i in reversed(range(self.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            model_out = self.diffusion._denoise_fn(z, t, y=y)
            z = self.diffusion.gaussian_p_sample(
                model_out[:, : self.n_features], z, t, clip_denoised=False,
            )["sample"].float()
        return z.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def impute(self, X: np.ndarray, target_cols: list[int], batch_size: int = 512) -> np.ndarray:
        """Replace `target_cols` by the anchor conditional given the other columns."""
        self.diffusion.eval()
        X = np.asarray(X, dtype=np.float32)
        n, d = X.shape
        known = np.ones(d, dtype=bool)
        known[list(target_cols)] = False
        known_t = torch.tensor(known, device=self.device)
        out = np.zeros((n, len(target_cols)), dtype=np.float32)
        x_all = torch.tensor(X, device=self.device)
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            x0 = x_all[start:stop]
            b = x0.shape[0]
            y = self._draw_labels(b)
            t_last = torch.full((b,), self.timesteps - 1, device=self.device, dtype=torch.long)
            z = torch.randn(x0.shape, device=self.device, dtype=torch.float32)
            z[:, known_t] = self.diffusion.gaussian_q_sample(x0, t_last).float()[:, known_t]
            for i in reversed(range(self.timesteps)):
                t = torch.full((b,), i, device=self.device, dtype=torch.long)
                model_out = self.diffusion._denoise_fn(z, t, y=y)
                z = self.diffusion.gaussian_p_sample(
                    model_out[:, : self.n_features], z, t, clip_denoised=False,
                )["sample"].float()
                if i > 0:
                    t_prev = torch.full((b,), i - 1, device=self.device, dtype=torch.long)
                    known_noised = self.diffusion.gaussian_q_sample(x0, t_prev).float()
                    z[:, known_t] = known_noised[:, known_t]
                else:
                    z[:, known_t] = x0[:, known_t]
            out[start:stop] = z[:, list(target_cols)].cpu().numpy()
        return out
