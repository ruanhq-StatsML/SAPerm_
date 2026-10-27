"""Conditional DDPM, quantile-regression MVP, and a linear Gaussian baseline.

The diffused variable is the scalar purchase index. A length-1 series has no
spatial U-Net to run, so the denoiser is an MLP with adaptive layer-norm
(the MLP analogue of AdaGN). Multi-scale structure lives in the conditioner:
hourly, daily, and weekly lag buckets reach the denoiser only after a
softmax gate. That gate is identified, because a closed gate drops the
bucket. Independent sigmoids are not identified (a gate and a downstream
weight trade off), so the gate is a softmax. The training objective is the
standard noise-prediction loss with 10% dropout on the lag block.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from ddpm_cf_attribution.protocol import (
    DIFFUSION_SCHEDULE,
    DIFFUSION_STEPS,
    HISTORY_DROPOUT,
    LAGS,
    SCALE_SLICES,
)


torch.set_num_threads(4)


class Standardizer:
    def __init__(self) -> None:
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.y_mean: float = 0.0
        self.y_std: float = 1.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Standardizer":
        self.x_mean = x.mean(axis=0)
        self.x_std = np.clip(x.std(axis=0), 1e-3, None)
        self.y_mean = float(y.mean())
        self.y_std = float(max(y.std(), 1e-3))
        return self

    def transform_x(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_mean) / self.x_std

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_mean) / self.y_std

    def inverse_y(self, y: np.ndarray) -> np.ndarray:
        return y * self.y_std + self.y_mean


def _one_hot(site: np.ndarray, n_sites: int = 10) -> np.ndarray:
    eye = np.eye(n_sites)
    return eye[np.asarray(site, dtype=int)]


def design_matrix(x: np.ndarray, site: np.ndarray, interact_lags: bool = False, lag_dim: int = LAGS) -> np.ndarray:
    """Site-interacted lags are the linear conditional that matches the DGP.

    An additive site intercept cannot represent ten different kernels, so the
    linear baseline multiplies the lag block by the site indicator.
    """
    oh = _one_hot(site)
    if not interact_lags:
        return np.concatenate([x, oh], axis=1)
    lags = x[:, :lag_dim]
    rest = x[:, lag_dim:]
    interacted = (lags[:, :, None] * oh[:, None, :]).reshape(len(x), lag_dim * oh.shape[1])
    return np.concatenate([interacted, rest, oh], axis=1)


class QuantileMVP:
    """Gradient-boosting quantile regression. Samples are drawn from a
    normal approximation to the 0.2/0.5/0.8 conditional quantiles."""

    def __init__(self) -> None:
        self.models: list[HistGradientBoostingRegressor] = []
        self.quantiles = (0.2, 0.5, 0.8)
        self._z80 = 0.8416212335729143

    def _frame(self, x: np.ndarray, site: np.ndarray) -> np.ndarray:
        # Site id is categorical so a tree can split kernels by site.
        return np.concatenate([x, np.asarray(site, dtype=np.int64).reshape(-1, 1)], axis=1)

    def fit(self, x: np.ndarray, y: np.ndarray, site: np.ndarray) -> "QuantileMVP":
        d = self._frame(x, site)
        cat = [d.shape[1] - 1]
        self.models = []
        for q in self.quantiles:
            model = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=q,
                max_iter=240,
                learning_rate=0.06,
                max_depth=8,
                min_samples_leaf=20,
                categorical_features=cat,
                random_state=0,
            )
            model.fit(d, y)
            self.models.append(model)
        return self

    def predict_moments(self, x: np.ndarray, site: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = self._frame(x, site)
        qs = [m.predict(d) for m in self.models]
        stacked = np.sort(np.vstack(qs), axis=0)
        q20, q50, q80 = stacked[0], stacked[1], stacked[2]
        std = np.maximum(q80 - q20, 1e-3) / (2.0 * self._z80)
        return q50, std

    def sample(self, x: np.ndarray, site: np.ndarray, n_samples: int, seed: int = 0) -> np.ndarray:
        mean, std = self.predict_moments(x, site)
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, 1.0, size=(n_samples, len(mean)))
        return (mean[None, :] + std[None, :] * noise).T


class LinearGaussian:
    """Homoskedastic ridge with site-specific lag coefficients.

    Correctly specified for the linear family up to heteroskedasticity and
    the unobserved confounder. Lag coefficients are not pooled across sites.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        self.model = Ridge(alpha=alpha)
        self.resid_std: float = 1.0

    def fit(self, x: np.ndarray, y: np.ndarray, site: np.ndarray) -> "LinearGaussian":
        d = design_matrix(x, site, interact_lags=True)
        self.model.fit(d, y)
        resid = y - self.model.predict(d)
        self.resid_std = float(max(resid.std(), 1e-3))
        return self

    def sample(self, x: np.ndarray, site: np.ndarray, n_samples: int, seed: int = 0) -> np.ndarray:
        mean = self.model.predict(design_matrix(x, site, interact_lags=True))
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, self.resid_std, size=(n_samples, len(mean)))
        return (mean[None, :] + noise).T


def cosine_betas(n_steps: int, s: float = 0.008) -> np.ndarray:
    steps = np.arange(n_steps + 1, dtype=np.float64)
    f = np.cos((steps / n_steps + s) / (1.0 + s) * np.pi / 2.0) ** 2
    alpha_bar = f / f[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return np.clip(betas, 1e-5, 0.999)


def _sinusoidal(t: torch.Tensor, dim: int = 16) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-np.log(10000.0) * torch.arange(half, device=t.device) / half)
    ang = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)


class Denoiser(nn.Module):
    def __init__(self, n_sites: int = 10, hidden: int = 128, lag_dim: int = LAGS) -> None:
        super().__init__()
        self.lag_dim = lag_dim
        h0, h1 = SCALE_SLICES["h"]
        d0, d1 = SCALE_SLICES["d"]
        w0, w1 = SCALE_SLICES["w"]
        self.slices = (h0, h1, d0, d1, w0, w1)
        self.site_emb = nn.Embedding(n_sites, 16)
        gate_in = lag_dim + 16
        self.gate = nn.Sequential(
            nn.Linear(gate_in, 32),
            nn.SiLU(),
            nn.Linear(32, 3),
        )
        # Gated raw lags keep each lag visible to the site-specific MLP.
        # 14 lags + X(3) + overlap(1) + dist(3) + site(16) + gate(3).
        cond_dim = lag_dim + 3 + 1 + 3 + 16 + 3
        self.cond_dim = cond_dim
        in_dim = 1 + 16 + cond_dim
        self.fc1 = nn.Linear(in_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.film1 = nn.Linear(cond_dim, hidden * 2)
        self.fc2 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.film2 = nn.Linear(cond_dim, hidden * 2)
        self.fc3 = nn.Linear(hidden, 1)
        nn.init.zeros_(self.film1.weight)
        nn.init.zeros_(self.film1.bias)
        nn.init.zeros_(self.film2.weight)
        nn.init.zeros_(self.film2.bias)

    def scales(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        lags = feats[:, : self.lag_dim]
        rest = feats[:, self.lag_dim:]
        h0, h1, d0, d1, w0, w1 = self.slices
        return lags[:, h0:h1], lags[:, d0:d1], lags[:, w0:w1], rest

    def forward(self, y: torch.Tensor, t: torch.Tensor, feats: torch.Tensor, site: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hh, hd, hw, rest = self.scales(feats)
        emb = self.site_emb(site)
        logits = self.gate(torch.cat([feats[:, : self.lag_dim], emb], dim=-1))
        g = torch.softmax(logits, dim=-1)
        gated = torch.cat([g[:, 0:1] * hh, g[:, 1:2] * hd, g[:, 2:3] * hw], dim=-1)
        cond = torch.cat([gated, rest, emb, g], dim=-1)
        temb = _sinusoidal(t, 16)
        h = self.fc1(torch.cat([y, temb, cond], dim=-1))
        scale, shift = self.film1(cond).chunk(2, dim=-1)
        h = self.ln1(h) * (1.0 + scale) + shift
        h = F.silu(h)
        h = self.fc2(h)
        scale, shift = self.film2(cond).chunk(2, dim=-1)
        h = self.ln2(h) * (1.0 + scale) + shift
        h = F.silu(h)
        eps = self.fc3(h)
        return eps, g


class ConditionalDDPM:
    def __init__(
        self,
        n_steps: int = DIFFUSION_STEPS,
        hidden: int = 128,
        lr: float = 2e-3,
        history_dropout: float = HISTORY_DROPOUT,
        schedule: str = DIFFUSION_SCHEDULE,
    ) -> None:
        self.n_steps = n_steps
        self.hidden = hidden
        self.lr = lr
        self.history_dropout = history_dropout
        self.schedule = schedule
        self.std = Standardizer()
        self.net: Denoiser | None = None
        self.device = torch.device("cpu")
        betas = cosine_betas(n_steps) if schedule == "cosine" else np.linspace(1e-4, 0.2, n_steps)
        alphas = 1.0 - betas
        alphabar = np.cumprod(alphas)
        self.betas = torch.tensor(betas, dtype=torch.float32)
        self.alphas = torch.tensor(alphas, dtype=torch.float32)
        self.alphabar = torch.tensor(alphabar, dtype=torch.float32)
        self.train_loss: list[float] = []
        self.val_loss: list[float] = []

    def _noise_loss(self, y0: torch.Tensor, feats: torch.Tensor, site: torch.Tensor) -> torch.Tensor:
        b = y0.shape[0]
        t = torch.randint(0, self.n_steps, (b,), device=self.device)
        eps = torch.randn_like(y0)
        ab = self.alphabar[t].unsqueeze(1)
        y_t = torch.sqrt(ab) * y0 + torch.sqrt(1.0 - ab) * eps
        if self.net.training and self.history_dropout > 0:
            drop = torch.rand(b, device=self.device) < self.history_dropout
            feats = feats.clone()
            feats[drop, :LAGS] = 0.0
        eps_hat, _ = self.net(y_t, t, feats, site)
        return F.mse_loss(eps_hat, eps)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        site: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        site_val: np.ndarray,
        epochs: int = 40,
        batch_size: int = 512,
        seed: int = 0,
    ) -> "ConditionalDDPM":
        self.std.fit(x, y)
        xt = self.std.transform_x(x).astype(np.float32)
        yt = self.std.transform_y(y).astype(np.float32)
        xv = self.std.transform_x(x_val).astype(np.float32)
        yv = self.std.transform_y(y_val).astype(np.float32)
        self.net = Denoiser(hidden=self.hidden).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=self.lr * 0.1)
        rng = np.random.default_rng(seed)
        n = len(yt)
        best_state = None
        best_val = float("inf")
        yv_t = torch.tensor(yv[:, None])
        xv_t = torch.tensor(xv)
        sv_t = torch.tensor(site_val.astype(np.int64))
        for epoch in range(epochs):
            self.net.train()
            perm = rng.permutation(n)
            losses = []
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                yb = torch.tensor(yt[idx, None])
                xb = torch.tensor(xt[idx])
                sb = torch.tensor(site[idx].astype(np.int64))
                loss = self._noise_loss(yb, xb, sb)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.detach()))
            self.net.eval()
            with torch.no_grad():
                # val loss on a subset to keep the epoch cheap
                take = min(4096, len(yv))
                vloss = float(self._noise_loss(yv_t[:take], xv_t[:take], sv_t[:take]))
            sched.step()
            self.train_loss.append(float(np.mean(losses)))
            self.val_loss.append(vloss)
            if vloss < best_val:
                best_val = vloss
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  epoch {epoch + 1:03d}  train {self.train_loss[-1]:.4f}  val {vloss:.4f}")
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    @torch.no_grad()
    def _sample_standardized(self, feats: np.ndarray, site: np.ndarray) -> np.ndarray:
        self.net.eval()
        n = feats.shape[0]
        y = torch.randn(n, 1)
        ft = torch.tensor(feats.astype(np.float32))
        st = torch.tensor(site.astype(np.int64))
        for ti in range(self.n_steps - 1, -1, -1):
            t = torch.full((n,), ti, dtype=torch.long)
            eps, _ = self.net(y, t, ft, st)
            beta = self.betas[ti]
            alpha = self.alphas[ti]
            ab = self.alphabar[ti]
            # Predict x0 and clip it. The last cosine step has a large beta;
            # an unconstrained noise update explodes when eps_hat is early in training.
            x0 = (y - torch.sqrt(1.0 - ab) * eps) / torch.sqrt(ab).clamp(min=1e-4)
            x0 = x0.clamp(-5.0, 5.0)
            if ti > 0:
                ab_prev = self.alphabar[ti - 1]
                coef_x0 = torch.sqrt(ab_prev) * beta / (1.0 - ab)
                coef_y = torch.sqrt(alpha) * (1.0 - ab_prev) / (1.0 - ab)
                mean = coef_x0 * x0 + coef_y * y
                var = beta * (1.0 - ab_prev) / (1.0 - ab)
                y = mean + torch.sqrt(var.clamp(min=1e-20)) * torch.randn_like(y)
            else:
                y = x0
        return y.squeeze(1).numpy()

    def sample(self, x: np.ndarray, site: np.ndarray, n_samples: int, seed: int = 0) -> np.ndarray:
        """Return an array of shape (n_cond, n_samples) on the original Y scale."""
        torch.manual_seed(seed)
        xs = self.std.transform_x(np.asarray(x, dtype=np.float64)).astype(np.float32)
        n_cond = xs.shape[0]
        x_rep = np.repeat(xs, n_samples, axis=0)
        s_rep = np.repeat(np.asarray(site, dtype=np.int64), n_samples, axis=0)
        # chunk so a long path evaluation stays inside memory
        out = np.empty(n_cond * n_samples, dtype=np.float64)
        chunk = 8192
        for i in range(0, len(x_rep), chunk):
            draw = self._sample_standardized(x_rep[i:i + chunk], s_rep[i:i + chunk])
            out[i:i + chunk] = self.std.inverse_y(draw)
        return out.reshape(n_cond, n_samples)

    @torch.no_grad()
    def gates(self, x: np.ndarray, site: np.ndarray) -> np.ndarray:
        self.net.eval()
        xs = self.std.transform_x(np.asarray(x, dtype=np.float64)).astype(np.float32)
        y = torch.zeros(len(xs), 1)
        t = torch.zeros(len(xs), dtype=torch.long)
        _, g = self.net(y, t, torch.tensor(xs), torch.tensor(np.asarray(site, dtype=np.int64)))
        return g.numpy()
