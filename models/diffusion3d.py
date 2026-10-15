"""Latent DDPM over a frozen VAR tokenizer's bottleneck -- a generative *shape
prior* for GT-free error LOCALIZATION (diffusion workstream, Part 2, Path A).

Motivation (advisor direction, 2026-10)
---------------------------------------
Part 1 established the voxel-level localization metric and ranked the signals the
EXISTING models already produce. The image-free ``recon_diff/VAR`` localizer
(``|broken - decode(quantize(encode(broken)))|``) was the weakest (voxel-AUROC
0.846) and, worse, *degraded* on the hard Dice bands: a single quantizer pass
snaps an atypical latent to the nearest codes but cannot reason about how
*unlikely* the whole configuration is. The right lever here is not a bigger
decoder but a proper likelihood model of the clean-shape manifold.

This module trains a small DDPM in the VAR tokenizer's **quantized** 4^3x64
latent (the exact tensor the frozen VAR decoder was trained on, so decoding a
denoised latent stays in-distribution). It is used AnoDDPM-style: a broken
mask's latent is partially forward-noised to a middle timestep and then reverse-
denoised toward the learned clean manifold; decoding gives a "what this shape
should look like" reconstruction, and ``|broken - decoded|`` is the anomaly map.
The success target is explicit and honest: beat ``recon_diff/VAR``'s 0.846,
especially on the hard bands. This is a reconstruction *localizer*, not a
prettier-mask generator -- every decision is gated on the localization metric.

Design choices
--------------
* **Latent, not pixel space.** With only 222 clean liver masks, a pixel-space
  3D DDPM overfits; the 4^3 latent is data-efficient and reuses the frozen
  encoder/decoder (fully additive -- nothing here edits a protected file).
* **No spatial up/down-sampling.** A 4^3 grid is already tiny, so the denoiser
  is a stack of time-conditioned 3D residual blocks at constant resolution.
* **Per-channel latent normalisation.** Latent statistics are stored as buffers
  (set once from the training set) so the epsilon-prediction target is unit-scale
  and the reverse process is stable.

Public API mirrors the project's other models enough to be loaded and scored by
additive eval code: ``training_loss`` (train), ``partial_denoise`` (localize),
``sample`` (sanity generation), plus ``normalize``/``denormalize`` helpers and a
``build_from_config`` factory. Nothing imports or edits ``models/vae3d_vq.py`` or
``models/vae3d_var.py``; the frozen VAR is passed in by the caller.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# --------------------------------------------------------------------------
# time embedding + denoiser network
# --------------------------------------------------------------------------
def sinusoidal_time_embedding(t: Tensor, dim: int) -> Tensor:
    """Standard transformer/DDPM sinusoidal embedding of integer timesteps.

    ``t`` is a 1D long tensor of timesteps; returns ``(len(t), dim)``.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:  # odd dim -> pad one zero column
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock3D(nn.Module):
    """GroupNorm-SiLU-Conv residual block with an additive time embedding.

    Constant spatial resolution (the 4^3 latent is too small to down-sample).
    The time embedding is projected per-block and broadcast over the grid.
    """

    def __init__(self, channels: int, time_dim: int, groups: int = 8) -> None:
        super().__init__()
        g = math.gcd(groups, channels) or 1
        self.norm1 = nn.GroupNorm(g, channels)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, channels)
        self.norm2 = nn.GroupNorm(g, channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class LatentDenoiser3D(nn.Module):
    """Epsilon-prediction network for the 4^3xC latent (no resolution change)."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_res_blocks: int = 3,
        time_dim: int = 128,
    ) -> None:
        super().__init__()
        self.time_dim = int(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )
        self.in_conv = nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [ResBlock3D(hidden_channels, self.time_dim) for _ in range(int(num_res_blocks))]
        )
        self.out_norm = nn.GroupNorm(math.gcd(8, hidden_channels) or 1, hidden_channels)
        self.out_conv = nn.Conv3d(hidden_channels, in_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, z_t: Tensor, t: Tensor) -> Tensor:
        t_emb = self.time_mlp(sinusoidal_time_embedding(t, self.time_dim))
        h = self.in_conv(z_t)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.out_conv(F.silu(self.out_norm(h)))


# --------------------------------------------------------------------------
# DDPM wrapper (schedule buffers + training loss + partial denoise)
# --------------------------------------------------------------------------
class LatentDDPM3D(nn.Module):
    """DDPM over a C-channel, cubic ``latent_side^3`` latent grid.

    Operates on *normalised* latents internally; callers pass/receive raw latents
    and normalisation is applied at the boundary. The frozen VAR encode/decode are
    the caller's responsibility (kept out of here so this stays VAR-agnostic).
    """

    def __init__(
        self,
        latent_channels: int,
        latent_side: int,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        hidden_channels: int = 128,
        num_res_blocks: int = 3,
        time_dim: int = 128,
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.latent_side = int(latent_side)
        self.timesteps = int(timesteps)
        self.net = LatentDenoiser3D(
            self.latent_channels, hidden_channels, num_res_blocks, time_dim
        )

        betas = torch.linspace(beta_start, beta_end, self.timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("posterior_variance", posterior_variance)

        # Per-channel latent normalisation (set once from the training set).
        shape = (1, self.latent_channels, 1, 1, 1)
        self.register_buffer("latent_mean", torch.zeros(shape))
        self.register_buffer("latent_std", torch.ones(shape))

    # -- normalisation ------------------------------------------------------
    def set_latent_stats(self, mean: Tensor, std: Tensor) -> None:
        self.latent_mean.copy_(mean.view_as(self.latent_mean))
        self.latent_std.copy_(std.clamp_min(1e-6).view_as(self.latent_std))

    def normalize(self, z: Tensor) -> Tensor:
        return (z - self.latent_mean) / self.latent_std

    def denormalize(self, z: Tensor) -> Tensor:
        return z * self.latent_std + self.latent_mean

    # -- forward (noising) process -----------------------------------------
    def q_sample(self, z0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        a = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        b = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        return a * z0 + b * noise

    def training_loss(self, z0_raw: Tensor) -> Tensor:
        """Epsilon-prediction MSE on a random timestep per sample. Input is a
        RAW latent; normalisation is applied here."""
        z0 = self.normalize(z0_raw)
        b = z0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=z0.device, dtype=torch.long)
        noise = torch.randn_like(z0)
        z_t = self.q_sample(z0, t, noise)
        eps = self.net(z_t, t)
        return F.mse_loss(eps, noise)

    # -- reverse (denoising) process ---------------------------------------
    @torch.no_grad()
    def p_sample_step(self, z_t: Tensor, t: Tensor) -> Tensor:
        """One DDPM posterior step ``p(z_{t-1} | z_t)`` (normalised space)."""
        beta = self.betas[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_recip = self.sqrt_recip_alphas[t].view(-1, 1, 1, 1, 1)
        eps = self.net(z_t, t)
        mean = sqrt_recip * (z_t - beta / sqrt_one_minus * eps)
        var = self.posterior_variance[t].view(-1, 1, 1, 1, 1)
        noise = torch.randn_like(z_t)
        nonzero = (t > 0).float().view(-1, 1, 1, 1, 1)
        return mean + nonzero * torch.sqrt(var) * noise

    @torch.no_grad()
    def partial_denoise(self, z0_raw: Tensor, t_start: int) -> Tensor:
        """AnoDDPM localize step: forward-noise a RAW latent to ``t_start`` then
        reverse-denoise to 0. Returns a RAW (denormalised) latent on the clean
        manifold. A middle ``t_start`` erases the injected error while keeping the
        overall shape (full ``t_start`` would resample an unrelated shape)."""
        z0 = self.normalize(z0_raw)
        t_start = int(max(1, min(t_start, self.timesteps)))
        t = torch.full((z0.shape[0],), t_start - 1, device=z0.device, dtype=torch.long)
        z_t = self.q_sample(z0, t, torch.randn_like(z0))
        for i in reversed(range(t_start)):
            tt = torch.full((z0.shape[0],), i, device=z0.device, dtype=torch.long)
            z_t = self.p_sample_step(z_t, tt)
        return self.denormalize(z_t)

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> Tensor:
        """Unconditional generation from pure noise (sanity / preview). Returns a
        RAW latent that the caller can decode with the frozen VAR."""
        z_t = torch.randn(
            n, self.latent_channels, self.latent_side, self.latent_side, self.latent_side,
            device=device,
        )
        for i in reversed(range(self.timesteps)):
            tt = torch.full((n,), i, device=device, dtype=torch.long)
            z_t = self.p_sample_step(z_t, tt)
        return self.denormalize(z_t)


# --------------------------------------------------------------------------
def build_from_config(config: dict[str, Any], latent_channels: int, latent_side: int) -> LatentDDPM3D:
    """Construct a :class:`LatentDDPM3D` from a config's ``diffusion3d`` block.

    ``latent_channels``/``latent_side`` come from the frozen VAR (``embedding_dim``
    and bottleneck side) so the two models always agree on the latent shape.
    """
    d = config["diffusion3d"]
    return LatentDDPM3D(
        latent_channels=int(latent_channels),
        latent_side=int(latent_side),
        timesteps=int(d.get("timesteps", 1000)),
        beta_start=float(d.get("beta_start", 1e-4)),
        beta_end=float(d.get("beta_end", 2e-2)),
        hidden_channels=int(d.get("hidden_channels", 128)),
        num_res_blocks=int(d.get("num_res_blocks", 3)),
        time_dim=int(d.get("time_dim", 128)),
    )
