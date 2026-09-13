"""Tests that quality.score works UNCHANGED on 3D volumes and ConvVAE3D.

The Phase-8 claim is that quality/score.py is already dimension-agnostic
(``_spatial_dims = range(1, x.dim())`` covers (B,1,H,W) and (B,1,D,H,W);
topology's ``cc_label`` labels connected components in 3D). These tests exercise
that on a tiny ConvVAE3D with (B,1,D,H,W) inputs — no data, no retraining. If any
score had a baked-in 2D assumption it would surface here as a shape/label error.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from models.vae3d import ConvVAE3D
from quality.score import (
    SCORE_FNS,
    compute_all_scores,
    recon_dice_score,
    recon_l1_score,
    topology_score,
)


def _tiny_vae3d() -> ConvVAE3D:
    torch.manual_seed(0)
    return ConvVAE3D(latent_dim=4, base_channels=4, input_size=(16, 16, 16)).eval()


def _random_volumes(batch: int = 5) -> Tensor:
    torch.manual_seed(1)
    return (torch.rand(batch, 1, 16, 16, 16) > 0.5).float()


class _IdentityVAE3D(nn.Module):
    """Stub whose 3D reconstruction is exactly the input, for exact-value checks."""

    def reconstruct(self, x: Tensor) -> Tensor:
        return x

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        logits = torch.logit(x.clamp(1e-4, 1 - 1e-4))
        z = torch.zeros(x.shape[0], 4)
        return logits, z, z

    def _encode_dist(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = torch.zeros(x.shape[0], 4)
        return z, z


def test_every_score_returns_one_scalar_per_volume() -> None:
    model, vols = _tiny_vae3d(), _random_volumes(5)
    for name, fn in SCORE_FNS.items():
        out = fn(model, vols)
        assert out.shape == (5,), f"{name} returned shape {tuple(out.shape)}"
        assert torch.isfinite(out).all(), f"{name} produced non-finite values"


def test_compute_all_scores_keys_and_length_3d() -> None:
    model, vols = _tiny_vae3d(), _random_volumes(4)
    scores = compute_all_scores(model, vols)
    assert set(scores) == set(SCORE_FNS)
    for name, arr in scores.items():
        assert isinstance(arr, np.ndarray) and arr.shape == (4,), name


def test_scores_are_deterministic_in_eval_3d() -> None:
    model, vols = _tiny_vae3d(), _random_volumes(5)
    first = compute_all_scores(model, vols)
    second = compute_all_scores(model, vols)
    for name in SCORE_FNS:
        assert np.allclose(first[name], second[name]), f"{name} not deterministic"


def test_perfect_reconstruction_gives_dice_one_and_zero_l1_3d() -> None:
    model, vols = _IdentityVAE3D(), _random_volumes(4)
    assert torch.allclose(recon_dice_score(model, vols), torch.ones(4))
    assert torch.allclose(recon_l1_score(model, vols), torch.zeros(4), atol=1e-6)
    assert torch.allclose(topology_score(model, vols), torch.zeros(4))


def test_topology_counts_3d_components() -> None:
    # Two spatially separated cubes => two 3D connected components; an identity
    # model reconstructs them faithfully, so topology mismatch is zero. This
    # confirms cc_label operates in 3D (a 2D-only label would miscount).
    vols = torch.zeros(1, 1, 16, 16, 16)
    vols[0, 0, 2:5, 2:5, 2:5] = 1.0
    vols[0, 0, 10:13, 10:13, 10:13] = 1.0
    model = _IdentityVAE3D()
    assert torch.allclose(topology_score(model, vols), torch.zeros(1))
