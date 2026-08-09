"""Tests for the Phase-6 quality scores and correlation utilities.

All data-free: a tiny randomly-initialised ``ConvVAE`` (and, for the exact-value
sanity checks, an identity stub) on synthetic masks. These cover the properties
the evaluation hinges on — correct shapes/batching, determinism in eval mode
(z = mu), the exact values on a perfect reconstruction, and that the correlation
helpers recover a known monotonic relationship and stratify by Dice band without
crashing on sparse bands.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from models.vae import ConvVAE
from quality.score import (
    DICE_BANDS,
    SCORE_FNS,
    banded_correlation,
    compute_all_scores,
    correlation,
    recon_dice_score,
    recon_l1_score,
    topology_score,
)


def _tiny_vae() -> ConvVAE:
    torch.manual_seed(0)
    return ConvVAE(latent_dim=4, base_channels=4, input_size=(32, 32)).eval()


def _random_masks(batch: int = 6) -> Tensor:
    torch.manual_seed(1)
    return (torch.rand(batch, 1, 32, 32) > 0.5).float()


class _IdentityVAE(nn.Module):
    """Stub whose reconstruction is exactly the input, for exact-value checks."""

    def reconstruct(self, x: Tensor) -> Tensor:
        return x

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        logits = torch.logit(x.clamp(1e-4, 1 - 1e-4))
        z = torch.zeros(x.shape[0], 4)
        return logits, z, z

    def _encode_dist(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = torch.zeros(x.shape[0], 4)
        return z, z


def test_every_score_returns_one_scalar_per_mask() -> None:
    model, masks = _tiny_vae(), _random_masks(6)
    for name, fn in SCORE_FNS.items():
        out = fn(model, masks)
        assert out.shape == (6,), f"{name} returned shape {tuple(out.shape)}"
        assert torch.isfinite(out).all(), f"{name} produced non-finite values"


def test_compute_all_scores_keys_and_length() -> None:
    model, masks = _tiny_vae(), _random_masks(5)
    scores = compute_all_scores(model, masks)
    assert set(scores) == set(SCORE_FNS)
    for name, arr in scores.items():
        assert isinstance(arr, np.ndarray) and arr.shape == (5,), name


def test_scores_are_deterministic_in_eval() -> None:
    model, masks = _tiny_vae(), _random_masks(6)
    first = compute_all_scores(model, masks)
    second = compute_all_scores(model, masks)
    for name in SCORE_FNS:
        assert np.allclose(first[name], second[name]), f"{name} not deterministic"


def test_perfect_reconstruction_gives_dice_one_and_zero_l1() -> None:
    model, masks = _IdentityVAE(), _random_masks(4)
    assert torch.allclose(recon_dice_score(model, masks), torch.ones(4))
    assert torch.allclose(recon_l1_score(model, masks), torch.zeros(4), atol=1e-6)
    # Identical reconstruction => identical topology => zero component mismatch.
    assert torch.allclose(topology_score(model, masks), torch.zeros(4))


def test_correlation_recovers_perfect_monotonic() -> None:
    true_dice = np.linspace(0.4, 1.0, 20)
    # A score that increases monotonically with Dice -> Spearman +1.
    stats = correlation(true_dice * 2.0, true_dice)
    assert abs(stats["spearman"] - 1.0) < 1e-9
    # A monotonically decreasing (error-like) score -> Spearman -1.
    stats_neg = correlation(1.0 - true_dice, true_dice)
    assert abs(stats_neg["spearman"] + 1.0) < 1e-9


def test_correlation_handles_degenerate_input() -> None:
    # Fewer than three points, or a constant score, must not raise.
    assert np.isnan(correlation(np.array([0.1, 0.2]), np.array([0.5, 0.6]))["spearman"])
    const = correlation(np.ones(10), np.linspace(0.4, 1.0, 10))
    assert np.isnan(const["spearman"]) and const["n"] == 10


def test_banded_correlation_covers_bands_and_survives_sparsity() -> None:
    rng = np.random.default_rng(0)
    true_dice = rng.uniform(0.4, 1.0, size=200)
    scores = 1.0 - true_dice + rng.normal(0, 0.01, size=200)  # error-like
    bands = banded_correlation(scores, true_dice)
    assert len(bands) == len(DICE_BANDS)
    for entry, (lo, hi) in zip(bands, DICE_BANDS):
        assert entry["band_lo"] == lo and entry["band_hi"] == hi
        assert "spearman" in entry and "n" in entry
    # The clean reference at exactly 1.0 lands in the last (closed) band.
    last = banded_correlation(np.array([0.0]), np.array([1.0]))[-1]
    assert last["n"] == 1
