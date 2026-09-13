"""Unit tests for the SDF-target VQ-VAE (WP-3), tiny tensors, no data.

Key invariants: the model is architecturally identical to the base VQ-VAE (same
param_count), ``predict_sdf`` is the raw continuous output (can be negative — no
sigmoid), and ``reconstruct`` is the binary bridge ``sdf < 0 -> inside`` producing a
valid {0,1} mask that the existing scores can consume.
"""
from __future__ import annotations

import torch

from models.vae3d_vq import ConvVQVAE3D
from models.vae3d_vq_sdf import ConvVQVAE3D_sdf

INPUT = (16, 16, 16)


def _base() -> ConvVQVAE3D:
    return ConvVQVAE3D(base_channels=8, num_embeddings=64, embedding_dim=16,
                       commitment_cost=0.25, input_size=INPUT, use_skip=False)


def _sdf() -> ConvVQVAE3D_sdf:
    return ConvVQVAE3D_sdf(base_channels=8, num_embeddings=64, embedding_dim=16,
                           commitment_cost=0.25, input_size=INPUT, use_skip=False)


def _batch(b: int = 2) -> torch.Tensor:
    torch.manual_seed(0)
    return (torch.rand(b, 1, *INPUT) > 0.5).float()


def test_sdf_model_matches_base_param_count() -> None:
    """No new architecture -> identical parameter count to the base VQ-VAE."""
    assert _sdf().param_count == _base().param_count


def test_predict_sdf_is_raw_continuous() -> None:
    model = _sdf()
    x = _batch()
    sdf = model.predict_sdf(x)
    assert sdf.shape == x.shape
    assert torch.isfinite(sdf).all()
    # raw regression output is unbounded; with random init it should not be a
    # probability (i.e. not confined to [0,1]) -> confirms no sigmoid was applied.
    assert float(sdf.min()) < 0.0


def test_reconstruct_is_binary_bridge() -> None:
    model = _sdf()
    x = _batch()
    recon = model.reconstruct(x)
    assert recon.shape == x.shape
    uniq = torch.unique(recon)
    assert set(uniq.tolist()).issubset({0.0, 1.0}), "bridge must yield a binary mask"
    # bridge is exactly (predict_sdf < 0)
    assert torch.equal(recon, (model.predict_sdf(x) < 0).float())


def test_forward_still_returns_triplet() -> None:
    model = _sdf()
    logits, vq_term, info = model.forward(_batch())
    assert logits.shape == (2, 1, *INPUT)
    assert torch.isfinite(vq_term) and vq_term.item() >= 0.0
    assert "perplexity" in info and "codebook_used" in info


def test_backprop_through_smooth_l1_reaches_encoder() -> None:
    import torch.nn.functional as F

    from quality.sdf import batch_target_sdf

    model = _sdf()
    x = _batch()
    target = batch_target_sdf(x)
    pred_sdf, vq_term, _ = model.forward(x)
    loss = F.smooth_l1_loss(pred_sdf, target) + vq_term
    loss.backward()
    g = model.enc_stages[0][0].weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
