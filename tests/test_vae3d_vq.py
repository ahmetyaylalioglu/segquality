"""Unit tests for the 3D VQ-VAE (Phase 9), all on tiny synthetic tensors (no data).

Kept at 16^3 so a full forward/backward is milliseconds. 16 / 2^4 = 1, so the
quantized grid is 1x1x1 here; the 64^3 training run uses a 4x4x4 grid. These tests
guard the contract the rest of the pipeline relies on: output shapes, that the
quantizer really emits codebook vectors, a differentiable straight-through path,
non-negative VQ loss, and codebook-usage diagnostics.
"""
from __future__ import annotations

import torch

from losses.vae_losses import vq_loss
from models.vae3d_vq import ConvVQVAE3D, VectorQuantizer3D

INPUT = (16, 16, 16)


def _model(use_skip: bool = False) -> ConvVQVAE3D:
    return ConvVQVAE3D(
        base_channels=8,
        num_embeddings=64,
        embedding_dim=16,
        commitment_cost=0.25,
        input_size=INPUT,
        use_skip=use_skip,
    )


def _batch(b: int = 2) -> torch.Tensor:
    torch.manual_seed(0)
    return (torch.rand(b, 1, *INPUT) > 0.5).float()


# -- quantizer ---------------------------------------------------------------
def test_quantizer_output_comes_from_codebook() -> None:
    """z_q values must equal exact codebook rows (straight-through preserves value)."""
    vq = VectorQuantizer3D(num_embeddings=8, embedding_dim=4, commitment_cost=0.25)
    z = torch.randn(2, 4, 3, 3, 3)
    z_q, loss, info = vq(z)
    assert z_q.shape == z.shape
    assert torch.isfinite(loss) and loss.item() >= 0.0
    # Recompute nearest indices independently and confirm z_q matches those rows.
    flat = z.permute(0, 2, 3, 4, 1).reshape(-1, 4)
    w = vq.embedding.weight
    dist = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ w.t() + w.pow(2).sum(1)
    idx = dist.argmin(1)
    expected = vq.embedding(idx).view(2, 3, 3, 3, 4).permute(0, 4, 1, 2, 3)
    assert torch.allclose(z_q, expected, atol=1e-6)


def test_quantizer_perplexity_and_usage() -> None:
    vq = VectorQuantizer3D(num_embeddings=16, embedding_dim=4, commitment_cost=0.25)
    z = torch.randn(4, 4, 2, 2, 2)
    _, _, info = vq(z)
    assert info["perplexity"].item() > 0.0
    assert 1 <= int(info["codebook_used"].item()) <= 16


# -- model forward / reconstruct --------------------------------------------
def test_forward_shapes_and_vq_loss() -> None:
    model = _model()
    x = _batch()
    logits, vq_term, info = model(x)
    assert logits.shape == x.shape                       # (B,1,16,16,16)
    assert torch.isfinite(vq_term) and vq_term.item() >= 0.0
    assert "perplexity" in info and "codebook_used" in info


def test_reconstruct_is_probability() -> None:
    model = _model()
    x = _batch()
    probs = model.reconstruct(x)
    assert probs.shape == x.shape
    assert float(probs.min()) >= 0.0 and float(probs.max()) <= 1.0


def test_encode_decode_roundtrip_shapes() -> None:
    model = _model(use_skip=False)
    x = _batch()
    code = model.encode(x)
    # flattened quantized grid: embedding_dim * prod(feat) = 16 * 1*1*1 = 16
    assert code.shape == (x.size(0), 16)
    z_q, _, _ = model.quantize(x)
    logits = model.decode(z_q)
    assert logits.shape == x.shape


def test_vq_loss_combines_recon_and_vq() -> None:
    model = _model()
    x = _batch()
    logits, vq_term, _ = model(x)
    terms = vq_loss(logits, x, vq_term, dice_weight=1.0)
    assert set(terms) == {"total", "recon", "bce", "dice", "vq"}
    assert torch.isfinite(terms["total"]) and terms["total"].item() >= 0.0
    # total = recon + vq (no KL)
    assert torch.allclose(terms["total"], terms["recon"] + terms["vq"])


def test_backprop_reaches_encoder() -> None:
    """Straight-through estimator must let gradients reach encoder weights."""
    model = _model()
    x = _batch()
    logits, vq_term, _ = model(x)
    terms = vq_loss(logits, x, vq_term)
    terms["total"].backward()
    first_conv = model.enc_stages[0][0]
    assert first_conv.weight.grad is not None
    assert torch.isfinite(first_conv.weight.grad).all()
    assert first_conv.weight.grad.abs().sum().item() > 0.0


# -- skip-connection variant -------------------------------------------------
def test_use_skip_forward_shapes() -> None:
    model = _model(use_skip=True)
    x = _batch()
    logits, vq_term, _ = model(x)
    assert logits.shape == x.shape
    assert torch.isfinite(vq_term)


def test_input_size_divisibility_validated() -> None:
    import pytest

    with pytest.raises(ValueError):
        ConvVQVAE3D(input_size=(20, 20, 20))  # 20 not divisible by 16
