"""Unit tests for the conditioned 3D VQ-VAE (WP-2), tiny tensors, no data.

Two invariants matter most:
* ``cond_dim=0`` is a byte-identical no-op — same param_count as the base model.
* ``cond_dim>0`` actually wires the features in — the first decoder stage grows by
  cond_dim, the cond MLP + standardisation buffers exist, and forward/reconstruct
  still produce valid single-channel logits/probabilities from ``x`` alone.
"""
from __future__ import annotations

import torch

from models.vae3d_vq import ConvVQVAE3D
from models.vae3d_vq_cond import ConvVQVAE3D_cond
from quality.geom_features import resolve_feature_names

INPUT = (16, 16, 16)
FEATURES = resolve_feature_names(["topology", "shape"])


def _base() -> ConvVQVAE3D:
    return ConvVQVAE3D(base_channels=8, num_embeddings=64, embedding_dim=16,
                       commitment_cost=0.25, input_size=INPUT, use_skip=False)


def _cond(cond_dim: int = 16, use_skip: bool = False, feature_names=FEATURES) -> ConvVQVAE3D_cond:
    return ConvVQVAE3D_cond(
        base_channels=8, num_embeddings=64, embedding_dim=16, commitment_cost=0.25,
        input_size=INPUT, use_skip=use_skip, cond_dim=cond_dim,
        feature_names=feature_names, cond_hidden=8,
    )


def _batch(b: int = 2) -> torch.Tensor:
    torch.manual_seed(0)
    return (torch.rand(b, 1, *INPUT) > 0.5).float()


# -- cond_dim = 0 : exact no-op ---------------------------------------------
def test_cond_zero_matches_base_param_count() -> None:
    base = _base()
    cond0 = _cond(cond_dim=0, feature_names=None)
    assert cond0.param_count == base.param_count
    assert not hasattr(cond0, "cond_mlp")


def test_cond_zero_forward_and_reconstruct_shapes() -> None:
    model = _cond(cond_dim=0, feature_names=None)
    x = _batch()
    logits, vq_term, info = model(x)
    assert logits.shape == x.shape
    assert torch.isfinite(vq_term) and vq_term.item() >= 0.0
    probs = model.reconstruct(x)
    assert probs.shape == x.shape
    assert float(probs.min()) >= 0.0 and float(probs.max()) <= 1.0


# -- cond_dim > 0 : conditioning wired in -----------------------------------
def test_cond_requires_feature_names() -> None:
    import pytest

    with pytest.raises(ValueError):
        ConvVQVAE3D_cond(input_size=INPUT, cond_dim=16, feature_names=None)


def test_cond_builds_mlp_and_buffers() -> None:
    model = _cond(cond_dim=16)
    assert hasattr(model, "cond_mlp")
    assert model.feat_mean.shape == (len(FEATURES),)
    assert model.feat_std.shape == (len(FEATURES),)
    # first decoder stage input grew by exactly cond_dim vs the base model
    base_in = _base().dec_stages[0][0].in_channels
    assert model.dec_stages[0][0].in_channels == base_in + 16


def test_cond_has_more_params_than_base() -> None:
    assert _cond(cond_dim=16).param_count > _base().param_count


def test_cond_forward_and_reconstruct_valid() -> None:
    model = _cond(cond_dim=16)
    x = _batch()
    logits, vq_term, info = model(x)
    assert logits.shape == x.shape
    assert torch.isfinite(logits).all()
    assert "perplexity" in info and "codebook_used" in info
    probs = model.reconstruct(x)
    assert probs.shape == x.shape
    assert float(probs.min()) >= 0.0 and float(probs.max()) <= 1.0


def test_cond_backprop_reaches_encoder_and_cond_mlp() -> None:
    model = _cond(cond_dim=16)
    x = _batch()
    logits, vq_term, _ = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, x) + vq_term
    loss.backward()
    enc_conv = model.enc_stages[0][0]
    assert enc_conv.weight.grad is not None and enc_conv.weight.grad.abs().sum() > 0
    cond_lin = model.cond_mlp[0]
    assert cond_lin.weight.grad is not None and torch.isfinite(cond_lin.weight.grad).all()


def test_cond_topology_only_feature_width() -> None:
    names = resolve_feature_names(["topology"])
    model = _cond(cond_dim=8, feature_names=names)
    assert model.cond_mlp[0].in_features == len(names) == 2
    x = _batch()
    assert model.reconstruct(x).shape == x.shape


def test_set_feature_stats() -> None:
    model = _cond(cond_dim=16)
    n = len(FEATURES)
    mean = torch.arange(n, dtype=torch.float32)
    std = torch.zeros(n)  # zero variance -> must be floored, not divide-by-zero
    model.set_feature_stats(mean, std)
    assert torch.allclose(model.feat_mean, mean)
    assert (model.feat_std >= 1e-6).all()
    # forward still finite after setting degenerate stats
    assert torch.isfinite(model.reconstruct(_batch())).all()


def test_cond_with_skip_forward_shapes() -> None:
    model = _cond(cond_dim=16, use_skip=True)
    x = _batch()
    logits, vq_term, _ = model(x)
    assert logits.shape == x.shape
    assert torch.isfinite(vq_term)
