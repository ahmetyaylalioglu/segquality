"""Unit tests for the SDF-IO correlation evaluator's model-specific scores.

The two deviations from the vanilla-VQ evaluator carry the correctness risk, so they
are what we pin: (1) ``quant_error_score_sdfio`` must encode the SDF INPUT (the field
the encoder was trained on), NOT the raw binary mask; (2) ``sdf_l1_score`` is the
SDF-space reconstruction residual and must be a finite, non-negative per-mask scalar.
The score registry must also exclude ``recon_bce`` (meaningless on a raw SDF output).
"""
from __future__ import annotations

import importlib

import numpy as np
import torch

evsdfio = importlib.import_module("scripts.evaluate_correlation_3d_sdf_io")

from models.vae3d_vq_sdf_io import ConvVQVAE3D_sdf_io

INPUT = (16, 16, 16)


def _model() -> ConvVQVAE3D_sdf_io:
    return ConvVQVAE3D_sdf_io(base_channels=8, num_embeddings=64, embedding_dim=16,
                              commitment_cost=0.25, input_size=INPUT, use_skip=False)


def _batch(b: int = 2) -> torch.Tensor:
    torch.manual_seed(0)
    return (torch.rand(b, 1, *INPUT) > 0.5).float()


def test_registry_drops_bce_and_adds_sdf_l1() -> None:
    names = set(evsdfio.SDFIO_SCORE_FNS)
    assert "recon_bce" not in names
    assert "sdf_l1" in names and "quant_error" in names
    assert names == set(evsdfio.SCORE_DIRECTIONS_SDFIO)


def test_quant_error_encodes_the_sdf_input_not_the_binary() -> None:
    """The SDF-IO quant_error must equal encoding ``to_sdf_input(x)``, and must DIFFER
    from encoding the raw binary mask (that is the whole point of the override)."""
    model = _model().eval()
    x = _batch()
    got = evsdfio.quant_error_score_sdfio(model, x)
    assert got.shape == (x.shape[0],)
    assert torch.isfinite(got).all() and (got >= 0).all()

    # reference: encode the SDF input explicitly -> identical
    sdf_in = model.to_sdf_input(x)
    z_e, _ = model._encode_features(sdf_in)
    z = z_e.permute(0, 2, 3, 4, 1).contiguous().view(-1, model.embedding_dim)
    w = model.quantizer.embedding.weight
    dist = z.pow(2).sum(1, keepdim=True) - 2.0 * z @ w.t() + w.pow(2).sum(1)
    ref = dist.min(1).values.view(x.shape[0], -1).mean(1)
    assert torch.allclose(got, ref, atol=1e-5)

    # encoding the BINARY mask directly gives a different distance (SDF != binary)
    z_bin, _ = model._encode_features(x)
    zb = z_bin.permute(0, 2, 3, 4, 1).contiguous().view(-1, model.embedding_dim)
    db = zb.pow(2).sum(1, keepdim=True) - 2.0 * zb @ w.t() + w.pow(2).sum(1)
    binref = db.min(1).values.view(x.shape[0], -1).mean(1)
    assert not torch.allclose(got, binref, atol=1e-5)


def test_sdf_l1_is_finite_nonneg_per_mask() -> None:
    model = _model().eval()
    x = _batch(3)
    s = evsdfio.sdf_l1_score(model, x)
    assert s.shape == (3,)
    assert torch.isfinite(s).all() and (s >= 0).all()
    # matches the explicit residual definition
    ref = (model.predict_sdf(x) - model.to_sdf_input(x)).abs().mean(dim=(1, 2, 3, 4))
    assert torch.allclose(s, ref, atol=1e-6)


def test_score_masks_returns_all_names() -> None:
    model = _model().eval()
    masks = (np.random.default_rng(0).random((3, *INPUT)) > 0.5).astype(np.float32)
    out = evsdfio.score_masks_sdfio(model, masks, torch.device("cpu"))
    assert set(out) == set(evsdfio.SDFIO_SCORE_FNS)
    for name, arr in out.items():
        assert arr.shape == (3,) and np.isfinite(arr).all()
