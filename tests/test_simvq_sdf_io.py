"""Unit tests for the SimVQ SDF-IO variant (baseline E1) and its correlation evaluator.

Two correctness risks are pinned here, since they are what the combined model changes:
(1) ``quant_error_score_simvq_sdfio`` must encode the SDF INPUT (like SDF-io) AND
compare against the EFFECTIVE codebook ``C @ W`` (like SimVQ — there is no
``quantizer.embedding``); (2) the model must expose the SDF bridge (``predict_sdf`` /
``reconstruct``) while carrying the SimVQ quantizer (frozen base + learnable proj).
The registry mirrors SDF-io (drops ``recon_bce``, keeps ``sdf_l1``).
"""
from __future__ import annotations

import importlib

import numpy as np
import torch

evsimvqsdfio = importlib.import_module("scripts.evaluate_correlation_3d_simvq_sdf_io")

from models.vae3d_simvq import SimVQuantizer3D
from models.vae3d_simvq_sdf_io import ConvVQVAE3D_SimVQ_sdf_io

INPUT = (16, 16, 16)


def _model() -> ConvVQVAE3D_SimVQ_sdf_io:
    return ConvVQVAE3D_SimVQ_sdf_io(base_channels=8, num_embeddings=64, embedding_dim=16,
                                    commitment_cost=0.25, input_size=INPUT, use_skip=False)


def _batch(b: int = 2) -> torch.Tensor:
    torch.manual_seed(0)
    return (torch.rand(b, 1, *INPUT) > 0.5).float()


def test_model_is_simvq_with_sdf_bridge() -> None:
    """The variant must carry the SimVQ quantizer (no ``embedding``; frozen base + proj)
    and the SDF-io bridge (predict_sdf continuous, reconstruct binary)."""
    model = _model().eval()
    assert isinstance(model.quantizer, SimVQuantizer3D)
    assert not hasattr(model.quantizer, "embedding")
    assert not model.quantizer.codebook.weight.requires_grad  # C frozen
    x = _batch()
    sdf = model.predict_sdf(x)
    assert sdf.shape == x.shape
    recon = model.reconstruct(x)
    assert set(torch.unique(recon).tolist()) <= {0.0, 1.0}


def test_registry_matches_sdf_io_shape() -> None:
    names = set(evsimvqsdfio.SIMVQ_SDFIO_SCORE_FNS)
    assert "recon_bce" not in names
    assert {"recon_dice", "recon_l1", "topology", "quant_error", "sdf_l1"} == names


def test_quant_error_uses_effective_codebook_and_sdf_input() -> None:
    """Must equal: encode ``to_sdf_input(x)`` then nearest-code distance to the EFFECTIVE
    codebook ``C @ W`` — and DIFFER from using the frozen base ``C`` alone."""
    model = _model().eval()
    x = _batch()
    got = evsimvqsdfio.quant_error_score_simvq_sdfio(model, x)
    assert got.shape == (x.shape[0],)
    assert torch.isfinite(got).all() and (got >= 0).all()

    sdf_in = model.to_sdf_input(x)
    z_e, _ = model._encode_features(sdf_in)
    z = z_e.permute(0, 2, 3, 4, 1).contiguous().view(-1, model.embedding_dim)
    w_eff = model.quantizer.effective_codebook()
    dist = z.pow(2).sum(1, keepdim=True) - 2.0 * z @ w_eff.t() + w_eff.pow(2).sum(1)
    ref = dist.min(1).values.view(x.shape[0], -1).mean(1)
    assert torch.allclose(got, ref, atol=1e-5)

    # using the frozen base C (not C@W) gives a different distance
    w_base = model.quantizer.codebook.weight
    db = z.pow(2).sum(1, keepdim=True) - 2.0 * z @ w_base.t() + w_base.pow(2).sum(1)
    baseref = db.min(1).values.view(x.shape[0], -1).mean(1)
    assert not torch.allclose(got, baseref, atol=1e-5)


def test_score_masks_returns_all_names() -> None:
    model = _model().eval()
    masks = (np.random.default_rng(0).random((3, *INPUT)) > 0.5).astype(np.float32)
    out = evsimvqsdfio.score_masks_simvq_sdfio(model, masks, torch.device("cpu"))
    assert set(out) == set(evsimvqsdfio.SIMVQ_SDFIO_SCORE_FNS)
    for name, arr in out.items():
        assert arr.shape == (3,) and np.isfinite(arr).all()
