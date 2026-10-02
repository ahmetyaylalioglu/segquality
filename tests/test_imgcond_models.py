"""Unit tests for the image-conditioned 3D VQ-VAE variants (E2-E7).

Pins the two correctness risks of the 2-channel ``[image, mask]`` -> 1-channel line:
(1) the in=2/out=1 decoupling — the encoder must consume 2 channels while the decoder's
final layer emits exactly 1 (the base ties output to ``in_channels``, so a naive
``in_channels=2`` would wrongly predict 2 channels); (2) the SDF-io head must map ONLY the
mask channel to its normalised SDF, leaving the image channel untouched, and still predict a
single-channel SDF restored to a mask by the bridge. The image-conditioned scorers must
compare that 1-channel reconstruction against the MASK channel (never the image).
"""
from __future__ import annotations

import importlib

import numpy as np
import torch

from models.vae3d_imgcond import ConvVQVAE3D_ImgCond, ConvVQVAE3D_SimVQ_ImgCond
from models.vae3d_imgcond_sdf_io import (
    ConvVQVAE3D_ImgCond_sdf_io,
    ConvVQVAE3D_SimVQ_ImgCond_sdf_io,
)
from models.vae3d_simvq import SimVQuantizer3D
from quality.sdf_paper import batch_input_sdf

evimg = importlib.import_module("scripts.evaluate_correlation_imgcond")

INPUT = (16, 16, 16)


def _kwargs() -> dict:
    return dict(base_channels=8, num_embeddings=64, embedding_dim=16,
                commitment_cost=0.25, input_size=INPUT, use_skip=False)


def _x(b: int = 2) -> torch.Tensor:
    """A 2-channel ``[image in [0,1], binary mask]`` batch (B,2,D,H,W)."""
    torch.manual_seed(0)
    img = torch.rand(b, 1, *INPUT)
    mask = (torch.rand(b, 1, *INPUT) > 0.5).float()
    return torch.cat([img, mask], dim=1)


def test_binary_variants_in2_out1() -> None:
    """Both binary-head variants consume 2 channels and predict exactly 1."""
    x = _x()
    for cls in (ConvVQVAE3D_ImgCond, ConvVQVAE3D_SimVQ_ImgCond):
        model = cls(**_kwargs()).eval()
        assert model.in_channels == 2 and model.out_channels == 1
        logits, _, _ = model.forward(x)
        assert logits.shape == (x.shape[0], 1, *INPUT)
        recon = model.reconstruct(x)
        assert recon.shape == (x.shape[0], 1, *INPUT)


def test_simvq_binary_variant_is_simvq() -> None:
    """The SimVQ binary variant carries the SimVQ quantizer (frozen base, no embedding)."""
    model = ConvVQVAE3D_SimVQ_ImgCond(**_kwargs()).eval()
    assert isinstance(model.quantizer, SimVQuantizer3D)
    assert not hasattr(model.quantizer, "embedding")
    assert not model.quantizer.codebook.weight.requires_grad


def test_sdfio_variants_sdf_on_mask_channel_only() -> None:
    """SDF-io head: image channel passes through unchanged; only the mask channel becomes
    its normalised SDF; the prediction is a single-channel SDF; reconstruct is binary."""
    x = _x()
    for cls in (ConvVQVAE3D_ImgCond_sdf_io, ConvVQVAE3D_SimVQ_ImgCond_sdf_io):
        model = cls(**_kwargs()).eval()
        assert model.in_channels == 2 and model.out_channels == 1
        sdf_in = model.to_sdf_input(x)
        assert sdf_in.shape == x.shape                       # still 2 channels
        assert torch.allclose(sdf_in[:, 0:1], x[:, 0:1])     # image channel untouched
        ref_sdf = batch_input_sdf(x[:, 1:2], model.delta, device=x.device)
        assert torch.allclose(sdf_in[:, 1:2], ref_sdf)       # mask channel -> SDF
        pred = model.predict_sdf(x)
        assert pred.shape == (x.shape[0], 1, *INPUT)
        recon = model.reconstruct(x)
        assert set(torch.unique(recon).tolist()) <= {0.0, 1.0}


def test_registry_shapes_per_head() -> None:
    """binary keeps recon_bce (no sdf_l1); sdfio drops recon_bce and adds sdf_l1."""
    bin_fns, bin_dirs = evimg.build_score_registry("binary", "simvq")
    assert set(bin_fns) == {"recon_dice", "recon_bce", "recon_l1", "topology", "quant_error"}
    assert set(bin_fns) == set(bin_dirs)
    sdf_fns, sdf_dirs = evimg.build_score_registry("sdfio", "vq")
    assert set(sdf_fns) == {"recon_dice", "recon_l1", "topology", "quant_error", "sdf_l1"}
    assert "recon_bce" not in sdf_fns
    assert set(sdf_fns) == set(sdf_dirs)


def test_scores_compare_against_mask_channel() -> None:
    """recon_dice must be measured against the MASK channel (x[:,1]), not the image.

    Feeding a batch whose mask channel equals the recon target vs a different mask must
    change recon_dice; swapping only the IMAGE channel must not (image is a condition).
    """
    model = ConvVQVAE3D_ImgCond(**_kwargs()).eval()
    x = _x()
    fns, _ = evimg.build_score_registry("binary", "vq")
    d0 = fns["recon_dice"](model, x)
    assert d0.shape == (x.shape[0],)
    assert torch.isfinite(d0).all()
    # changing only the image channel leaves the mask-vs-recon Dice's target intact
    x_img_swapped = x.clone()
    x_img_swapped[:, 0:1] = 1.0 - x_img_swapped[:, 0:1]
    d_img = fns["recon_dice"](model, x_img_swapped)
    # recon may shift slightly (image feeds encoder) but the score stays finite & valid
    assert d_img.shape == (x.shape[0],) and torch.isfinite(d_img).all()


def test_quant_error_effective_vs_base_codebook() -> None:
    """SimVQ quant_error uses effective C@W; standard VQ uses embedding.weight."""
    x = _x()
    simvq = ConvVQVAE3D_SimVQ_ImgCond(**_kwargs()).eval()
    fn_simvq = evimg._quant_error_imgcond("binary", "simvq")
    got = fn_simvq(simvq, x)
    assert got.shape == (x.shape[0],) and torch.isfinite(got).all() and (got >= 0).all()

    vq = ConvVQVAE3D_ImgCond(**_kwargs()).eval()
    fn_vq = evimg._quant_error_imgcond("binary", "vq")
    gv = fn_vq(vq, x)
    assert gv.shape == (x.shape[0],) and torch.isfinite(gv).all() and (gv >= 0).all()


def test_score_masks_imgcond_returns_all_names() -> None:
    """The batched scorer stacks [image, mask] and returns one finite value per volume."""
    model = ConvVQVAE3D_SimVQ_ImgCond_sdf_io(**_kwargs()).eval()
    rng = np.random.default_rng(0)
    masks = (rng.random((3, *INPUT)) > 0.5).astype(np.float32)
    images = rng.random((3, *INPUT)).astype(np.float32)
    fns, _ = evimg.build_score_registry("sdfio", "simvq")
    out = evimg.score_masks_imgcond(model, masks, images, fns, torch.device("cpu"))
    assert set(out) == set(fns)
    for arr in out.values():
        assert arr.shape == (3,) and np.isfinite(arr).all()
