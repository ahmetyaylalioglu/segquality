"""Unit tests for the FP/FN dual-head image-conditioned 3D VQ-VAE (direction G).

Pins the direction's correctness guarantees: (1) the model consumes 2 channels and emits
exactly 2 (FP + FN); (2) the FP/FN targets are the correct set relations w.r.t. the input
mask; (3) ``correct_mask`` with the TARGET FP/FN recovers the clean mask exactly; (4) the
GT-free ``estimated_dice`` equals the true Dice when fed the target FP/FN (so it is a
faithful estimator at the supervision limit); (5) the eval score registry/scorer shapes.
"""
from __future__ import annotations

import importlib

import numpy as np
import torch

from data.fpfn_targets import correct_mask, error_volume, estimated_dice, fpfn_targets
from models.vae3d_imgcond_fpfn import (
    ConvVQVAE3D_ImgCond_FPFN,
    ConvVQVAE3D_ImgCond_FPFN_sdf_io,
    ConvVQVAE3D_SimVQ_ImgCond_FPFN,
    ConvVQVAE3D_SimVQ_ImgCond_FPFN_sdf_io,
)
from models.vae3d_simvq import SimVQuantizer3D
from quality.sdf_paper import batch_input_sdf

evfpfn = importlib.import_module("scripts.evaluate_correlation_imgcond_fpfn")

INPUT = (16, 16, 16)


def _kwargs() -> dict:
    return dict(base_channels=8, num_embeddings=64, embedding_dim=16,
                commitment_cost=0.25, input_size=INPUT, use_skip=False)


def _x(b: int = 2) -> torch.Tensor:
    """A 2-channel ``[image in [0,1], broken binary mask]`` batch (B,2,D,H,W)."""
    torch.manual_seed(0)
    img = torch.rand(b, 1, *INPUT)
    mask = (torch.rand(b, 1, *INPUT) > 0.5).float()
    return torch.cat([img, mask], dim=1)


def test_fpfn_variants_in2_out2() -> None:
    """Both FP/FN variants consume 2 channels and predict exactly 2 (FP + FN)."""
    x = _x()
    for cls in (ConvVQVAE3D_ImgCond_FPFN, ConvVQVAE3D_SimVQ_ImgCond_FPFN):
        model = cls(**_kwargs()).eval()
        assert model.in_channels == 2 and model.out_channels == 2
        logits, _, _ = model.forward(x)
        assert logits.shape == (x.shape[0], 2, *INPUT)
        fp, fn = model.predict_fpfn(x)
        assert fp.shape == fn.shape == (x.shape[0], 1, *INPUT)
        assert (fp >= 0).all() and (fp <= 1).all() and (fn >= 0).all() and (fn <= 1).all()


def test_simvq_variant_is_simvq() -> None:
    model = ConvVQVAE3D_SimVQ_ImgCond_FPFN(**_kwargs()).eval()
    assert isinstance(model.quantizer, SimVQuantizer3D)
    assert not hasattr(model.quantizer, "embedding")


def test_sdfio_variants_in2_out2_binary_bridge() -> None:
    """SDF-io FP/FN: 2-channel SDF prediction; image channel passes through to_sdf_input
    unchanged, mask channel becomes its SDF; predict_fpfn bridges both channels to binary."""
    x = _x()
    for cls in (ConvVQVAE3D_ImgCond_FPFN_sdf_io, ConvVQVAE3D_SimVQ_ImgCond_FPFN_sdf_io):
        model = cls(**_kwargs()).eval()
        assert model.in_channels == 2 and model.out_channels == 2
        sdf_in = model.to_sdf_input(x)
        assert sdf_in.shape == x.shape
        assert torch.allclose(sdf_in[:, 0:1], x[:, 0:1])             # image untouched
        ref = batch_input_sdf(x[:, 1:2], model.delta, device=x.device)
        assert torch.allclose(sdf_in[:, 1:2], ref)                   # mask -> SDF
        sdf = model.predict_sdf(x)
        assert sdf.shape == (x.shape[0], 2, *INPUT)
        fp, fn = model.predict_fpfn(x)
        assert fp.shape == fn.shape == (x.shape[0], 1, *INPUT)
        for t in (fp, fn):
            assert set(torch.unique(t).tolist()) <= {0.0, 1.0}       # binary via bridge


def test_sdfio_simvq_variant_is_simvq() -> None:
    model = ConvVQVAE3D_SimVQ_ImgCond_FPFN_sdf_io(**_kwargs()).eval()
    assert isinstance(model.quantizer, SimVQuantizer3D)


def test_fpfn_targets_set_relations_numpy() -> None:
    """FP = broken AND NOT clean; FN = clean AND NOT broken, on a hand example."""
    broken = np.array([[1, 1, 0, 0]], dtype=np.float32)
    clean = np.array([[1, 0, 1, 0]], dtype=np.float32)
    fp, fn = fpfn_targets(broken, clean)
    np.testing.assert_array_equal(fp, np.array([[0, 1, 0, 0]], dtype=np.float32))  # remove index 1
    np.testing.assert_array_equal(fn, np.array([[0, 0, 1, 0]], dtype=np.float32))  # add index 2


def test_fpfn_targets_backend_matches_torch() -> None:
    """Torch and NumPy backends agree on the same data."""
    rng = np.random.default_rng(0)
    broken = (rng.random((2, 8, 8, 8)) > 0.5).astype(np.float32)
    clean = (rng.random((2, 8, 8, 8)) > 0.5).astype(np.float32)
    fp_np, fn_np = fpfn_targets(broken, clean)
    fp_t, fn_t = fpfn_targets(torch.from_numpy(broken), torch.from_numpy(clean))
    np.testing.assert_array_equal(fp_np, fp_t.numpy())
    np.testing.assert_array_equal(fn_np, fn_t.numpy())


def test_correct_mask_with_targets_recovers_clean() -> None:
    """(broken AND NOT FP_tgt) OR FN_tgt == clean, exactly."""
    rng = np.random.default_rng(1)
    broken = (rng.random((3, 10, 10, 10)) > 0.5)
    clean = (rng.random((3, 10, 10, 10)) > 0.5)
    fp, fn = fpfn_targets(broken, clean)
    recovered = correct_mask(broken, fp, fn)
    np.testing.assert_array_equal(recovered, clean)


def test_estimated_dice_equals_true_dice_at_target() -> None:
    """With the TARGET FP/FN, estimated_dice equals the real dice(broken, clean)."""
    rng = np.random.default_rng(2)
    broken = (rng.random((4, 1, 8, 8, 8)) > 0.5).astype(np.float32)
    clean = (rng.random((4, 1, 8, 8, 8)) > 0.5).astype(np.float32)
    bt = torch.from_numpy(broken)
    ct = torch.from_numpy(clean)
    fp, fn = fpfn_targets(bt, ct)
    est = estimated_dice(bt, fp, fn, eps=0.0)
    dims = (1, 2, 3, 4)
    inter = (bt * ct).sum(dim=dims)
    true_dice = (2 * inter) / (bt.sum(dim=dims) + ct.sum(dim=dims))
    assert torch.allclose(est, true_dice, atol=1e-5)


def test_error_volume_counts_target_errors() -> None:
    """error_volume at the target FP/FN == |broken XOR clean| per sample."""
    rng = np.random.default_rng(3)
    broken = (rng.random((2, 1, 8, 8, 8)) > 0.5).astype(np.float32)
    clean = (rng.random((2, 1, 8, 8, 8)) > 0.5).astype(np.float32)
    bt = torch.from_numpy(broken); ct = torch.from_numpy(clean)
    fp, fn = fpfn_targets(bt, ct)
    ev = error_volume(fp, fn)
    xor = ((bt > 0.5) ^ (ct > 0.5)).float().sum(dim=(1, 2, 3, 4))
    assert torch.allclose(ev, xor)


def test_eval_score_registry_names() -> None:
    fns, dirs = evfpfn.build_score_registry()
    assert set(fns) == {"est_dice", "error_volume", "fp_volume", "fn_volume"}
    assert set(fns) == set(dirs)
    assert dirs["est_dice"] == +1 and dirs["error_volume"] == -1


def test_score_masks_fpfn_returns_finite() -> None:
    """The batched scorer stacks [image, broken] and returns one finite value per volume."""
    model = ConvVQVAE3D_ImgCond_FPFN(**_kwargs()).eval()
    rng = np.random.default_rng(0)
    broken = (rng.random((3, *INPUT)) > 0.5).astype(np.float32)
    images = rng.random((3, *INPUT)).astype(np.float32)
    fns, _ = evfpfn.build_score_registry()
    out = evfpfn.score_masks_fpfn(model, broken, images, fns, torch.device("cpu"))
    assert set(out) == set(fns)
    for arr in out.values():
        assert arr.shape == (3,) and np.isfinite(arr).all()
