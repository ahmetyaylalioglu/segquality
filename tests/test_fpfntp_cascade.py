"""Unit tests for the FP/FN/TP triple-head stage-1 and the recurrent stage-2 refiner.

Pins the 2026-10-20 direction's guarantees: (1) the TP/FP/FN targets are the correct set
relations and satisfy the decomposition ``G = TP + FN`` / ``M = TP + FP``; (2)
``reconstruct_from_fntp`` recovers the clean mask from the target FN/TP; (3) stage-1 consumes 2
channels and emits exactly 3; (4) the recurrent cascade returns one ``(FP, FN, TP)`` triple per
iteration, trains ONLY the refiner (stage-1 frozen), and ``predict_reference`` is a valid
probability map; (5) the per-iteration loss and its accumulation are finite and differentiable.
"""
from __future__ import annotations

import numpy as np
import torch

from data.fpfntp_targets import fpfntp_targets, reconstruct_from_fntp
from losses.fpfntp_losses import FPFNTPLossWeights, cascade_loss, refiner_iteration_loss
from models.vae3d_cascade_tp import CascadeFPFNTP, VARVQVAE3D_RefinerTP
from models.vae3d_var_fpfntp import VARVQVAE3D_ImgCond_FPFNTP

INPUT = (32, 32, 32)


def _kwargs() -> dict:
    return dict(base_channels=8, num_embeddings=64, embedding_dim=16, input_size=INPUT,
                v_patch_nums=(1, 2), using_znorm=False, beta=0.25, quant_resi=0.5)


def _x(b: int = 2) -> torch.Tensor:
    torch.manual_seed(0)
    img = torch.rand(b, 1, *INPUT)
    mask = (torch.rand(b, 1, *INPUT) > 0.5).float()
    return torch.cat([img, mask], dim=1)


def test_fpfntp_targets_decomposition_numpy() -> None:
    """FP=broken&~clean, FN=clean&~broken, TP=broken&clean, on a hand example."""
    broken = np.array([[1, 1, 0, 0]], dtype=np.float32)
    clean = np.array([[1, 0, 1, 0]], dtype=np.float32)
    fp, fn, tp = fpfntp_targets(broken, clean)
    np.testing.assert_array_equal(fp, np.array([[0, 1, 0, 0]], dtype=np.float32))
    np.testing.assert_array_equal(fn, np.array([[0, 0, 1, 0]], dtype=np.float32))
    np.testing.assert_array_equal(tp, np.array([[1, 0, 0, 0]], dtype=np.float32))


def test_fpfntp_targets_satisfy_decomposition_identities() -> None:
    """TP+FN == clean and TP+FP == broken, both backends in agreement."""
    rng = np.random.default_rng(0)
    broken = (rng.random((2, 1, 8, 8, 8)) > 0.5).astype(np.float32)
    clean = (rng.random((2, 1, 8, 8, 8)) > 0.5).astype(np.float32)
    fp, fn, tp = fpfntp_targets(broken, clean)
    np.testing.assert_array_equal(tp + fn, (clean > 0.5))
    np.testing.assert_array_equal(tp + fp, (broken > 0.5))
    bt, ct = torch.from_numpy(broken), torch.from_numpy(clean)
    fpt, fnt, tpt = fpfntp_targets(bt, ct)
    np.testing.assert_array_equal(fp, fpt.numpy())
    np.testing.assert_array_equal(fn, fnt.numpy())
    np.testing.assert_array_equal(tp, tpt.numpy())


def test_reconstruct_from_fntp_recovers_clean() -> None:
    """TP OR FN with the TARGET maps == clean, exactly."""
    rng = np.random.default_rng(1)
    broken = (rng.random((3, 10, 10, 10)) > 0.5)
    clean = (rng.random((3, 10, 10, 10)) > 0.5)
    _, fn, tp = fpfntp_targets(broken, clean)
    np.testing.assert_array_equal(reconstruct_from_fntp(fn, tp), clean)


def test_stage1_in2_out3() -> None:
    x = _x()
    model = VARVQVAE3D_ImgCond_FPFNTP(**_kwargs()).eval()
    assert model.in_channels == 2
    logits, _, _ = model.forward(x)
    assert logits.shape == (x.shape[0], 3, *INPUT)
    fp, fn, tp = model.predict_fpfntp(x)
    for t in (fp, fn, tp):
        assert t.shape == (x.shape[0], 1, *INPUT)
        assert (t >= 0).all() and (t <= 1).all()


def test_cascade_iterations_and_frozen_stage1() -> None:
    """forward returns one triple per iteration; backward trains only the refiner."""
    x = _x(1)
    s1 = VARVQVAE3D_ImgCond_FPFNTP(**_kwargs())
    ref = VARVQVAE3D_RefinerTP(**_kwargs())
    cas = CascadeFPFNTP(s1, ref, iters=3)
    assert cas.in_channels == 4 and cas.out_channels == 3
    outs, vq_total = cas.forward(x)
    assert len(outs) == 3 and torch.isfinite(vq_total)
    for fp, fn, tp in outs:
        assert fp.shape == fn.shape == tp.shape == (1, 1, *INPUT)

    clean = (torch.rand(1, 1, *INPUT) > 0.5).float()
    fp_t, fn_t, tp_t = fpfntp_targets(x[:, 1:2], clean)
    maps_loss, per_iter = cascade_loss(outs, fp_t, fn_t, tp_t, x[:, 0:1], clean)
    total = maps_loss + vq_total
    assert len(per_iter) == 3 and torch.isfinite(total)
    total.backward()
    assert not any(p.grad is not None for p in cas.stage1.parameters())
    assert any(p.grad is not None for p in cas.refiner.parameters())


def test_refiner_is_identity_at_init() -> None:
    """Zero-init head => residual delta is 0 at init, so refine_step reproduces its input maps."""
    cas = CascadeFPFNTP(VARVQVAE3D_ImgCond_FPFNTP(**_kwargs()),
                        VARVQVAE3D_RefinerTP(**_kwargs()), iters=1).eval()
    image = torch.rand(1, 1, *INPUT)
    fp = torch.rand(1, 1, *INPUT)
    fn = torch.rand(1, 1, *INPUT)
    tp = torch.rand(1, 1, *INPUT)
    with torch.no_grad():
        fp2, fn2, tp2, _ = cas.refine_step(image, fp, fn, tp)
    assert torch.allclose(fp2, fp, atol=1e-3)
    assert torch.allclose(fn2, fn, atol=1e-3)
    assert torch.allclose(tp2, tp, atol=1e-3)


def test_predict_reference_is_probability() -> None:
    x = _x(1)
    cas = CascadeFPFNTP(VARVQVAE3D_ImgCond_FPFNTP(**_kwargs()),
                        VARVQVAE3D_RefinerTP(**_kwargs()), iters=2).eval()
    ref = cas.predict_reference(x)
    assert ref.shape == (1, 1, *INPUT)
    assert (ref >= 0).all() and (ref <= 1).all()


def test_iteration_loss_components_present() -> None:
    """Single-iteration loss exposes the BCE and four image-aware MSE terms, all finite."""
    torch.manual_seed(0)
    shp = (2, 1, *INPUT)
    fp, fn, tp = (torch.rand(*shp) for _ in range(3))
    clean = (torch.rand(*shp) > 0.5).float()
    fp_t, fn_t, tp_t = fpfntp_targets((torch.rand(*shp) > 0.5).float(), clean)
    image = torch.rand(*shp)
    total, comps = refiner_iteration_loss(fp, fn, tp, fp_t, fn_t, tp_t, image, clean,
                                          FPFNTPLossWeights())
    assert set(comps) == {"bce", "dice", "mse_per_map_image", "mse_per_map_gt",
                          "mse_combined_image", "mse_combined_gt"}
    assert torch.isfinite(total) and all(torch.isfinite(v) for v in comps.values())
