"""Tests for the 3D VAE and 3D volume transforms (Phase 7).

All data-free and fast: a tiny 16^3 ``ConvVAE3D`` (base 4, latent 8) on synthetic
volumes, and pure-function checks on ``volume_transforms``. These mirror the 2D
``test_vae.py`` one axis higher, covering shapes, eval determinism (z = mu),
encode/decode round-trip, and the transform invariants (cubing, nearest resize
staying binary, zero-padding off the volume edge).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from data.volume_transforms import (
    compute_bbox_3d,
    crop_with_padding_3d,
    cube_bbox,
    extract_organ_volume,
    resize_nearest_3d,
)
from models.vae3d import ConvVAE3D

SIZE = (16, 16, 16)  # divisible by 2^4, so 4 stride-2 stages -> 1^3 bottleneck


def _tiny_vae3d() -> ConvVAE3D:
    torch.manual_seed(0)
    return ConvVAE3D(latent_dim=8, base_channels=4, input_size=SIZE)


def _random_volumes(batch: int = 2) -> torch.Tensor:
    torch.manual_seed(1)
    return (torch.rand(batch, 1, *SIZE) > 0.5).float()


# -- model ------------------------------------------------------------------
def test_forward_shapes() -> None:
    model, x = _tiny_vae3d().eval(), _random_volumes(2)
    logits, mu, logvar = model(x)
    assert logits.shape == (2, 1, *SIZE)
    assert mu.shape == (2, 8) and logvar.shape == (2, 8)


def test_reparameterize_deterministic_in_eval() -> None:
    model, x = _tiny_vae3d().eval(), _random_volumes(2)
    a, _, _ = model(x)
    b, _, _ = model(x)
    assert torch.allclose(a, b)


def test_reparameterize_stochastic_in_train() -> None:
    model, x = _tiny_vae3d().train(), _random_volumes(2)
    torch.manual_seed(0)
    a, _, _ = model(x)
    torch.manual_seed(1)
    b, _, _ = model(x)
    assert not torch.allclose(a, b)


def test_reconstruct_is_probability() -> None:
    model, x = _tiny_vae3d().eval(), _random_volumes(2)
    probs = model.reconstruct(x)
    assert probs.shape == (2, 1, *SIZE)
    assert float(probs.min()) >= 0.0 and float(probs.max()) <= 1.0


def test_encode_decode_roundtrip_shapes() -> None:
    model, x = _tiny_vae3d().eval(), _random_volumes(3)
    z = model.encode(x)
    assert z.shape == (3, 8)
    assert model.decode(z).shape == (3, 1, *SIZE)


def test_decoder_uses_latent() -> None:
    model = _tiny_vae3d().eval()
    torch.manual_seed(2)
    z1 = torch.randn(1, 8)
    z2 = torch.randn(1, 8)
    assert not torch.allclose(model.decode(z1), model.decode(z2))


def test_backprop_runs() -> None:
    model, x = _tiny_vae3d().train(), _random_volumes(2)
    logits, mu, logvar = model(x)
    loss = logits.mean() + mu.pow(2).mean() + logvar.pow(2).mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_input_size_indivisible_raises() -> None:
    with pytest.raises(ValueError):
        ConvVAE3D(latent_dim=8, base_channels=4, input_size=(16, 16, 20))


def test_introspection_positive() -> None:
    model = _tiny_vae3d()
    assert model.param_count > 0
    assert model.estimate_forward_mb(batch_size=4) > 0.0


# -- volume transforms ------------------------------------------------------
def test_compute_bbox_3d_is_tight() -> None:
    vol = np.zeros((10, 10, 10), dtype=bool)
    vol[2:5, 3:7, 4:8] = True
    assert compute_bbox_3d(vol) == (2, 5, 3, 7, 4, 8)


def test_compute_bbox_3d_empty_raises() -> None:
    with pytest.raises(ValueError):
        compute_bbox_3d(np.zeros((4, 4, 4), dtype=bool))


def test_cube_bbox_is_cube_and_centered() -> None:
    # A non-cubic box: sides 3,4,4 -> cubed with margin 0.
    box = cube_bbox((2, 5, 3, 7, 4, 8), margin=0.0)
    zmin, zmax, ymin, ymax, xmin, xmax = box
    d, h, w = zmax - zmin, ymax - ymin, xmax - xmin
    assert d == h == w == 4  # max side = 4
    # Centre preserved on each axis (within rounding).
    assert abs((zmin + zmax) / 2 - 3.5) <= 0.5
    assert abs((ymin + ymax) / 2 - 5.0) <= 0.5
    assert abs((xmin + xmax) / 2 - 6.0) <= 0.5


def test_crop_with_padding_3d_zero_pads_off_edge() -> None:
    vol = np.ones((4, 4, 4), dtype=np.uint8)
    # Box starts one voxel before the origin on every axis -> a zero border.
    out = crop_with_padding_3d(vol, (-1, 3, -1, 3, -1, 3))
    assert out.shape == (4, 4, 4)
    assert out[0, :, :].sum() == 0 and out[:, 0, :].sum() == 0 and out[:, :, 0].sum() == 0
    assert out[1:, 1:, 1:].all()


def test_resize_nearest_3d_shape_and_binary() -> None:
    vol = np.zeros((8, 8, 8), dtype=bool)
    vol[2:6, 2:6, 2:6] = True
    out = resize_nearest_3d(vol, (16, 16, 16))
    assert out.shape == (16, 16, 16)
    assert set(np.unique(out)).issubset({False, True})


def test_extract_organ_volume_binary_and_sized() -> None:
    vol = np.zeros((20, 24, 18), dtype=bool)
    vol[5:12, 6:16, 4:12] = True
    out = extract_organ_volume(vol, output_size=SIZE, bbox_margin=0.25)
    assert out.shape == SIZE
    assert out.dtype == bool and out.any()
    assert set(np.unique(out)).issubset({False, True})
