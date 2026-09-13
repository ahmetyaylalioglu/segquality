"""Unit tests for the SDF helpers (WP-3), synthetic volumes only.

The most important test is the CONVENTION LOCK (``test_sign_convention_sphere``):
outside must be POSITIVE, inside NEGATIVE. ``data/degradations.py`` uses the opposite
sign for a different purpose; this lock catches any future mix-up that would silently
invert the whole SDF pipeline.
"""
from __future__ import annotations

import numpy as np
import torch

from quality.sdf import (
    TRUNCATION,
    batch_target_sdf,
    sdf_to_binary,
    signed_distance,
    truncate,
)


def _sphere(size: int = 24, radius: float = 6.0) -> np.ndarray:
    c = (size - 1) / 2.0
    zz, yy, xx = np.mgrid[0:size, 0:size, 0:size]
    return ((zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2) <= radius**2


# -- CONVENTION LOCK ---------------------------------------------------------
def test_sign_convention_sphere() -> None:
    """Outside POSITIVE, inside NEGATIVE, centre most negative — advisor convention."""
    b = _sphere()
    sdf = signed_distance(b)
    c = (b.shape[0] - 1) // 2
    assert sdf[c, c, c] < 0, "sphere centre must be negative (inside)"
    assert sdf[b].max() < 0 or np.isclose(sdf[b].max(), 0, atol=1.0)  # all inside <= ~0
    assert sdf[b].min() < -1.0, "deep interior must be clearly negative"
    assert sdf[~b].min() >= 0.0, "every outside voxel must be non-negative"
    assert sdf[~b].max() > 1.0, "far outside must be clearly positive"
    # centre is the single most-negative point
    assert sdf[c, c, c] == sdf.min()


def test_boundary_near_zero() -> None:
    sdf = signed_distance(_sphere())
    assert np.mean(np.abs(sdf) <= 1.0) > 0.0  # a boundary shell of near-zero values exists


# -- truncation --------------------------------------------------------------
def test_truncate_clamps_and_preserves_sign() -> None:
    sdf = signed_distance(_sphere())
    t = truncate(sdf, 3.0)
    assert t.min() >= -3.0 and t.max() <= 3.0
    # sign preserved everywhere -> bridge unaffected
    assert np.array_equal(np.sign(np.where(sdf == 0, 0, sdf)), np.sign(np.where(t == 0, 0, t)))


def test_truncate_tensor() -> None:
    x = torch.linspace(-10, 10, 21)
    assert torch.allclose(truncate(x, 3.0), x.clamp(-3, 3))


# -- binary bridge -----------------------------------------------------------
def test_bridge_roundtrip_exact() -> None:
    b = _sphere()
    recovered = sdf_to_binary(signed_distance(b))
    assert np.array_equal(recovered, b)


def test_bridge_roundtrip_after_truncation() -> None:
    """Clamping to ±3 preserves sign, so the bridge still recovers the mask exactly."""
    b = _sphere()
    recovered = sdf_to_binary(truncate(signed_distance(b), 3.0))
    assert np.array_equal(recovered, b)


def test_bridge_tensor_returns_float_mask() -> None:
    sdf = torch.tensor([[-2.0, 0.5], [-0.1, 3.0]])
    out = sdf_to_binary(sdf)
    assert out.dtype == torch.float32
    assert torch.equal(out, torch.tensor([[1.0, 0.0], [1.0, 0.0]]))


# -- batch target ------------------------------------------------------------
def test_batch_target_shape_and_range() -> None:
    b = _sphere(size=16, radius=4.0)
    batch = torch.from_numpy(b.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1,1,16,16,16)
    tsdf = batch_target_sdf(batch, TRUNCATION)
    assert tsdf.shape == batch.shape
    assert float(tsdf.min()) >= -TRUNCATION and float(tsdf.max()) <= TRUNCATION
    # bridge on the truncated target recovers the mask
    assert torch.equal(sdf_to_binary(tsdf), batch)


def test_smooth_l1_runs_on_sdf() -> None:
    import torch.nn.functional as F

    b = _sphere(size=16, radius=4.0)
    batch = torch.from_numpy(b.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    target = batch_target_sdf(batch, TRUNCATION)
    pred = torch.zeros_like(target, requires_grad=True)
    loss = F.smooth_l1_loss(pred, target)
    loss.backward()
    assert torch.isfinite(loss) and loss.item() > 0.0
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
