"""Unit tests for the ASSD surface-distance metric (advisor's edge-error check).

Pins the boundary-sensitivity ASSD gives over Dice: identical masks have ASSD 0; a rigidly
shifted mask has ASSD ~ the shift; the metric is symmetric; and a mask with no surface
(empty) yields NaN so a batch can be safely ``nanmean``-ed.
"""
from __future__ import annotations

import numpy as np

from quality.surface_metrics import assd, batch_assd, surface_voxels


def _cube(shape=(20, 20, 20), lo=5, hi=15) -> np.ndarray:
    v = np.zeros(shape, dtype=bool)
    v[lo:hi, lo:hi, lo:hi] = True
    return v


def test_surface_is_hollow_shell() -> None:
    """The surface is the 1-voxel border: its volume is strictly less than the solid."""
    cube = _cube()
    surf = surface_voxels(cube)
    assert surf.sum() > 0
    assert surf.sum() < cube.sum()          # interior removed
    assert not surf[10, 10, 10]             # centre is interior, not surface
    assert surf[5, 10, 10]                  # a face voxel is surface


def test_identical_masks_zero_assd() -> None:
    cube = _cube()
    assert assd(cube, cube) == 0.0


def test_assd_symmetric() -> None:
    a = _cube(lo=5, hi=15)
    b = _cube(lo=6, hi=16)
    assert np.isclose(assd(a, b), assd(b, a))


def test_shift_increases_assd() -> None:
    """A larger rigid shift yields a larger ASSD (monotone in displacement)."""
    base = _cube(lo=5, hi=15)
    near = _cube(lo=6, hi=16)     # shifted by 1
    far = _cube(lo=9, hi=19)      # shifted by 4
    assert assd(base, base) < assd(base, near) < assd(base, far)


def test_empty_mask_is_nan() -> None:
    cube = _cube()
    empty = np.zeros_like(cube)
    assert np.isnan(assd(cube, empty))
    assert np.isnan(assd(empty, cube))


def test_batch_assd_shapes_and_values() -> None:
    a = np.stack([_cube(lo=5, hi=15), _cube(lo=5, hi=15)])
    b = np.stack([_cube(lo=5, hi=15), _cube(lo=7, hi=17)])
    out = batch_assd(a, b)
    assert out.shape == (2,)
    assert out[0] == 0.0 and out[1] > 0.0


def test_batch_assd_accepts_channel_dim() -> None:
    a = _cube()[None, None]      # (1,1,D,H,W)
    b = _cube()[None, None]
    out = batch_assd(a, b)
    assert out.shape == (1,) and out[0] == 0.0
