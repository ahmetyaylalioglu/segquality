"""Unit tests for the severe / partial 3D degradations (low-Dice completion study).

These primitives must (a) obey the same ``(mask, severity, rng) -> mask`` contract
as the base set, (b) monotonically lower Dice as severity rises so the degrader's
severity search works, and (c) reach the low-Dice regime (~0.5 and below) that the
base ``erode`` cannot. Tested on a synthetic solid cube — no dataset dependency.
"""
from __future__ import annotations

import numpy as np

from data.degradations import dice
from data.degradations_partial_3d import (
    PARTIAL_PRIMITIVES_3D,
    center_only,
    heavy_erode,
    partial,
)


def _cube(n: int = 24, pad: int = 6) -> np.ndarray:
    vol = np.zeros((n, n, n), dtype=np.float32)
    vol[pad:-pad, pad:-pad, pad:-pad] = 1.0
    return vol


def _rng() -> np.random.Generator:
    return np.random.default_rng(0)


# -- contract ----------------------------------------------------------------
def test_registry_keys() -> None:
    assert set(PARTIAL_PRIMITIVES_3D) == {"heavy_erode", "center_only", "partial"}


def test_output_binary_and_shape() -> None:
    cube = _cube()
    for fn in PARTIAL_PRIMITIVES_3D.values():
        out = fn(cube, 0.5, _rng())
        assert out.shape == cube.shape
        assert out.dtype == cube.dtype
        assert set(np.unique(out)).issubset({0.0, 1.0})


def test_empty_mask_is_safe() -> None:
    empty = np.zeros((16, 16, 16), dtype=np.float32)
    for fn in PARTIAL_PRIMITIVES_3D.values():
        out = fn(empty, 0.5, _rng())
        assert out.sum() == 0


# -- reaches the low-Dice regime --------------------------------------------
def test_heavy_erode_reaches_low_dice() -> None:
    """Heavy erosion at full severity must drop Dice well below 0.5."""
    cube = _cube()
    out = heavy_erode(cube, 1.0, _rng())
    assert dice(cube, out) < 0.5


def test_center_only_keeps_core() -> None:
    """center_only leaves a non-empty core smaller than the original."""
    cube = _cube()
    out = center_only(cube, 0.6, _rng())
    assert 0 < out.sum() < cube.sum()
    assert dice(cube, out) < 1.0


def test_partial_removes_a_fragment() -> None:
    cube = _cube()
    out = partial(cube, 0.5, _rng())
    assert 0 < out.sum() < cube.sum()
    assert dice(cube, out) < 1.0


# -- monotonicity (severity search relies on this) ---------------------------
def test_dice_monotone_in_severity() -> None:
    cube = _cube()
    for name, fn in PARTIAL_PRIMITIVES_3D.items():
        prev = 1.01
        for s in (0.1, 0.3, 0.5, 0.7, 0.9):
            d = dice(cube, fn(cube, s, np.random.default_rng(0)))
            assert d <= prev + 1e-6, f"{name} not monotone at severity {s}"
            prev = d
