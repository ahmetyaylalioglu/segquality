"""3D primitive degradations — the volumetric siblings of the 2D primitives.

Phase 8. Only the four *geometric* primitives need a 3D form; the other two
(``erode`` / ``dilate`` via SciPy's N-dimensional morphology, and
``boundary_noise`` via an N-dimensional signed distance field + Gaussian noise)
are already dimension-agnostic and are re-exported unchanged from
:mod:`data.degradations`.

The engine that drives these (:class:`data.degradations.SyntheticDegrader`) is
itself dimension-agnostic: it only dispatches through the ``primitives`` dict it
was handed. Passing :data:`PRIMITIVES_3D` (below) makes the SAME calibrate /
severity-search / chain / guard logic operate on volumes. Crucially the keys are
identical to the 2D set (``erode``/``dilate``/``remove_blob``/
``add_false_positive``/``boundary_noise``/``translate``), so the name-based
module constants in degradations.py (``_STRONG_TYPES`` / ``_CHAIN_TYPES`` /
``_LIGHT_ONLY_TYPES``) keep working verbatim.

Consistency rule (Phase 7/8): every consumer — degradation, VAE reconstruction,
true-Dice measurement — operates in the SAME squashed 64^3 space that
``RAOSVolumes`` already returns. Degrading that volume directly and measuring
Dice on it therefore keeps all three in one coordinate frame (the 3D mirror of
the 2D ``degradation.space:"resized"`` rule).

Volume-vs-area geometry note
----------------------------
The blob primitives dissolve a *ball* whose volume scales as r^3, not a disk
whose area scales as r^2. So removing/adding "the same fraction of the organ"
needs a radius solved from ``r = (3 * frac * V / (4*pi))^(1/3)``. Because volume
grows faster with radius than area does, a 3D blob of a given radius perturbs a
smaller *fraction* of a compact organ than the 2D disk did — this shifts each
3D primitive's reachable [dice_min, dice_max] relative to 2D, which is exactly
why the evaluation script reports the post-calibration ranges.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.ndimage import shift as ndi_shift

from data.degradations import (
    MAX_BLOB_FRAC,
    MAX_FP_FRAC,
    MAX_TRANSLATE_FRAC,
    _FP_GAP_ITERS,
    _as_input,
    _to_bool,
    boundary_noise,
    dilate,
    erode,
)


# --------------------------------------------------------------------------
# 3D geometric helpers
# --------------------------------------------------------------------------
def _ball(
    shape: tuple[int, int, int],
    center: tuple[float, float, float],
    radius: float,
) -> np.ndarray:
    """Solid sphere mask — the 3D analogue of :func:`data.degradations._disk`."""
    cz, cy, cx = center
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    return (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2


def _blob_radius(fraction: float, voxels: int) -> float:
    """Radius of a ball whose volume is ``fraction`` of ``voxels`` foreground.

    Inverse of V = 4/3 * pi * r^3. Floored at 1.0 so a tiny severity still edits
    at least one voxel (mirrors the 2D disk's ``max(1.0, ...)`` floor).
    """
    return max(1.0, ((3.0 * fraction * voxels) / (4.0 * np.pi)) ** (1.0 / 3.0))


# --------------------------------------------------------------------------
# 3D primitive degradations: (mask, severity, rng) -> mask, values in {0, 1}
# --------------------------------------------------------------------------
def remove_blob_3d(
    mask: np.ndarray, severity: float, rng: np.random.Generator
) -> np.ndarray:
    """Erase a random-located ball from the volume (can break topology)."""
    binary = _to_bool(mask)
    fg = np.argwhere(binary)
    if fg.size == 0:
        return _as_input(mask, binary)
    radius = _blob_radius(severity * MAX_BLOB_FRAC, len(fg))
    center = fg[rng.integers(len(fg))]
    out = binary & ~_ball(binary.shape, tuple(center), radius)
    return _as_input(mask, out)


def add_false_positive_3d(
    mask: np.ndarray, severity: float, rng: np.random.Generator
) -> np.ndarray:
    """Add a disconnected spurious ball outside the volume."""
    binary = _to_bool(mask)
    voxels = max(int(binary.sum()), 1)
    radius = _blob_radius(severity * MAX_FP_FRAC, voxels)
    # Keep a gap so the blob stays disconnected from the true mask.
    forbidden = binary_dilation(binary, iterations=_FP_GAP_ITERS)
    background = np.argwhere(~forbidden)
    if background.size == 0:
        return _as_input(mask, binary)
    center = background[rng.integers(len(background))]
    out = binary | _ball(binary.shape, tuple(center), radius)
    return _as_input(mask, out)


def translate_3d(
    mask: np.ndarray, severity: float, rng: np.random.Generator
) -> np.ndarray:
    """Shift the volume by a severity-scaled amount in a random 3D direction."""
    binary = _to_bool(mask).astype(np.uint8)
    side = min(binary.shape)
    magnitude = severity * MAX_TRANSLATE_FRAC * side
    # Random unit vector on the 3D sphere (isotropic direction), then scale.
    vec = rng.standard_normal(3)
    vec = vec / (np.linalg.norm(vec) or 1.0) * magnitude
    shifted = ndi_shift(
        binary,
        (float(vec[0]), float(vec[1]), float(vec[2])),
        order=0,
        mode="constant",
        cval=0,
    )
    return _as_input(mask, shifted > 0)


# Same keys as data.degradations.PRIMITIVES (name-based module constants rely on
# this); erode/dilate/boundary_noise are the re-used N-dimensional 2D functions.
PRIMITIVES_3D = {
    "erode": erode,
    "dilate": dilate,
    "remove_blob": remove_blob_3d,
    "add_false_positive": add_false_positive_3d,
    "boundary_noise": boundary_noise,
    "translate": translate_3d,
}
