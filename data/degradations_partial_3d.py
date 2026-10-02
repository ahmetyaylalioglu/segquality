"""Severe / partial 3D degradations — the low-Dice regime the advisor asked for.

The base :data:`data.degradations_3d.PRIMITIVES_3D` set tops out around a modest
amount of corruption (``erode`` caps at ``MAX_MORPH_ITERS`` iterations). To answer
the advisor's question — *how does the VQ-VAE complete a mask given only its
centre, or only a fragment, or after heavy erosion down to Dice ~0.5?* — we need
degradations that reach much lower Dice than the standard set. This module adds
three, all following the SAME primitive contract as ``degradations.py``:

    (mask, severity, rng) -> mask   with values in {0, 1}

so they drop straight into :class:`data.degradations.SyntheticDegrader` (calibrate
/ severity-search / chain / guard) exactly like the existing primitives. Dice is
monotone-decreasing in ``severity`` for each, which is what the degrader's binary
search over severity assumes.

New primitives
--------------
* ``heavy_erode`` — morphological erosion with a much larger iteration budget
  (``MAX_HEAVY_ERODE_ITERS``) so a compact organ can be eroded to Dice ~0.5 and
  below (the "more erosion" request).
* ``center_only`` — keep only a ball around the mask's centroid; ``severity``
  shrinks the kept radius, leaving just the central core (the "only the centre
  region is given" case).
* ``partial`` — cut the mask with a plane through its centroid along a random
  axis; ``severity`` is the fraction of the extent removed (the "only a fragment
  is given" case).

New files only — nothing in ``degradations.py`` / ``degradations_3d.py`` changes.
"""
from __future__ import annotations

import numpy as np

from data.degradations import _as_input, _to_bool
from data.degradations_3d import _ball

# Erosion iteration budget at severity 1.0 for the *heavy* variant. The base
# ``erode`` uses MAX_MORPH_ITERS=12; a 64^3 squashed organ needs a larger budget
# to be driven down to Dice ~0.5 and lower, which is the whole point here.
MAX_HEAVY_ERODE_ITERS = 40


def heavy_erode(mask: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    """Erode with a large iteration budget so Dice can be pushed to ~0.5 and below.

    Identical mechanism to :func:`data.degradations.erode` but with a much bigger
    iteration cap (``MAX_HEAVY_ERODE_ITERS``). SciPy's ``binary_erosion`` is
    N-dimensional, so this works on 3D volumes unchanged.
    """
    from scipy.ndimage import binary_erosion

    iters = max(1, int(round(severity * MAX_HEAVY_ERODE_ITERS)))
    out = binary_erosion(_to_bool(mask), iterations=iters)
    return _as_input(mask, out)


def center_only(mask: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    """Keep only a ball around the centroid; ``severity`` shrinks the kept radius.

    ``severity=0`` keeps the whole mask (radius = full extent), ``severity->1``
    keeps only the central core. The kept radius is ``(1 - severity) * R`` where
    ``R`` is the max centroid-to-foreground distance, so the result is always the
    intersection of the mask with a centred sphere.
    """
    binary = _to_bool(mask)
    fg = np.argwhere(binary)
    if fg.size == 0:
        return _as_input(mask, binary)
    center = fg.mean(axis=0)
    max_r = float(np.sqrt(((fg - center) ** 2).sum(axis=1).max()))
    radius = max(1.0, (1.0 - float(severity)) * max_r)
    out = binary & _ball(binary.shape, tuple(center), radius)
    return _as_input(mask, out)


def partial(mask: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    """Cut the mask with a plane through its centroid along a random axis.

    A random axis is chosen and a random side kept; ``severity`` is the fraction
    of that axis's foreground extent removed from the centroid outward. At
    ``severity=0.5`` roughly one half of the organ is discarded, modelling a mask
    where only a fragment was provided.
    """
    binary = _to_bool(mask)
    fg = np.argwhere(binary)
    if fg.size == 0:
        return _as_input(mask, binary)
    axis = int(rng.integers(binary.ndim))
    coords = fg[:, axis]
    center = float(coords.mean())
    lo, hi = float(coords.min()), float(coords.max())
    keep_high = bool(rng.integers(2))
    if keep_high:
        # remove the lower side: push the cut plane up from the low extent.
        cut = lo + float(severity) * (center - lo)
        mask_axis = fg[:, axis] >= cut
    else:
        cut = hi - float(severity) * (hi - center)
        mask_axis = fg[:, axis] <= cut
    keep = fg[mask_axis]
    out = np.zeros_like(binary)
    if keep.size:
        out[keep[:, 0], keep[:, 1], keep[:, 2]] = True
    return _as_input(mask, out)


# Same contract as PRIMITIVES_3D; these are the *severe* extensions used to reach
# the low-Dice / partial-mask regime for the completion study.
PARTIAL_PRIMITIVES_3D = {
    "heavy_erode": heavy_erode,
    "center_only": center_only,
    "partial": partial,
}
