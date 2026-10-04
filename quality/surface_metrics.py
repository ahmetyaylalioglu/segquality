"""Surface-distance metrics — Average Symmetric Surface Distance (ASSD).

Dice is a volume-overlap metric: it can stay high while the boundary is wrong (the advisor's
edge-error complaint). ASSD measures error *on the surface* — the mean, over both surfaces,
of each surface voxel's distance to the nearest surface voxel of the other mask — so it is
sensitive precisely to the boundary degradations Dice misses.

Definition (``pred``, ``gt`` binary)::

    S_p, S_g            = surface voxels of pred, gt
    d(x, S)             = distance from voxel x to nearest voxel in S
    ASSD = ( Σ_{x∈S_p} d(x, S_g) + Σ_{y∈S_g} d(y, S_p) ) / (|S_p| + |S_g|)

Surfaces are the 1-voxel border (mask minus its erosion); distances come from SciPy's exact
EDT with optional anisotropic ``spacing``. Pure NumPy/SciPy, side-effect free, unit tested
with no dataset. A mask with no surface (empty, or — with ``border_value`` erosion — full)
yields ``nan`` (no surface to measure), so callers can ``nanmean`` over a batch.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def surface_voxels(mask: np.ndarray) -> np.ndarray:
    """1-voxel-thick boundary of a binary volume: ``mask AND NOT erosion(mask)``.

    ``border_value=0`` treats the volume edge as background, so foreground touching the
    volume boundary is correctly part of the surface. An empty mask has an empty surface.
    """
    m = np.asarray(mask) > 0.5
    if not m.any():
        return np.zeros_like(m)
    eroded = binary_erosion(m, border_value=0)
    return m & ~eroded


def surface_distances(
    pred: np.ndarray, gt: np.ndarray, spacing: float | tuple[float, ...] = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(d_pred->gt, d_gt->pred)`` arrays of per-surface-voxel distances.

    Each array holds, for every surface voxel of one mask, its distance to the nearest
    surface voxel of the other. Empty if either surface is empty.
    """
    sp = surface_voxels(pred)
    sg = surface_voxels(gt)
    if not sp.any() or not sg.any():
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    dt_to_gt = distance_transform_edt(~sg, sampling=spacing)
    dt_to_pred = distance_transform_edt(~sp, sampling=spacing)
    return dt_to_gt[sp], dt_to_pred[sg]


def assd(pred: np.ndarray, gt: np.ndarray, spacing: float | tuple[float, ...] = 1.0) -> float:
    """Average Symmetric Surface Distance between two binary volumes (lower = better edges).

    Returns ``nan`` when either mask has no surface (empty/full) — no boundary to compare.
    """
    d_pg, d_gp = surface_distances(pred, gt, spacing)
    if d_pg.size == 0 or d_gp.size == 0:
        return float("nan")
    return float((d_pg.sum() + d_gp.sum()) / (d_pg.size + d_gp.size))


def batch_assd(
    preds: np.ndarray, gts: np.ndarray, spacing: float | tuple[float, ...] = 1.0
) -> np.ndarray:
    """Per-sample ASSD over ``(B, D, H, W)`` (or ``(B, 1, D, H, W)``) batches → ``(B,)``."""
    p = np.asarray(preds)
    g = np.asarray(gts)
    if p.ndim == 5:
        p = p[:, 0]
    if g.ndim == 5:
        g = g[:, 0]
    return np.array([assd(p[i], g[i], spacing) for i in range(p.shape[0])], dtype=np.float64)
