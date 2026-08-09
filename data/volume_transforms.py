"""Pure, testable 3D mask transforms: bbox, cubing, zero-padded crop, nearest
resize and binarization — the volumetric analogue of :mod:`data.transforms`.

This is a *separate* module from the 2D ``transforms.py`` (which is untouched):
the 3D pipeline (Phase 7) mirrors the 2D logic one axis higher. The scalar
helpers ``binarize`` / ``assert_binary`` are dimension-agnostic and reused from
the 2D module rather than duplicated.

Key invariant (mirrors 2D): a bbox that runs off a volume edge is **zero-padded**
outward, never shifted inward, so the organ stays centred and the output is
exactly ``output_size``.
"""
from __future__ import annotations

import numpy as np

from data.transforms import assert_binary, binarize

# (zmin, zmax, ymin, ymax, xmin, xmax), max indices exclusive.
BBox3D = tuple[int, int, int, int, int, int]


def compute_bbox_3d(mask: np.ndarray) -> BBox3D:
    """Tight bounding box of the foreground as a 6-tuple (max-exclusive).

    Raises ``ValueError`` if the mask is empty.
    """
    zs, ys, xs = np.where(mask > 0)
    if zs.size == 0:
        raise ValueError("compute_bbox_3d called on an empty mask")
    return (
        int(zs.min()), int(zs.max()) + 1,
        int(ys.min()), int(ys.max()) + 1,
        int(xs.min()), int(xs.max()) + 1,
    )


def cube_bbox(bbox: BBox3D, margin: float) -> BBox3D:
    """Expand ``bbox`` by ``margin`` of its longest side, then make it a cube.

    Centred on the original bbox centre with an equal side on all three axes, so
    a later resize to a cubic target does not distort the mask. The returned box
    may extend outside the volume; that is handled by :func:`crop_with_padding_3d`.
    """
    zmin, zmax, ymin, ymax, xmin, xmax = bbox
    depth = zmax - zmin
    height = ymax - ymin
    width = xmax - xmin
    side = int(round(max(depth, height, width) * (1.0 + margin)))
    side = max(side, 1)
    cz = 0.5 * (zmin + zmax)
    cy = 0.5 * (ymin + ymax)
    cx = 0.5 * (xmin + xmax)
    half = side / 2.0
    new_zmin = int(round(cz - half))
    new_ymin = int(round(cy - half))
    new_xmin = int(round(cx - half))
    return (
        new_zmin, new_zmin + side,
        new_ymin, new_ymin + side,
        new_xmin, new_xmin + side,
    )


def expand_bbox_3d(bbox: BBox3D, margin: float) -> BBox3D:
    """Expand ``bbox`` by ``margin`` **per axis** (each side grown independently).

    Unlike :func:`cube_bbox`, the aspect ratio is *not* squared off: every axis
    keeps its own length (scaled by ``1 + margin``), centred on the original
    centre. A later resize to ``output_size`` then squashes each axis to fill the
    whole target volume, so the organ occupies the cube instead of leaving ~97%
    empty (the "per-axis squash" chosen for Phase 7 — mirrors the 2D pipeline,
    which resizes each slice to 128x128 without preserving aspect).
    """
    zmin, zmax, ymin, ymax, xmin, xmax = bbox
    out: list[int] = []
    for lo, hi in ((zmin, zmax), (ymin, ymax), (xmin, xmax)):
        length = max(int(round((hi - lo) * (1.0 + margin))), 1)
        centre = 0.5 * (lo + hi)
        new_lo = int(round(centre - length / 2.0))
        out.extend((new_lo, new_lo + length))
    return (out[0], out[1], out[2], out[3], out[4], out[5])


def crop_with_padding_3d(mask: np.ndarray, bbox: BBox3D) -> np.ndarray:
    """Crop ``mask`` to ``bbox``, zero-padding any part of the box off the volume.

    The output shape is exactly the box side lengths regardless of how far the
    box extends past the volume borders.
    """
    zmin, zmax, ymin, ymax, xmin, xmax = bbox
    out = np.zeros((zmax - zmin, ymax - ymin, xmax - xmin), dtype=mask.dtype)

    src_z0, src_z1 = max(zmin, 0), min(zmax, mask.shape[0])
    src_y0, src_y1 = max(ymin, 0), min(ymax, mask.shape[1])
    src_x0, src_x1 = max(xmin, 0), min(xmax, mask.shape[2])
    if src_z1 > src_z0 and src_y1 > src_y0 and src_x1 > src_x0:
        dz, dy, dx = src_z0 - zmin, src_y0 - ymin, src_x0 - xmin
        out[dz:dz + (src_z1 - src_z0),
            dy:dy + (src_y1 - src_y0),
            dx:dx + (src_x1 - src_x0)] = mask[src_z0:src_z1, src_y0:src_y1, src_x0:src_x1]
    return out


def resize_nearest_3d(mask: np.ndarray, output_size: tuple[int, int, int]) -> np.ndarray:
    """Nearest-neighbour resize to ``(D, H, W)``. Never trilinear (would blur edges).

    Uses explicit per-axis index mapping so the output is exactly ``output_size``.
    """
    out_d, out_h, out_w = output_size
    in_d, in_h, in_w = mask.shape
    zs = np.minimum((np.arange(out_d) * in_d) // out_d, in_d - 1)
    ys = np.minimum((np.arange(out_h) * in_h) // out_h, in_h - 1)
    xs = np.minimum((np.arange(out_w) * in_w) // out_w, in_w - 1)
    return mask[np.ix_(zs, ys, xs)]


def extract_organ_volume(
    mask: np.ndarray, output_size: tuple[int, int, int], bbox_margin: float
) -> np.ndarray:
    """bbox-centred per-axis squash: expand the bbox per axis, zero-pad, resize.

    The organ's bounding box is expanded by ``bbox_margin`` on each axis
    independently (:func:`expand_bbox_3d`), then resized to ``output_size`` — so
    each axis is squashed to fill the target and the organ occupies the whole
    volume (high occupancy) rather than sitting inside a mostly-empty cube. This
    trades away aspect ratio for occupancy, consistently for every volume, so the
    VAE learns a shape prior in one fixed squashed space. **Every consumer
    (degradation, reconstruction, Dice) must operate in this same squashed space**
    for the reconstruction-error signal to stay meaningful.

    Returns a boolean volume of shape ``output_size``, re-binarized at 0.5 and
    asserted binary — the 3D analogue of :func:`data.transforms.extract_organ_patch`.

    Known limitation (future work): even a *tight* axis-aligned bbox around the
    liver is only ~25% filled — the liver is a wedge/concave organ, so its cross-
    sectional area shrinks fast along z and an axis-aligned box wastes most of its
    volume. An **oriented (PCA-aligned) bounding box** would fit such concave
    organs far more tightly and is the natural next occupancy improvement; it is
    deliberately out of scope for Phase 7 (kept simple: consistent axis-aligned
    squash), but the ~25% ceiling here is direct evidence for why it would help.
    """
    binary = binarize(mask)
    box = expand_bbox_3d(compute_bbox_3d(binary), bbox_margin)
    patch = crop_with_padding_3d(binary, box)
    resized = binarize(resize_nearest_3d(patch, output_size))
    assert_binary(resized)
    return resized
