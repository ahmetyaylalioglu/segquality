"""Pure, testable mask transforms: bbox, squaring, zero-padded crop, nearest
resize and binarization.

Everything here is a plain function over 2D numpy arrays so it can be unit
tested without any dataset or raw data. The dataset composes these; the logic
is never inlined into the ``Dataset`` class.

Key invariant (see tests): a bbox that runs off the image edge is **zero-padded**
outward, never shifted inward. Shifting inward would move the organ off-centre
and silently corrupt the sample; padding keeps the organ centred and the output
exactly ``output_size``.
"""
from __future__ import annotations

import numpy as np

BBox = tuple[int, int, int, int]  # (ymin, ymax, xmin, xmax), ymax/xmax exclusive


def binarize(arr: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Return a boolean mask, ``arr > threshold``."""
    return np.asarray(arr) > threshold


def assert_binary(mask: np.ndarray) -> None:
    """Raise if ``mask`` holds values other than {0, 1} / {False, True}."""
    values = np.unique(mask)
    assert np.all(np.isin(values, (0, 1))), f"mask is not binary: unique={values}"


def compute_bbox(mask: np.ndarray) -> BBox:
    """Tight bounding box of the foreground as (ymin, ymax, xmin, xmax), max-excl.

    Raises ``ValueError`` if the mask is empty.
    """
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        raise ValueError("compute_bbox called on an empty mask")
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def square_bbox(bbox: BBox, margin: float) -> BBox:
    """Expand ``bbox`` by ``margin`` of its longest side, then make it square.

    The result is centred on the original bbox centre and shares its side length
    on both axes (aspect ratio 1:1), so a later resize to a square target does
    not distort the mask. The returned box may extend outside the image; that is
    intentional and handled by :func:`crop_with_padding`.
    """
    ymin, ymax, xmin, xmax = bbox
    height = ymax - ymin
    width = xmax - xmin
    side = int(round(max(height, width) * (1.0 + margin)))
    side = max(side, 1)
    cy = 0.5 * (ymin + ymax)
    cx = 0.5 * (xmin + xmax)
    half = side / 2.0
    new_ymin = int(round(cy - half))
    new_xmin = int(round(cx - half))
    return new_ymin, new_ymin + side, new_xmin, new_xmin + side


def crop_with_padding(mask: np.ndarray, bbox: BBox) -> np.ndarray:
    """Crop ``mask`` to ``bbox``, zero-padding any part of the box off the image.

    The output shape is exactly ``(ymax - ymin, xmax - xmin)`` regardless of how
    far the box extends past the image borders.
    """
    ymin, ymax, xmin, xmax = bbox
    out = np.zeros((ymax - ymin, xmax - xmin), dtype=mask.dtype)

    src_y0, src_y1 = max(ymin, 0), min(ymax, mask.shape[0])
    src_x0, src_x1 = max(xmin, 0), min(xmax, mask.shape[1])
    if src_y1 > src_y0 and src_x1 > src_x0:
        dst_y0 = src_y0 - ymin
        dst_x0 = src_x0 - xmin
        out[dst_y0 : dst_y0 + (src_y1 - src_y0),
            dst_x0 : dst_x0 + (src_x1 - src_x0)] = mask[src_y0:src_y1, src_x0:src_x1]
    return out


def resize_nearest(mask: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize to ``(H, W)``. Never bilinear (would blur edges).

    Uses explicit index mapping so the output is exactly ``output_size``.
    """
    out_h, out_w = output_size
    in_h, in_w = mask.shape
    ys = np.minimum((np.arange(out_h) * in_h) // out_h, in_h - 1)
    xs = np.minimum((np.arange(out_w) * in_w) // out_w, in_w - 1)
    return mask[ys][:, xs]


def extract_organ_patch(
    mask: np.ndarray, output_size: tuple[int, int], bbox_margin: float
) -> np.ndarray:
    """bbox-centred crop: square the (margin-expanded) bbox, zero-pad, resize.

    Returns a boolean mask of shape ``output_size``, re-binarized at 0.5 and
    asserted binary.
    """
    binary = binarize(mask)
    box = square_bbox(compute_bbox(binary), bbox_margin)
    patch = crop_with_padding(binary, box)
    resized = binarize(resize_nearest(patch, output_size))
    assert_binary(resized)
    return resized


def resize_full_slice(mask: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    """full_slice mode: no crop, resize the whole slice to ``output_size``.

    Aspect ratio is not preserved here by design (the whole slice is squashed to
    the target); bbox_centred mode is the aspect-preserving path.
    """
    resized = binarize(resize_nearest(binarize(mask), output_size))
    assert_binary(resized)
    return resized
