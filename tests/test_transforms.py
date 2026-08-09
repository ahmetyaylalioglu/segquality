"""Tests for data.transforms — synthetic masks only, no raw data required."""
from __future__ import annotations

import numpy as np

from data.transforms import (
    assert_binary,
    compute_bbox,
    crop_with_padding,
    extract_organ_patch,
    resize_full_slice,
    resize_nearest,
    square_bbox,
)


def make_disk(shape: tuple[int, int], cy: float, cx: float, radius: float) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2


def centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask > 0)
    return float(ys.mean()), float(xs.mean())


def test_compute_bbox_matches_rectangle():
    mask = np.zeros((50, 60), dtype=bool)
    mask[10:20, 15:35] = True
    assert compute_bbox(mask) == (10, 20, 15, 35)


def test_compute_bbox_empty_raises():
    try:
        compute_bbox(np.zeros((10, 10), dtype=bool))
    except ValueError:
        return
    raise AssertionError("expected ValueError on empty mask")


def test_square_bbox_is_square_for_non_square_input():
    # A wide rectangle must still yield an equal-sided (square) box, so a later
    # resize to a square target preserves aspect ratio.
    box = square_bbox((10, 20, 5, 45), margin=0.25)
    ymin, ymax, xmin, xmax = box
    assert (ymax - ymin) == (xmax - xmin)


def test_square_bbox_stays_centred():
    box = (10, 30, 40, 60)  # centre (20, 50)
    ymin, ymax, xmin, xmax = square_bbox(box, margin=0.0)
    assert abs(0.5 * (ymin + ymax) - 20) <= 1
    assert abs(0.5 * (xmin + xmax) - 50) <= 1


def test_resize_nearest_exact_size_and_binary():
    mask = make_disk((40, 40), 20, 20, 12)
    out = resize_nearest(mask, (128, 128))
    assert out.shape == (128, 128)
    assert_binary(out)


def test_crop_with_padding_zero_pads_out_of_bounds():
    mask = np.ones((10, 10), dtype=bool)
    # Box extends 5px past every edge -> exact 20x20 with a 10x10 core of ones.
    out = crop_with_padding(mask, (-5, 15, -5, 15))
    assert out.shape == (20, 20)
    assert out[5:15, 5:15].all()
    assert out.sum() == 100  # only the in-bounds region is filled


def test_extract_organ_patch_output_size_and_binary():
    mask = make_disk((100, 100), 50, 50, 15)
    out = extract_organ_patch(mask, (64, 64), bbox_margin=0.25)
    assert out.shape == (64, 64)
    assert_binary(out)
    assert out.any()


def test_corner_organ_is_zero_padded_and_stays_centred():
    # Organ touches the top edge: the squared+margined box runs off the image and
    # must be zero-padded outward (not shifted inward), so the output is exactly
    # output_size and the organ stays centred. This is the classic edge-bbox bug.
    shape = (100, 100)
    mask = make_disk(shape, cy=12, cx=50, radius=12)  # bbox touches y=0
    out = extract_organ_patch(mask, (64, 64), bbox_margin=0.25)

    assert out.shape == (64, 64)
    assert out.any()
    cy, cx = centroid(out)
    assert abs(cy - 32) <= 6, f"organ not vertically centred: cy={cy}"
    assert abs(cx - 32) <= 6, f"organ not horizontally centred: cx={cx}"


def test_true_corner_organ_does_not_crash_and_fills_output():
    mask = make_disk((100, 100), cy=3, cx=3, radius=10)  # clipped into the corner
    out = extract_organ_patch(mask, (48, 48), bbox_margin=0.25)
    assert out.shape == (48, 48)
    assert out.any()


def test_resize_full_slice_keeps_binary_and_size():
    mask = make_disk((80, 120), 40, 60, 20)
    out = resize_full_slice(mask, (128, 128))
    assert out.shape == (128, 128)
    assert_binary(out)
    assert out.any()
