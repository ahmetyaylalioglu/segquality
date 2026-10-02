"""Unit tests for the grayscale-CT image pipeline (image-channel workstream).

Pins the choices that make the image channel voxel-aligned with the mask and safe to feed
a conv encoder: (1) HU windowing clips to ``[lo, hi]`` and maps to ``[0, 1]`` (air end -> 0,
dense end -> 1); (2) trilinear resize hits the exact target grid and stays in ``[0, 1]``;
(3) ``extract_organ_image`` uses the SAME mask bbox as ``extract_organ_volume`` so the two
share a coordinate frame, and normalises BEFORE cropping so off-volume padding is air (0);
(4) ``image_path_for`` appends the nnU-Net ``_0000`` suffix ONLY for the Ts split.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from data.raos_image import (
    WINDOW_HI,
    WINDOW_LO,
    extract_organ_image,
    image_path_for,
    resize_trilinear_3d,
    window_normalize,
)
from data.volume_transforms import extract_organ_volume

OUT = (16, 16, 16)
MARGIN = 0.10


def test_window_normalize_maps_hu_to_unit_range() -> None:
    hu = np.array([WINDOW_LO - 500, WINDOW_LO, 0.0, WINDOW_HI, WINDOW_HI + 500], dtype=np.float32)
    out = window_normalize(hu, WINDOW_LO, WINDOW_HI)
    assert out.min() == 0.0 and out.max() == 1.0            # clipped ends
    assert out[0] == 0.0 and out[-1] == 1.0                 # below lo -> 0, above hi -> 1
    mid = (0.0 - WINDOW_LO) / (WINDOW_HI - WINDOW_LO)        # HU 0 maps linearly
    assert abs(float(out[2]) - mid) < 1e-6


def test_resize_trilinear_exact_shape_and_range() -> None:
    vol = np.random.default_rng(0).random((9, 11, 7)).astype(np.float32)
    out = resize_trilinear_3d(vol, OUT)
    assert out.shape == OUT
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_extract_image_aligns_with_mask_bbox() -> None:
    """A CT that is bright exactly inside the liver mask must, after extraction, be brighter
    inside the extracted mask than outside — proving the image is cropped to the mask bbox."""
    rng = np.random.default_rng(1)
    shape = (40, 44, 36)
    mask = np.zeros(shape, dtype=bool)
    mask[12:28, 14:30, 10:24] = True
    # CT: HU high inside the organ, low (air) elsewhere, + a little noise
    ct = np.full(shape, WINDOW_LO, dtype=np.float32)
    ct[mask] = WINDOW_HI
    ct += rng.normal(0, 1.0, size=shape).astype(np.float32)

    ex_mask = extract_organ_volume(mask, OUT, MARGIN)
    ex_img = extract_organ_image(ct, mask, OUT, MARGIN)
    assert ex_img.shape == OUT
    assert ex_img.min() >= 0.0 and ex_img.max() <= 1.0
    inside = ex_img[ex_mask > 0.5].mean()
    outside = ex_img[ex_mask <= 0.5].mean()
    assert inside > outside                                  # image is aligned to the mask


def test_extract_image_pads_offvolume_with_air() -> None:
    """An elongated, edge-hugging mask forces the cube bbox off the array; because the CT is
    normalised before cropping, that padded margin must be air (0), not a bright block."""
    shape = (30, 30, 30)
    mask = np.zeros(shape, dtype=bool)
    # near-full mask: the margin (a fraction of the large extent) pushes the expanded bbox
    # past every array edge, so the crop must zero-pad off-volume -> that pad is air.
    mask[1:29, 1:29, 1:29] = True
    ct = np.full(shape, WINDOW_HI, dtype=np.float32)         # everything bright
    ex_img = extract_organ_image(ct, mask, OUT, MARGIN)
    # A pure in-bounds crop of an all-bright CT would be all 1.0; the low values here are the
    # zero (air) pad entering from off-volume (trilinear resize blurs the exact 0 border up).
    assert ex_img.min() < 0.5
    assert ex_img.max() <= 1.0


def test_image_path_for_ts_suffix_only() -> None:
    root = Path("/data/RAOS")
    tr = {"patient_id": "RAOS_001", "source_subset": "CancerImages(Set1)", "original_split": "Tr"}
    ts = {"patient_id": "RAOS_500", "source_subset": "CancerImages(Set1)", "original_split": "Ts"}
    p_tr = image_path_for(root, tr)
    p_ts = image_path_for(root, ts)
    assert p_tr.parent.name == "imagesTr" and "RAOS_001" in p_tr.name and "_0000" not in p_tr.name
    assert p_ts.parent.name == "imagesTs" and "RAOS_500_0000" in p_ts.name
