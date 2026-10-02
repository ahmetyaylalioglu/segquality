"""Grayscale CT image loading + organ-aligned extraction (image-channel workstream).

The image sibling of the mask-only pipeline. Until now only the label (mask) volume
was ever read; the advisor asked to feed the **grayscale CT** alongside the mask so a
model can complete a broken mask using image evidence. This module adds exactly that,
as NEW code that reuses the existing pure transforms — it never edits
``data/volume_transforms.py`` or ``data/raos_volume.py``.

Alignment is the whole point: the CT is cropped with the **same liver bounding box** the
mask pipeline uses (``expand_bbox_3d(compute_bbox_3d(mask), margin)``), so image voxel
(z,y,x) and mask voxel (z,y,x) refer to the same anatomy after the 64³ squash. Two
image-specific choices differ from the mask path:

* **intensity, not a label** → resize is **trilinear** (mask stays nearest, which would
  shred a continuous field), and values are **HU-windowed then normalised to [0,1]**.
* **off-volume padding** → the CT is normalised to [0,1] *before* cropping, so the air
  end of the window maps to 0 and the existing zero-padding crop (``crop_with_padding_3d``)
  fills any off-volume margin with background/air rather than a water-like block.

RAOS filename quirk (see reference_raos_dataset): Tr/Val images share the label's exact
filename; **Ts images append the nnU-Net ``_0000`` channel suffix** while labels never
do. ``image_path_for`` handles that per ``original_split``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from data.raos_dataset import NIFTI_SUFFIX
from data.transforms import binarize
from data.volume_transforms import compute_bbox_3d, crop_with_padding_3d, expand_bbox_3d

# Liver / soft-tissue HU window (config-overridable). Clip to this range, then map to
# [0,1]: below WINDOW_LO -> 0 (air/lung end), above WINDOW_HI -> 1 (dense end).
WINDOW_LO = -160.0
WINDOW_HI = 240.0


def image_path_for(root: Path, record: dict[str, Any]) -> Path:
    """Reconstruct a patient's CT image path from its index/precompute record.

    Mirrors :func:`data.raos_dataset.label_path_for` but into ``images{split}`` and,
    for the ``Ts`` split only, appends the nnU-Net ``_0000`` channel suffix.
    """
    split = record["original_split"]
    image_dir = "images" + split
    stem = record["patient_id"]
    if split == "Ts":
        stem = stem + "_0000"
    return Path(root) / record["source_subset"] / image_dir / (stem + NIFTI_SUFFIX)


def load_ct_volume(path: Path) -> np.ndarray:
    """Load a NIfTI CT volume as float32 HU (lazy dataobj, no float64 blow-up)."""
    import nibabel as nib

    return np.asanyarray(nib.load(str(path)).dataobj).astype(np.float32)


def window_normalize(volume: np.ndarray, lo: float = WINDOW_LO, hi: float = WINDOW_HI) -> np.ndarray:
    """Clip HU to ``[lo, hi]`` and linearly map to ``[0, 1]`` (air end -> 0)."""
    v = np.clip(volume, lo, hi)
    return ((v - lo) / (hi - lo)).astype(np.float32)


def resize_trilinear_3d(volume: np.ndarray, output_size: tuple[int, int, int]) -> np.ndarray:
    """Trilinear resize a continuous volume to exactly ``(D, H, W)``.

    Uses ``F.interpolate`` (align_corners=False) for an exact output shape; unlike the
    mask's nearest resize this preserves the smooth intensity field. Output is clamped
    back to [0,1] (interpolation can overshoot by ~1e-7).
    """
    t = torch.from_numpy(np.ascontiguousarray(volume)).float()[None, None]
    out = F.interpolate(t, size=tuple(output_size), mode="trilinear", align_corners=False)
    return out[0, 0].clamp_(0.0, 1.0).numpy()


def extract_organ_image(
    image_hu: np.ndarray,
    mask: np.ndarray,
    output_size: tuple[int, int, int],
    bbox_margin: float,
    *,
    window: tuple[float, float] = (WINDOW_LO, WINDOW_HI),
) -> np.ndarray:
    """CT image cropped to the MASK's liver bbox and trilinear-resized to ``output_size``.

    The bounding box is derived from ``mask`` with the SAME
    ``expand_bbox_3d(compute_bbox_3d(...), bbox_margin)`` the mask path uses, so the
    returned image is voxel-aligned with ``extract_organ_volume(mask, ...)``. The CT is
    HU-windowed + normalised to [0,1] BEFORE cropping so the zero-pad margin is air.

    Returns a float32 volume of shape ``output_size`` in [0,1].
    """
    lo, hi = window
    norm = window_normalize(image_hu, lo, hi)
    box = expand_bbox_3d(compute_bbox_3d(binarize(mask)), bbox_margin)
    patch = crop_with_padding_3d(norm, box)
    return resize_trilinear_3d(patch, output_size).astype(np.float32)
