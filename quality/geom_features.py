"""Geometric features extracted from a binary mask, for VQ-VAE conditioning (WP-2).

What these are — and are NOT
----------------------------
Every feature here is computed on the **64^3 squashed** binary volume (the space
all training/eval/degradation live in; see ``data/volume_transforms.py``). They
split into two honestly-different groups:

* ``topology`` — ``n_components`` and ``largest_cc_fraction``. These are
  **squash-invariant** (an anisotropic resize cannot merge or split connected
  components, nor change which fraction of voxels the largest blob holds) and
  directly **degradation-sensitive** (fragmentation / speckle changes them). They
  are the anatomically-meaningful, strongest conditioning candidate.

* ``shape`` — ``centroid_{z,y,x}``, ``volume_frac`` and ``elongation``. These are
  **degradation-sensitive shape descriptors of the squashed 64^3 representation**.
  They are NOT anatomical measures: because the 64^3 volume was produced by a
  per-axis (aspect-destroying) squash, a centroid or an elongation here says
  nothing about the organ's real geometry in millimetres. They may still carry QC
  signal (a degraded mask shifts/shrinks/stretches in 64^3 space), which is why we
  test them — but they must be labelled as squashed-space descriptors, not anatomy.

Ablation-ready by construction
------------------------------
``FEATURE_GROUPS`` names the groups; ``resolve_feature_names(groups)`` flattens a
selection into a fixed, ordered feature-name list. This lets a training run use
``["topology"]`` (topology in isolation, the key experiment) or
``["topology", "shape"]`` (full set) with no code change — only the group list
differs. Feature *order* is stable so a model's standardisation buffers and a
checkpoint's column meaning never drift.

Everything is pure NumPy/SciPy and side-effect free, so it is trivially unit
tested with no dataset and can be called from inside a model's ``forward`` on CPU.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import label as cc_label
from torch import Tensor

# Group -> ordered feature names. The flattened order (topology first, then shape)
# is the canonical feature vector layout; do not reorder without retraining.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "topology": ("n_components", "largest_cc_fraction"),
    "shape": ("centroid_z", "centroid_y", "centroid_x", "volume_frac", "elongation"),
}

# Every known feature name in canonical order (topology block, then shape block).
ALL_FEATURE_NAMES: tuple[str, ...] = tuple(
    name for group in FEATURE_GROUPS.values() for name in group
)


def resolve_feature_names(groups: list[str] | tuple[str, ...]) -> list[str]:
    """Flatten a list of group names into an ordered feature-name list.

    Order follows :data:`FEATURE_GROUPS` insertion order (topology before shape),
    not the order groups are requested, so ``["shape", "topology"]`` and
    ``["topology", "shape"]`` yield the identical vector layout.
    """
    unknown = [g for g in groups if g not in FEATURE_GROUPS]
    if unknown:
        raise ValueError(f"unknown feature group(s): {unknown}; known: {list(FEATURE_GROUPS)}")
    return [name for group, names in FEATURE_GROUPS.items() if group in groups for name in names]


def _elongation(coords: np.ndarray) -> float:
    """Ratio of largest to smallest principal spread of the foreground voxels.

    Uses the eigenvalues of the voxel-coordinate covariance (PCA). A roughly
    isotropic blob (e.g. a cube) gives ~1.0; an elongated shape gives >1. Robust
    to degenerate inputs: fewer than two voxels, or a flat/zero-variance axis,
    returns 1.0 (no measurable elongation) rather than dividing by zero.
    """
    if coords.shape[0] < 2:
        return 1.0
    cov = np.cov(coords.astype(np.float64), rowvar=False)
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 0.0, None)
    lo, hi = float(eig[0]), float(eig[-1])
    if hi <= 1e-8:
        return 1.0
    lo = max(lo, 1e-8)
    return float(np.sqrt(hi / lo))


def compute_features_single(vol: np.ndarray) -> dict[str, float]:
    """All geometric features for one binary 3D volume ``vol`` (D, H, W).

    Empty masks are handled explicitly (no foreground): counts/fractions are 0,
    the centroid defaults to the volume centre (0.5 on each normalised axis) and
    elongation defaults to 1.0. Centroids are normalised by the axis length so
    they are grid-size independent and live in [0, 1].
    """
    vol = np.asarray(vol) > 0.5
    d, h, w = vol.shape
    total = float(d * h * w)
    n_fg = int(vol.sum())

    if n_fg == 0:
        return {
            "n_components": 0.0,
            "largest_cc_fraction": 0.0,
            "centroid_z": 0.5,
            "centroid_y": 0.5,
            "centroid_x": 0.5,
            "volume_frac": 0.0,
            "elongation": 1.0,
        }

    labelled, n_comp = cc_label(vol)
    if n_comp > 0:
        sizes = np.bincount(labelled.ravel())[1:]  # drop background label 0
        largest = int(sizes.max())
    else:
        largest = 0

    coords = np.argwhere(vol)  # (n_fg, 3) in (z, y, x)
    cz, cy, cx = coords.mean(axis=0)

    return {
        "n_components": float(n_comp),
        "largest_cc_fraction": float(largest) / float(n_fg),
        "centroid_z": float(cz) / float(max(d - 1, 1)),
        "centroid_y": float(cy) / float(max(h - 1, 1)),
        "centroid_x": float(cx) / float(max(w - 1, 1)),
        "volume_frac": float(n_fg) / total,
        "elongation": _elongation(coords),
    }


def compute_features_batch(
    masks: Tensor, feature_names: list[str], device: torch.device | None = None
) -> Tensor:
    """Selected features for a batch of masks, as a ``(B, len(feature_names))`` tensor.

    ``masks`` is ``(B, 1, D, H, W)`` (or ``(B, D, H, W)``) float/bool in {0, 1}.
    Connected-component labelling is CPU/NumPy only, so masks are pulled to CPU,
    scored per sample, then the resulting vector is moved to ``device`` (defaults
    to the masks' own device). Column order matches ``feature_names`` exactly.
    """
    if device is None:
        device = masks.device
    if masks.dim() == 5:
        vols = masks[:, 0]
    elif masks.dim() == 4:
        vols = masks
    else:
        raise ValueError(f"expected 4D or 5D mask tensor, got shape {tuple(masks.shape)}")
    np_vols = vols.detach().cpu().numpy()
    out = np.empty((np_vols.shape[0], len(feature_names)), dtype=np.float32)
    for i in range(np_vols.shape[0]):
        feats = compute_features_single(np_vols[i])
        out[i] = [feats[name] for name in feature_names]
    return torch.from_numpy(out).to(device)
