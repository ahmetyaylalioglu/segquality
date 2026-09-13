"""Signed distance function (SDF) helpers for the SDF-target VQ-VAE (WP-3).

Sign convention (LOCKED — the advisor's reference)
--------------------------------------------------
``sdf = outside_dist - inside_dist`` so that **outside is POSITIVE, inside is
NEGATIVE, the boundary is ~0**. The binary bridge is therefore ``sdf < 0 -> inside``.

WARNING: ``data/degradations.py``'s ``boundary_noise`` uses the OPPOSITE convention
(``edt(binary) - edt(~binary)`` => inside positive). That code is for degradation and
is left untouched; WP-3 uses this module end to end. ``test_sdf.py`` asserts the sign
here (a "convention lock") so a future mix-up with the degradation code is caught.

Everything is pure NumPy/SciPy + a thin Torch wrapper, side-effect free, so it is
unit tested with no dataset and can be called on-the-fly inside a training loop.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch import Tensor

# Truncation half-width (TSDF ±TRUNCATION). Chosen from the real data: |sdf| p99 ~34
# with huge outside distances, so ±3 keeps the whole boundary band and focuses the
# regression there. Clamping to ±3 preserves sign, so the binary bridge is unaffected.
TRUNCATION: float = 3.0


def signed_distance(binary: np.ndarray) -> np.ndarray:
    """Signed distance map of a binary volume — outside +, inside -, boundary ~0.

    Uses the advisor's exact formula ``edt(~b) - edt(b)``. An all-background or
    all-foreground volume has no boundary; EDT then returns 0 everywhere on the
    empty side, which is the correct degenerate behaviour (no gradient to a
    non-existent surface).
    """
    b = np.asarray(binary) > 0.5
    outside = distance_transform_edt(~b)
    inside = distance_transform_edt(b)
    return outside - inside


def truncate(sdf: np.ndarray | Tensor, t: float = TRUNCATION):
    """Clamp an SDF to ``[-t, +t]`` (TSDF). Sign-preserving, so the bridge is intact."""
    if isinstance(sdf, Tensor):
        return sdf.clamp(-t, t)
    return np.clip(sdf, -t, t)


def sdf_to_binary(sdf: np.ndarray | Tensor):
    """Binary bridge: ``sdf < 0 -> inside`` (foreground). Returns same type as input.

    This is the heart of the WP-3 QC comparison: it turns a predicted SDF back into a
    binary mask so the existing recon_dice/topology scores and the whole correlation
    protocol run unchanged.
    """
    if isinstance(sdf, Tensor):
        return (sdf < 0).float()
    return (np.asarray(sdf) < 0)


def batch_target_sdf(masks: Tensor, t: float = TRUNCATION, device: torch.device | None = None) -> Tensor:
    """Truncated target SDF for a batch of binary masks.

    ``masks`` is ``(B, 1, D, H, W)`` (or ``(B, D, H, W)``) float/bool in {0, 1}. SciPy
    EDT is CPU/NumPy only, so masks go to CPU, are transformed per sample, truncated,
    then moved to ``device`` (defaults to the masks' own device). Output matches the
    input's ``(B, 1, D, H, W)`` / ``(B, D, H, W)`` shape.
    """
    if device is None:
        device = masks.device
    squeeze_channel = masks.dim() == 5
    vols = masks[:, 0] if squeeze_channel else masks
    np_vols = vols.detach().cpu().numpy()
    out = np.empty(np_vols.shape, dtype=np.float32)
    for i in range(np_vols.shape[0]):
        out[i] = truncate(signed_distance(np_vols[i]), t)
    tens = torch.from_numpy(out).to(device)
    return tens.unsqueeze(1) if squeeze_channel else tens
