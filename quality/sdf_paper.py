"""Normalised truncated SDF (the attached paper's eq. 1) for the SDF-input VQ-VAE.

WP-3 fed the encoder a **binary** mask and only the *target* was an SDF. The advisor
asked for a model whose **input is also an SDF**: binary mask -> SDF -> model -> SDF
-> binary. This module provides that binary->SDF encoding, following the definition
in *Score-Based Generative Models for Medical Image Segmentation* (2303.05966), eq. 1:

    phi(m_ij) = -min(min_{y in dS} ||y - m_ij||, delta)   if m_ij inside  S
              = +min(min_{y in dS} ||y - m_ij||, delta)   if m_ij outside S
              =  0                                          on the boundary dS

i.e. **inside negative, outside positive, boundary 0**, truncated at ``delta`` — the
SAME sign convention as :mod:`quality.sdf` (``signed_distance = edt(~b) - edt(b)``).
The paper additionally *normalises* the truncated SDF; we divide by ``delta`` so the
map lands in ``[-1, 1]`` (a resolution/δ-independent range that suits a conv encoder).

Thin wrapper over :mod:`quality.sdf` — no new distance code, so the sign lock in
``test_sdf.py`` still governs the convention. The binary bridge is unchanged:
``phi < 0 -> inside`` (thresholding at 0, exactly as the paper retrieves ``m``).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from quality.sdf import TRUNCATION, signed_distance, truncate

# Default truncation half-width δ, reused from quality.sdf so WP-3 and this share it.
DELTA: float = TRUNCATION


def normalized_sdf(binary: np.ndarray, delta: float = DELTA) -> np.ndarray:
    """Paper eq. 1 map for one binary volume, normalised to ``[-1, 1]``.

    ``truncate(signed_distance(b), delta) / delta`` — inside negative, outside
    positive, boundary 0, clamped to ±delta then scaled by 1/delta. Sign-preserving,
    so ``normalized_sdf(b) < 0`` recovers the foreground.
    """
    return (truncate(signed_distance(binary), delta) / float(delta)).astype(np.float32)


def normalized_to_binary(sdf: np.ndarray | Tensor):
    """Binary bridge: ``phi < 0 -> inside`` (same threshold-at-0 as the paper)."""
    if isinstance(sdf, Tensor):
        return (sdf < 0).float()
    return (np.asarray(sdf) < 0)


def batch_input_sdf(masks: Tensor, delta: float = DELTA, device: torch.device | None = None) -> Tensor:
    """Normalised truncated SDF for a batch of binary masks (the encoder INPUT).

    ``masks`` is ``(B, 1, D, H, W)`` or ``(B, D, H, W)`` float/bool in {0, 1}. SciPy
    EDT is CPU/NumPy only, so masks go to CPU, are transformed per sample, normalised,
    then moved to ``device`` (defaults to the masks' own device). Shape is preserved.
    """
    if device is None:
        device = masks.device
    squeeze_channel = masks.dim() == 5
    vols = masks[:, 0] if squeeze_channel else masks
    np_vols = vols.detach().cpu().numpy()
    out = np.empty(np_vols.shape, dtype=np.float32)
    for i in range(np_vols.shape[0]):
        out[i] = normalized_sdf(np_vols[i], delta)
    tens = torch.from_numpy(out).to(device)
    return tens.unsqueeze(1) if squeeze_channel else tens
