"""FP/FN/TP decomposition targets (advisor direction, 2026-10-20 meeting).

Direction G supervised only the two ERROR maps relative to the broken input mask:
``FP = broken AND NOT clean`` (remove) and ``FN = clean AND NOT broken`` (add). The advisor
now wants the decomposition completed with the **TP** (true-positive) map so the model also
reasons about what the broken mask already got RIGHT::

    TP = broken AND clean          (voxels correct in the input, keep)
    FP = broken AND NOT clean      (voxels to remove)
    FN = clean  AND NOT broken     (voxels to add)

These satisfy ``G = TP + FN`` (clean) and ``M = TP + FP`` (broken), so the corrected mask is
``TP + FN`` -- equivalently the existing ``correct_mask`` result ``(broken AND NOT FP) OR FN``.

This module only ADDS the three-way target; the two-way :func:`data.fpfn_targets.fpfn_targets`
is untouched and still used by the Direction-G line. Pure NumPy/Torch, backend-agnostic,
side-effect free so it is unit testable with no dataset and callable in collate / eval loops.
"""
from __future__ import annotations

import numpy as np
import torch

from data.fpfn_targets import _is_torch


def fpfntp_targets(broken, clean, threshold: float = 0.5):
    """Return ``(fp, fn, tp)`` maps relative to the ``broken`` input and the ``clean`` mask.

    ``FP = broken AND NOT clean`` (remove), ``FN = clean AND NOT broken`` (add),
    ``TP = broken AND clean`` (keep). Accepts NumPy arrays or Torch tensors of any matching
    shape, float or bool; both are thresholded at ``threshold``. Output matches the input
    backend: ``float32`` arrays / float tensors in ``{0, 1}``.
    """
    if _is_torch(broken):
        b = (broken > threshold).float()
        c = (clean > threshold).float()
        return b * (1.0 - c), c * (1.0 - b), b * c
    b = np.asarray(broken) > threshold
    c = np.asarray(clean) > threshold
    return (b & ~c).astype(np.float32), (c & ~b).astype(np.float32), (b & c).astype(np.float32)


def reconstruct_from_fntp(fn, tp, threshold: float = 0.5):
    """Rebuild the clean mask from the ``FN`` and ``TP`` maps: ``G = TP OR FN``.

    This is the TP-head analogue of :func:`data.fpfn_targets.correct_mask`; with the *target*
    FN/TP it reproduces the clean mask exactly, with predictions it is the refined candidate.
    Backend follows ``tp`` (bool array / float tensor).
    """
    if _is_torch(tp):
        n = fn > threshold
        t = tp > threshold
        return (t | n).float()
    n = np.asarray(fn) > threshold
    t = np.asarray(tp) > threshold
    return t | n
