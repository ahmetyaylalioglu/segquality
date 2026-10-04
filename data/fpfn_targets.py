"""False-positive / false-negative correction targets (advisor direction G).

The image-conditioned completion models (E2-E7) learned to IGNORE the input mask and
segment from the CT image alone, so ``recon_dice = dice(output, input) ~= dice(true_GT,
input) = true_dice`` collapsed into a near-tautology. This module re-frames the output as
CORRECTIONS defined *relative to the input (broken) mask*, so the model can no longer
ignore it:

* **FP** (false positive) = voxels the input marks foreground but the clean mask does not
  → voxels to **remove**: ``broken AND NOT clean``.
* **FN** (false negative) = voxels the input misses but the clean mask has
  → voxels to **add**: ``clean AND NOT broken``.

The corrected mask is ``(broken AND NOT FP) OR FN``. Because FP/FN are keyed to the input,
the clean mask is never a network input — it only builds these supervision targets
(:func:`fpfn_targets`). At inference the predicted FP/FN give a GT-free quality signal: the
error volume ``|FP| + |FN|`` (:func:`error_volume`) and an estimated Dice
(:func:`estimated_dice`) of the input against the unknown clean shape.

Pure NumPy/Torch, backend-agnostic (duck-typed on ``torch.Tensor``), side-effect free so it
is unit tested with no dataset and callable inside the collate / eval loops.
"""
from __future__ import annotations

import numpy as np
import torch


def _is_torch(x) -> bool:
    return isinstance(x, torch.Tensor)


def _reduce_dims(x: torch.Tensor) -> tuple[int, ...]:
    """All dims except the batch dim 0 (so per-sample sums over C,D,H,W)."""
    return tuple(range(1, x.dim()))


def fpfn_targets(broken, clean, threshold: float = 0.5):
    """Return ``(fp, fn)`` correction targets relative to the ``broken`` input mask.

    ``FP = broken AND NOT clean`` (remove), ``FN = clean AND NOT broken`` (add). Accepts
    NumPy arrays or Torch tensors of any matching shape, float or bool; both are thresholded
    at ``threshold``. Output matches the input backend: ``float32`` arrays / float tensors in
    ``{0, 1}``.
    """
    if _is_torch(broken):
        b = (broken > threshold).float()
        c = (clean > threshold).float()
        return b * (1.0 - c), c * (1.0 - b)
    b = np.asarray(broken) > threshold
    c = np.asarray(clean) > threshold
    return (b & ~c).astype(np.float32), (c & ~b).astype(np.float32)


def correct_mask(broken, fp, fn, threshold: float = 0.5):
    """Apply predicted corrections: ``corrected = (broken AND NOT FP) OR FN``.

    With the *target* FP/FN this reconstructs the clean mask exactly; with predictions it is
    the model's corrected candidate. Backend follows ``broken`` (bool array / float tensor).
    """
    if _is_torch(broken):
        b = broken > threshold
        f = fp > threshold
        n = fn > threshold
        return ((b & ~f) | n).float()
    b = np.asarray(broken) > threshold
    f = np.asarray(fp) > threshold
    n = np.asarray(fn) > threshold
    return (b & ~f) | n


def error_volume(fp: torch.Tensor, fn: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Per-sample predicted error volume ``|FP| + |FN|`` (higher = worse mask).

    ``fp``/``fn`` are probability maps ``(B, 1, D, H, W)`` (or ``(B, ...)``); thresholded and
    summed over every non-batch dim. This is the primary GT-free quality score.
    """
    f = (fp > threshold).float()
    n = (fn > threshold).float()
    dims = _reduce_dims(f)
    return f.sum(dim=dims) + n.sum(dim=dims)


def estimated_dice(
    broken: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor,
    threshold: float = 0.5, eps: float = 1.0,
) -> torch.Tensor:
    """GT-free estimate of ``dice(broken, clean)`` from predicted FP/FN (higher = better).

    FP is counted only where it overlaps the input (removable voxels) and FN only where the
    input is absent (addable voxels), mirroring the target definitions and keeping the
    estimate consistent even if the net fires outside those regions::

        TP        = |broken| - |FP in broken|
        |clean|_est = |broken| - |FP in broken| + |FN outside broken|
        est_dice  = 2 TP / (|broken| + |clean|_est)

    With the target FP/FN this returns the true Dice exactly (see tests).
    """
    b = (broken > threshold).float()
    f = (fp > threshold).float()
    n = (fn > threshold).float()
    dims = _reduce_dims(b)
    broken_vol = b.sum(dim=dims)
    fp_vol = (b * f).sum(dim=dims)
    fn_vol = ((1.0 - b) * n).sum(dim=dims)
    tp = broken_vol - fp_vol
    gt_vol = broken_vol - fp_vol + fn_vol
    return (2.0 * tp + eps) / (broken_vol + gt_vol + eps)
