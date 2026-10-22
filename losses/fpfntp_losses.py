"""Losses for the recurrent FP/FN/TP refiner (2026-10-20 advisor direction).

The advisor specified two loss families for the stage-2 refiner, applied at EACH recurrent
iteration:

1. **Per-map BCE** -- an independent binary cross-entropy on FP, FN and TP against their
   targets from :func:`data.fpfntp_targets.fpfntp_targets`.
2. **Image-aware MSE** -- the model must "learn from the grayscale image", so every predicted
   map is multiplied by the IMAGE and by the GT mask and matched to the same product of the
   target map, via MSE, both per-map and combined::

       per-map image term : sum_X  MSE( Im * X_hat , Im * X_target )
       per-map GT term    : sum_X  MSE( GT * X_hat , GT * X_target )     X in {FP, FN, TP}
       combined image term: MSE( (TP_hat + FN_hat) * Im , G * Im )
       combined GT term   : MSE( (TP_hat + FN_hat) * GT , G * GT )

   ``TP_hat + FN_hat`` is the reconstructed clean mask (``G = TP + FN``). The advisor called
   this a "GAN" term but confirmed **no discriminator network is required** -- it is an
   adversarial-style consistency matched with MSE. An optional PatchGAN3D hinge term is provided
   separately (:func:`adversarial_terms`) for later experiments; it is OFF by default.

Pure Torch, side-effect free, head-agnostic; the training loop owns the optimisation.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class FPFNTPLossWeights:
    """Relative weights of the refiner loss components (per iteration).

    ``dice`` is a soft-Dice on the reconstructed clean mask (soft ``TP OR FN``) vs the clean
    mask: it directly optimises the metric the advisor tracks and keeps the recurrent refiner
    from collapsing the union (per-map BCE alone on sparse FP/FN does not pin the union Dice).
    """

    bce: float = 1.0
    dice: float = 1.0
    mse_per_map_image: float = 1.0
    mse_per_map_gt: float = 1.0
    mse_combined_image: float = 1.0
    mse_combined_gt: float = 1.0


def _bce(prob: Tensor, target: Tensor) -> Tensor:
    """BCE on a probability map (clamped for numerical safety)."""
    return F.binary_cross_entropy(prob.clamp(1e-6, 1.0 - 1e-6), target)


def _soft_union(fn: Tensor, tp: Tensor) -> Tensor:
    """Probabilistic OR ``TP + FN - TP*FN`` -- the soft reconstructed clean mask."""
    return fn + tp - fn * tp


def soft_dice_loss(fn: Tensor, tp: Tensor, clean: Tensor, eps: float = 1.0) -> Tensor:
    """``1 - soft_dice(TP OR FN, clean)`` summed over all non-batch dims, mean over batch."""
    recon = _soft_union(fn, tp)
    dims = tuple(range(1, recon.dim()))
    inter = (recon * clean).sum(dim=dims)
    card = recon.sum(dim=dims) + clean.sum(dim=dims)
    return (1.0 - (2.0 * inter + eps) / (card + eps)).mean()


def per_map_bce(
    fp: Tensor, fn: Tensor, tp: Tensor,
    fp_t: Tensor, fn_t: Tensor, tp_t: Tensor,
) -> Tensor:
    """Sum of independent BCE on FP, FN and TP."""
    return _bce(fp, fp_t) + _bce(fn, fn_t) + _bce(tp, tp_t)


def image_aware_mse(
    fp: Tensor, fn: Tensor, tp: Tensor,
    fp_t: Tensor, fn_t: Tensor, tp_t: Tensor,
    image: Tensor, clean: Tensor,
    w: FPFNTPLossWeights = FPFNTPLossWeights(),
) -> dict[str, Tensor]:
    """Return the four image-aware MSE terms (per-map image/GT, combined image/GT).

    ``image`` and ``clean`` are ``(B, 1, D, H, W)`` grayscale CT and the clean (GT) mask. Keys:
    ``mse_per_map_image``, ``mse_per_map_gt``, ``mse_combined_image``, ``mse_combined_gt``.
    """
    # per-map terms: each predicted map vs its target, weighted by image / GT
    pm_img = (
        F.mse_loss(image * fp, image * fp_t)
        + F.mse_loss(image * fn, image * fn_t)
        + F.mse_loss(image * tp, image * tp_t)
    )
    pm_gt = (
        F.mse_loss(clean * fp, clean * fp_t)
        + F.mse_loss(clean * fn, clean * fn_t)
        + F.mse_loss(clean * tp, clean * tp_t)
    )
    # combined terms: reconstructed clean mask (TP + FN) vs GT, weighted by image / GT
    recon = (tp + fn).clamp(0.0, 1.0)
    comb_img = F.mse_loss(recon * image, clean * image)
    comb_gt = F.mse_loss(recon * clean, clean * clean)
    return {
        "mse_per_map_image": w.mse_per_map_image * pm_img,
        "mse_per_map_gt": w.mse_per_map_gt * pm_gt,
        "mse_combined_image": w.mse_combined_image * comb_img,
        "mse_combined_gt": w.mse_combined_gt * comb_gt,
    }


def refiner_iteration_loss(
    fp: Tensor, fn: Tensor, tp: Tensor,
    fp_t: Tensor, fn_t: Tensor, tp_t: Tensor,
    image: Tensor, clean: Tensor,
    w: FPFNTPLossWeights = FPFNTPLossWeights(),
) -> tuple[Tensor, dict[str, Tensor]]:
    """Full single-iteration loss: weighted per-map BCE + the four image-aware MSE terms.

    Returns ``(total, components)`` where ``components`` holds every scalar term (including
    ``bce``) for logging.
    """
    bce = w.bce * per_map_bce(fp, fn, tp, fp_t, fn_t, tp_t)
    dice = w.dice * soft_dice_loss(fn, tp, clean)
    mse = image_aware_mse(fp, fn, tp, fp_t, fn_t, tp_t, image, clean, w)
    components = {"bce": bce, "dice": dice, **mse}
    total = bce + dice + sum(mse.values())
    return total, components


def cascade_loss(
    outputs: list[tuple[Tensor, Tensor, Tensor]],
    fp_t: Tensor, fn_t: Tensor, tp_t: Tensor,
    image: Tensor, clean: Tensor,
    w: FPFNTPLossWeights = FPFNTPLossWeights(),
) -> tuple[Tensor, list[dict[str, Tensor]]]:
    """Accumulate :func:`refiner_iteration_loss` over every recurrent iteration.

    ``outputs`` is the per-iteration ``(FP, FN, TP)`` list from
    :meth:`models.vae3d_cascade_tp.CascadeFPFNTP.forward`. Returns ``(total, per_iter)`` where
    ``per_iter`` is the list of per-iteration component dicts.
    """
    total = outputs[0][0].new_zeros(())
    per_iter: list[dict[str, Tensor]] = []
    for fp, fn, tp in outputs:
        it_total, comps = refiner_iteration_loss(fp, fn, tp, fp_t, fn_t, tp_t, image, clean, w)
        total = total + it_total
        per_iter.append(comps)
    return total, per_iter


def adversarial_terms(
    discriminator,
    outputs: list[tuple[Tensor, Tensor, Tensor]],
    image: Tensor,
    clean: Tensor,
) -> dict[str, Tensor]:
    """OPTIONAL PatchGAN3D hinge terms on the image*recon products (OFF by default).

    Provided for later experiments: the discriminator judges ``(TP + FN) * image`` (fake) vs
    ``clean * image`` (real) on the final iteration. Returns ``{"g": generator_loss,
    "d": discriminator_loss}`` with the VQGAN hinge objective. Not called by the default loop.
    """
    _, fn, tp = outputs[-1]
    recon = (tp + fn).clamp(0.0, 1.0)
    fake = recon * image
    real = clean * image
    d_real = discriminator(real)
    d_fake = discriminator(fake.detach())
    d_loss = 0.5 * (F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean())
    g_loss = -discriminator(fake).mean()
    return {"g": g_loss, "d": d_loss}
