"""VAE training objectives for binary shape masks.

The reconstruction term mixes two complementary views of "how wrong is this
mask": ``BCEWithLogits`` scores every pixel independently, while soft Dice scores
region overlap and so rewards correct *boundaries* (the part BCE under-weights
because boundary pixels are a small fraction of the image). Their weighted sum is
the reconstruction loss; the KL term regularises the latent toward N(0, I).

All functions operate on logits (not probabilities) where a network output is
expected, so the numerically stable ``binary_cross_entropy_with_logits`` can be
used. Each term is returned separately as well, so training can log recon / bce /
dice / kl independently.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def soft_dice_loss(probs: Tensor, target: Tensor, eps: float = 1.0) -> Tensor:
    """1 - soft Dice, averaged over the batch.

    ``probs`` are probabilities in [0, 1] (post-sigmoid); ``target`` is a binary
    mask. Overlap is computed per-sample over all non-batch dims. ``eps``
    smooths both numerator and denominator, keeping the loss finite for empty
    masks and its gradient well-behaved.
    """
    dims = tuple(range(1, probs.dim()))
    intersection = (probs * target).sum(dim=dims)
    cardinality = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return (1.0 - dice).mean()


def reconstruction_loss(
    logits: Tensor, target: Tensor, dice_weight: float = 1.0
) -> tuple[Tensor, Tensor, Tensor]:
    """Combined reconstruction loss and its two components.

    Returns ``(recon, bce, dice)`` where ``recon = bce + dice_weight * dice``.
    """
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = soft_dice_loss(torch.sigmoid(logits), target)
    recon = bce + dice_weight * dice
    return recon, bce, dice


def kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
    """Closed-form KL(N(mu, sigma^2) || N(0, I)), averaged over the batch.

    Per-sample KL is summed over latent dims, then averaged over the batch, so
    the value is on the same per-example scale as the reconstruction loss.
    """
    per_sample = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return per_sample.mean()


def vae_loss(
    logits: Tensor,
    target: Tensor,
    mu: Tensor,
    logvar: Tensor,
    beta: float = 1.0,
    dice_weight: float = 1.0,
) -> dict[str, Tensor]:
    """Full VAE loss ``recon + beta * kl`` plus every component for logging.

    Returns a dict with keys ``total, recon, bce, dice, kl``. ``beta`` scales the
    KL term (warmup / beta-VAE); the caller owns the warmup schedule.
    """
    recon, bce, dice = reconstruction_loss(logits, target, dice_weight)
    kl = kl_divergence(mu, logvar)
    total = recon + beta * kl
    return {"total": total, "recon": recon, "bce": bce, "dice": dice, "kl": kl}


def vq_loss(
    logits: Tensor,
    target: Tensor,
    vq_term: Tensor,
    dice_weight: float = 1.0,
) -> dict[str, Tensor]:
    """Full VQ-VAE loss ``recon + vq_term`` plus every component for logging.

    Unlike :func:`vae_loss` there is **no KL term**: a VQ-VAE replaces the KL
    regulariser with the quantizer's codebook + commitment loss (``vq_term``,
    already combined and weighted by the quantizer's ``commitment_cost``). The
    reconstruction term is reused verbatim from :func:`reconstruction_loss`, so the
    2D/3D VAE and the VQ-VAE are trained against the identical shape objective.

    Returns a dict with keys ``total, recon, bce, dice, vq``.
    """
    recon, bce, dice = reconstruction_loss(logits, target, dice_weight)
    total = recon + vq_term
    return {"total": total, "recon": recon, "bce": bce, "dice": dice, "vq": vq_term}
