"""Reconstruction-error quality scores for a trained shape-VAE.

The Phase-6 hypothesis: a VAE trained only on *clean* masks reconstructs a clean
mask well but a degraded one poorly, so the reconstruction error is a
ground-truth-free proxy for the mask's true Dice. This module defines several
candidate error-to-quality scores (a mini-ablation — we do not know a priori
which correlates best) plus the correlation utilities that judge them.

Conventions
-----------
* Every score takes ``(model, masks)`` where ``masks`` is a ``(B, 1, H, W)``
  float tensor in {0, 1}, and returns a ``(B,)`` float tensor (one score per
  mask). Batched for speed.
* Scores run under ``torch.no_grad`` and assume the model is in eval mode, where
  ``reparameterize`` returns ``mu`` — so scoring is deterministic (the same mask
  always yields the same score, needed for a reproducible evaluation).
* Direction differs per score (higher = better quality for Dice; higher = worse
  for the error scores). ``SCORE_DIRECTIONS`` records the sign so callers/plots
  can orient them; correlation is computed against true Dice regardless of sign.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label as cc_label
from scipy.stats import pearsonr, spearmanr
from torch import Tensor

from models.vae import ConvVAE

# Dice bands for stratified correlation. A single global rho hides whether the
# method separates *bad* masks (the clinically important low band) as well as it
# separates near-perfect ones. Bands are half-open [lo, hi) except the last,
# which is closed so the clean reference (true Dice == 1.0) is included.
DICE_BANDS: tuple[tuple[float, float], ...] = (
    (0.40, 0.55),
    (0.55, 0.70),
    (0.70, 0.85),
    (0.85, 1.00),
)

# +1: higher score means higher quality (correlates positively with true Dice).
# -1: higher score means an error, i.e. lower quality (correlates negatively).
SCORE_DIRECTIONS: dict[str, int] = {
    "recon_dice": +1,
    "recon_bce": -1,
    "recon_l1": -1,
    "latent_kl": -1,
    "topology": -1,
}


def _spatial_dims(x: Tensor) -> tuple[int, ...]:
    return tuple(range(1, x.dim()))


@torch.no_grad()
def recon_dice_score(model: ConvVAE, masks: Tensor) -> Tensor:
    """Hard Dice between the binarised reconstruction and the input mask.

    Higher = the VAE reproduced this mask well = higher quality.
    """
    probs = model.reconstruct(masks)
    pred = (probs > 0.5).float()
    dims = _spatial_dims(masks)
    inter = (pred * masks).sum(dim=dims)
    denom = pred.sum(dim=dims) + masks.sum(dim=dims)
    dice = (2.0 * inter + 1.0) / (denom + 1.0)
    return dice


@torch.no_grad()
def recon_bce_score(model: ConvVAE, masks: Tensor) -> Tensor:
    """Per-mask BCE between reconstruction logits and the input. Higher = worse."""
    logits, _, _ = model.forward(masks)
    bce = F.binary_cross_entropy_with_logits(logits, masks, reduction="none")
    return bce.mean(dim=_spatial_dims(masks))


@torch.no_grad()
def recon_l1_score(model: ConvVAE, masks: Tensor) -> Tensor:
    """Mean |reconstruction_prob - input| (soft, thresholdless). Higher = worse."""
    probs = model.reconstruct(masks)
    return (probs - masks).abs().mean(dim=_spatial_dims(masks))


@torch.no_grad()
def latent_kl_score(model: ConvVAE, masks: Tensor) -> Tensor:
    """Per-mask KL(q(z|mask) || N(0, I)). A mask the VAE finds "unusual" pushes
    its posterior away from the prior. Higher = more atypical = worse."""
    mu, logvar = model._encode_dist(masks)
    return -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)


@torch.no_grad()
def topology_score(model: ConvVAE, masks: Tensor) -> Tensor:
    """|#connected-components(recon) - #connected-components(input)|.

    Dice is blind to topology (the FCSN argument): a fragmented mask can still
    overlap well. The VAE, having learned a single-blob liver prior, tends to
    reconstruct one component, so a fragmented input yields a component-count
    mismatch. Higher = more topological disagreement = worse. Computed on CPU via
    ``scipy.ndimage.label`` (integer connected components, not differentiable).
    """
    pred = (model.reconstruct(masks) > 0.5).cpu().numpy()
    inp = (masks > 0.5).cpu().numpy()
    diffs = np.empty(pred.shape[0], dtype=np.float32)
    for i in range(pred.shape[0]):
        _, n_pred = cc_label(pred[i, 0])
        _, n_inp = cc_label(inp[i, 0])
        diffs[i] = abs(n_pred - n_inp)
    return torch.from_numpy(diffs).to(masks.device)


SCORE_FNS: dict[str, Callable[[ConvVAE, Tensor], Tensor]] = {
    "recon_dice": recon_dice_score,
    "recon_bce": recon_bce_score,
    "recon_l1": recon_l1_score,
    "latent_kl": latent_kl_score,
    "topology": topology_score,
}


def compute_all_scores(model: ConvVAE, masks: Tensor) -> dict[str, np.ndarray]:
    """Run every score on a batch. Returns ``{name: (B,) float array}``.

    Puts the model in eval mode first, so scoring is deterministic (z = mu)
    regardless of how the caller left it.
    """
    model.eval()
    return {name: fn(model, masks).cpu().numpy() for name, fn in SCORE_FNS.items()}


def correlation(scores: np.ndarray, true_dice: np.ndarray) -> dict[str, float]:
    """Spearman rho and Pearson r of a score against true Dice, with p-values.

    Returns NaNs (rather than raising) when there are fewer than three points or
    no variance, so a sparse Dice band never crashes the whole report.
    """
    scores = np.asarray(scores, dtype=np.float64)
    true_dice = np.asarray(true_dice, dtype=np.float64)
    n = int(scores.size)
    nan = {"spearman": float("nan"), "spearman_p": float("nan"),
           "pearson": float("nan"), "pearson_p": float("nan"), "n": n}
    if n < 3 or np.ptp(scores) == 0 or np.ptp(true_dice) == 0:
        return nan
    rho, rho_p = spearmanr(scores, true_dice)
    r, r_p = pearsonr(scores, true_dice)
    return {
        "spearman": float(rho),
        "spearman_p": float(rho_p),
        "pearson": float(r),
        "pearson_p": float(r_p),
        "n": n,
    }


def banded_correlation(
    scores: np.ndarray,
    true_dice: np.ndarray,
    bands: tuple[tuple[float, float], ...] = DICE_BANDS,
) -> list[dict[str, float]]:
    """Correlation within each Dice band (half-open [lo, hi), last band closed).

    The clinically important question — can the method flag a *bad* mask? — lives
    in the low bands, which a single global rho can mask. Each entry carries its
    ``band`` (lo, hi) plus the same fields as :func:`correlation`.
    """
    scores = np.asarray(scores, dtype=np.float64)
    true_dice = np.asarray(true_dice, dtype=np.float64)
    out: list[dict[str, float]] = []
    for i, (lo, hi) in enumerate(bands):
        if i == len(bands) - 1:
            sel = (true_dice >= lo) & (true_dice <= hi)
        else:
            sel = (true_dice >= lo) & (true_dice < hi)
        stats = correlation(scores[sel], true_dice[sel])
        stats["band_lo"] = float(lo)
        stats["band_hi"] = float(hi)
        out.append(stats)
    return out
