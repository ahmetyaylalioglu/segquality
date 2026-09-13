"""SDF-target 3D VQ-VAE (WP-3).

Motivation
----------
WP-2 showed that adding geometry as an external *condition* did not help QC. WP-3
instead bakes geometry into the target itself: the model regresses a **signed
distance function** (SDF) rather than a binary mask. Two payoffs: (1) the boundary
is represented continuously (sub-voxel), and (2) — the main goal — modality
independence, since an SDF is purely geometric and carries no CT/MR intensity.

Why this needs almost no new architecture
------------------------------------------
:class:`models.vae3d_vq.ConvVQVAE3D` already emits a **raw continuous** volume from
``forward`` (its last decoder layer is a bare ``ConvTranspose3d``; the sigmoid lives
only in ``reconstruct``). So the base ``forward`` output is already a valid raw SDF
prediction — no new output head or last-layer change is required. This subclass only
reinterprets that output:

* ``predict_sdf(x)`` — the raw forward output, i.e. the predicted (truncated) SDF.
* ``reconstruct(x)`` — overridden to the **binary bridge** ``sdf < 0 -> inside``, so
  the existing recon_dice / topology / correlation machinery scores this model with
  ZERO changes and rho stays comparable to WP-1/2.

The encoder, quantizer and codebook are inherited verbatim, so codebook diagnostics
(perplexity/usage) remain directly comparable across all phases.
"""
from __future__ import annotations

from torch import Tensor

from models.vae3d_vq import ConvVQVAE3D
from quality.sdf import sdf_to_binary


class ConvVQVAE3D_sdf(ConvVQVAE3D):
    """VQ-VAE whose decoder regresses a signed distance map instead of a binary mask.

    Identical constructor to :class:`ConvVQVAE3D` — no extra parameters. The only
    differences are behavioural (SDF interpretation + binary bridge), so a checkpoint
    trained here loads with the same config block.
    """

    def predict_sdf(self, x: Tensor) -> Tensor:
        """Predicted signed distance map (raw, continuous). Outside +, inside -.

        This is exactly the base ``forward`` logits — no sigmoid — trained against the
        truncated target SDF. Kept as a named method so downstream code reads intent.
        """
        logits, _, _ = self.forward(x)
        return logits

    def reconstruct(self, x: Tensor) -> Tensor:
        """Binary reconstruction via the SDF bridge ``sdf < 0 -> inside`` (float mask).

        Overrides the base sigmoid path so every existing score (recon_dice,
        recon_l1, topology) and the correlation protocol operate unchanged.
        """
        return sdf_to_binary(self.predict_sdf(x))
