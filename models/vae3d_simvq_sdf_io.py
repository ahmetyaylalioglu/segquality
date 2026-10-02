"""SimVQ + SDF-input/SDF-output 3D VQ-VAE (baseline experiment E1).

Combines two prior, independent variants under one class:

* the **SDF-io interpretation** (:class:`models.vae3d_vq_sdf_io.ConvVQVAE3D_sdf_io`):
  the binary mask is turned into the normalised truncated SDF (``quality.sdf_paper``,
  eq. 1), and the model both consumes and predicts that continuous field; the binary
  bridge ``sdf < 0 -> inside`` restores a mask for every existing score.
* the **SimVQ quantizer** (:class:`models.vae3d_simvq.ConvVQVAE3D_SimVQ`): the
  collapse-curing frozen Gaussian base codebook ``C`` + one learnable linear ``W``, so
  the effective codebook ``C @ W`` moves jointly (near-full utilization).

Advisor request ("SDF-io'yu SimVQ ile eğitelim"): keep the exact SDF-in/SDF-out data
pipeline of WP-3's SDF-io run but swap the discrete bottleneck for SimVQ, so its
mask-only QC numbers are directly comparable to standard-VQ SDF-io (recon_dice
rho 0.832) and to plain SimVQ (recon_dice 0.843) under the byte-identical exam.

Architecture is inherited verbatim from :class:`ConvVQVAE3D_SimVQ` (one float channel
in, one raw float channel out). Only the *interpretation* of that channel is new — the
same three methods as ``ConvVQVAE3D_sdf_io``, so the eval machinery that passes binary
volumes is unchanged. The one downstream deviation is ``quant_error``: like every
SimVQ score it must read the EFFECTIVE codebook ``C @ W`` (there is no
``quantizer.embedding``), handled in the eval wrapper.
"""
from __future__ import annotations

from torch import Tensor

from models.vae3d_simvq import ConvVQVAE3D_SimVQ
from quality.sdf_paper import DELTA, batch_input_sdf, normalized_to_binary


class ConvVQVAE3D_SimVQ_sdf_io(ConvVQVAE3D_SimVQ):
    """SimVQ-VAE whose encoder INPUT and decoder OUTPUT are both the normalised SDF.

    Same constructor as :class:`ConvVQVAE3D_SimVQ` plus a ``delta`` (TSDF half-width
    used to build/normalise the SDF); a checkpoint loads with the same ``vae3d_vq`` block.
    """

    def __init__(self, *args, delta: float = DELTA, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.delta = float(delta)

    def to_sdf_input(self, x: Tensor) -> Tensor:
        """Binary mask batch ``(B,1,D,H,W)`` -> normalised truncated SDF input."""
        return batch_input_sdf(x, self.delta, device=x.device)

    def predict_sdf(self, x: Tensor) -> Tensor:
        """Predicted normalised SDF for a BINARY input (converts to SDF internally)."""
        sdf_in = self.to_sdf_input(x)
        logits, _, _ = self.forward(sdf_in)
        return logits

    def reconstruct(self, x: Tensor) -> Tensor:
        """Binary reconstruction via the SDF bridge ``sdf < 0 -> inside`` (float mask)."""
        return normalized_to_binary(self.predict_sdf(x))
