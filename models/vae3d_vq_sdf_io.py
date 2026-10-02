"""SDF-input / SDF-output 3D VQ-VAE (advisor request: binary -> SDF -> model -> SDF -> binary).

Difference from WP-3 (:class:`models.vae3d_vq_sdf.ConvVQVAE3D_sdf`)
------------------------------------------------------------------
WP-3 fed the encoder a **binary** mask and regressed an SDF (input binary, output
SDF). The advisor asked for the input to *also* be an SDF, following the attached
paper: the binary mask is first turned into the normalised truncated SDF
(``quality.sdf_paper``, eq. 1 — inside negative, outside positive, boundary 0,
clamped to ±δ then scaled to ``[-1, 1]``), and the model both consumes and predicts
that continuous field. The intuition (paper §2.3): an SDF is a smooth shape field, so
the encoder sees the boundary continuously on the input side too, not just the target.

Architecture is inherited verbatim from :class:`ConvVQVAE3D` (one float channel in,
one raw float channel out — the last decoder layer is already a bare transpose conv,
so the output is a valid raw SDF with no head change). Only the *interpretation* is
new:

* ``to_sdf_input(x)`` — binary batch -> normalised SDF (the encoder input). Computed
  on the fly (SciPy EDT on CPU), like WP-3's target.
* ``predict_sdf(x)`` — feed the SDF input through the base forward; the logits are the
  predicted (normalised, truncated) SDF.
* ``reconstruct(x)`` — binary bridge ``sdf < 0 -> inside``, so every existing score
  (recon_dice / recon_l1 / topology) and the correlation protocol run unchanged.

Because the public methods still take a **binary** mask (converting to SDF inside),
the model is a drop-in for the eval machinery that passes binary volumes. The trainer
feeds the SDF directly to ``forward`` (both input and target are the same normalised
SDF — an SDF autoencoder with a discrete bottleneck).
"""
from __future__ import annotations

from torch import Tensor

from models.vae3d_vq import ConvVQVAE3D
from quality.sdf_paper import DELTA, batch_input_sdf, normalized_to_binary


class ConvVQVAE3D_sdf_io(ConvVQVAE3D):
    """VQ-VAE whose encoder INPUT and decoder OUTPUT are both the normalised SDF.

    Same constructor as :class:`ConvVQVAE3D` plus a ``delta`` (TSDF half-width used
    to build/normalise the SDF); a checkpoint loads with the same ``vae3d_vq`` block.
    """

    def __init__(self, *args, delta: float = DELTA, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.delta = float(delta)

    def to_sdf_input(self, x: Tensor) -> Tensor:
        """Binary mask batch ``(B,1,D,H,W)`` -> normalised truncated SDF input."""
        return batch_input_sdf(x, self.delta, device=x.device)

    def predict_sdf(self, x: Tensor) -> Tensor:
        """Predicted normalised SDF for a BINARY input (converts to SDF internally).

        ``x`` is binary; it is mapped to its SDF, encoded/quantised/decoded, and the
        raw logits are returned as the predicted SDF (same normalised scale).
        """
        sdf_in = self.to_sdf_input(x)
        logits, _, _ = self.forward(sdf_in)
        return logits

    def reconstruct(self, x: Tensor) -> Tensor:
        """Binary reconstruction via the SDF bridge ``sdf < 0 -> inside`` (float mask).

        Overrides the base sigmoid path so recon_dice / recon_l1 / topology and the
        whole correlation protocol operate unchanged on a binary input.
        """
        return normalized_to_binary(self.predict_sdf(x))
