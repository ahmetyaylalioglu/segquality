"""Image-conditioned SDF-io 3D VQ-VAE variants: 2-channel [image, SDF(mask)] IN, 1-channel SDF OUT.

The SDF-io analogue of :mod:`models.vae3d_imgcond`. Where those variants feed the encoder
raw ``[grayscale CT, binary mask]`` and predict a binary mask, these apply the SDF-io
interpretation (``quality.sdf_paper`` eq. 1) to the **mask channel only**: the input becomes
``[grayscale CT, normalised SDF(mask)]`` and the single output channel is the predicted
normalised SDF, restored to a mask by the bridge ``sdf < 0 -> inside``.

The grayscale image is a CONDITION, left as-is (never turned into an SDF, never regressed);
only the mask channel round-trips through the SDF representation. This mirrors the mask-only
SDF-io line (:class:`models.vae3d_vq_sdf_io.ConvVQVAE3D_sdf_io` /
:class:`models.vae3d_simvq_sdf_io.ConvVQVAE3D_SimVQ_sdf_io`) but with the extra image channel
that carries the CT evidence needed to complete a broken mask.

The in=2/out=1 decoupling (encoder consumes 2 channels, decoder emits 1) is inherited from
:class:`models.vae3d_imgcond._ImgCondMixin`, exactly as in the binary-head variants.

Two quantizers, matching the mask-only SDF-io line:
* :class:`ConvVQVAE3D_ImgCond_sdf_io` — standard VQ (E4/E6).
* :class:`ConvVQVAE3D_SimVQ_ImgCond_sdf_io` — SimVQ frozen-base + learnable W (E5/E7).
"""
from __future__ import annotations

import torch
from torch import Tensor

from models.vae3d_imgcond import _ImgCondMixin
from models.vae3d_simvq import ConvVQVAE3D_SimVQ
from models.vae3d_vq import ConvVQVAE3D
from quality.sdf_paper import DELTA, batch_input_sdf, normalized_to_binary


class _ImgCondSdfIoMixin(_ImgCondMixin):
    """SDF-io behaviour over a 2-channel ``[image, mask]`` input, 1-channel SDF output.

    The image channel (index 0) passes through untouched; only the mask channel (index 1)
    is mapped to the normalised truncated SDF before encoding. ``predict_sdf`` returns the
    single predicted SDF channel, and ``reconstruct`` applies the binary bridge so every
    existing (binary-consuming) score runs unchanged.
    """

    delta: float

    def to_sdf_input(self, x: Tensor) -> Tensor:
        """``(B,2,D,H,W)`` ``[image, binary mask]`` -> ``[image, normalised SDF(mask)]``.

        Channel 0 (CT image, already in [0,1]) is left as-is; channel 1 (binary mask) is
        replaced by its normalised truncated SDF via :func:`batch_input_sdf`.
        """
        image = x[:, 0:1]
        mask = x[:, 1:2]
        sdf_mask = batch_input_sdf(mask, self.delta, device=x.device)
        return torch.cat([image, sdf_mask], dim=1)

    def predict_sdf(self, x: Tensor) -> Tensor:
        """Predicted normalised SDF (1 channel) for a 2-channel ``[image, binary mask]``."""
        sdf_in = self.to_sdf_input(x)
        logits, _, _ = self.forward(sdf_in)
        return logits

    def reconstruct(self, x: Tensor) -> Tensor:
        """Binary reconstruction via the SDF bridge ``sdf < 0 -> inside`` (float mask)."""
        return normalized_to_binary(self.predict_sdf(x))


class ConvVQVAE3D_ImgCond_sdf_io(_ImgCondSdfIoMixin, ConvVQVAE3D):
    """Standard-VQ SDF-io VAE: 2-channel [image, SDF(mask)] input, 1-channel SDF output."""

    def __init__(self, *args, out_channels: int = 1, delta: float = DELTA, **kwargs) -> None:
        kwargs["in_channels"] = 2
        super().__init__(*args, **kwargs)
        self._shrink_output_head(out_channels)
        self.delta = float(delta)


class ConvVQVAE3D_SimVQ_ImgCond_sdf_io(_ImgCondSdfIoMixin, ConvVQVAE3D_SimVQ):
    """SimVQ SDF-io VAE: 2-channel [image, SDF(mask)] input, 1-channel SDF output."""

    def __init__(self, *args, out_channels: int = 1, delta: float = DELTA, **kwargs) -> None:
        kwargs["in_channels"] = 2
        super().__init__(*args, **kwargs)
        self._shrink_output_head(out_channels)
        self.delta = float(delta)
