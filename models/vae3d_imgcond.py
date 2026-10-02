"""Image-conditioned 3D VQ-VAE variants: 2-channel [image, mask] IN, 1-channel mask OUT.

The base :class:`~models.vae3d_vq.ConvVQVAE3D` ties the decoder's output channel count to
``in_channels`` (``out_channels = ... + [in_channels]``), so simply passing ``in_channels=2``
would also make it PREDICT 2 channels — wrong, the target is the single mask. These
variants decouple the two: the encoder consumes ``[grayscale CT, mask]`` (2 channels) while
the decoder's final layer is rebuilt to emit **1** channel (the mask logits). Everything
else — encoder/decoder body, quantizer, ``forward``/``reconstruct`` — is inherited verbatim.

Two quantizers, matching the mask-only line:
* :class:`ConvVQVAE3D_ImgCond` — standard VQ (E4/E6 backbone once SDF is added; E-binary VQ).
* :class:`ConvVQVAE3D_SimVQ_ImgCond` — SimVQ frozen-base + learnable W (E2/E3).

The image is a CONDITION, not a reconstruction target: loss is computed on the 1-channel
mask output only (the image channel is never regressed). Whether the input mask is clean
(baseline) or degraded (completion test) is a training-data choice, not a model change.
"""
from __future__ import annotations

from torch import nn

from models.vae3d_simvq import ConvVQVAE3D_SimVQ
from models.vae3d_vq import ConvVQVAE3D


class _ImgCondMixin:
    """Rebuild the inherited decoder's final transpose-conv to emit ``out_channels``.

    The last decoder stage of :class:`ConvVQVAE3D` is a bare
    ``nn.Sequential(ConvTranspose3d(in_ch, in_channels))`` (no BN/activation on the final
    logits stage). We swap that single layer for one with the same geometry but
    ``out_channels`` outputs, leaving the ``in_ch`` (which already accounts for ``use_skip``
    concatenation) untouched.
    """

    def _shrink_output_head(self, out_channels: int) -> None:
        old = self.dec_stages[-1][0]  # the final ConvTranspose3d
        self.dec_stages[-1] = nn.Sequential(
            nn.ConvTranspose3d(
                old.in_channels, int(out_channels),
                kernel_size=old.kernel_size, stride=old.stride, padding=old.padding,
            )
        )
        self.out_channels = int(out_channels)


class ConvVQVAE3D_ImgCond(_ImgCondMixin, ConvVQVAE3D):
    """Standard-VQ VAE: 2-channel [image, mask] input, 1-channel mask output."""

    def __init__(self, *args, out_channels: int = 1, **kwargs) -> None:
        kwargs["in_channels"] = 2
        super().__init__(*args, **kwargs)
        self._shrink_output_head(out_channels)


class ConvVQVAE3D_SimVQ_ImgCond(_ImgCondMixin, ConvVQVAE3D_SimVQ):
    """SimVQ VAE: 2-channel [image, mask] input, 1-channel mask output."""

    def __init__(self, *args, out_channels: int = 1, **kwargs) -> None:
        kwargs["in_channels"] = 2
        super().__init__(*args, **kwargs)
        self._shrink_output_head(out_channels)
