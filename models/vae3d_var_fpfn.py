"""FP/FN dual-head on the VAR multi-scale tokenizer backbone (Faz 3).

Direction G (FP/FN) asks: given ``[grayscale CT, broken mask]``, predict a
false-positive map (voxels to REMOVE) and a false-negative map (voxels to ADD), from
which a corrected mask and a GT-free ``est_dice`` quality score follow. The existing
winner is the binary-head image-conditioned VQ-VAE
(:class:`models.vae3d_imgcond_fpfn.ConvVQVAE3D_ImgCond_FPFN`).

Faz 3 swaps ONLY the backbone's quantizer for the VAR multi-scale residual quantizer
(:class:`models.vae3d_var.VARVQVAE3D`), keeping the exact same 2-channel input and
2-channel ``[FP, FN]`` output. The question: does the VAR backbone make the FP/FN
maps resemble the reference better (higher corrected Dice) and/or improve the
``est_dice`` quality correlation vs the plain-VQ FP/FN model?

Only the binary head is ported (it is the established best head for FP/FN); the SDF-io
head is out of scope for Faz 3. Reuses ``_ImgCondMixin`` (rebuild the final decoder
layer to 2 output channels) and ``_FPFNMixin`` (``predict_fpfn``) verbatim, so the
FP/FN eval / ASSD pipeline stays head-agnostic. No protected file is touched.
"""
from __future__ import annotations

from models.vae3d_imgcond import _ImgCondMixin
from models.vae3d_imgcond_fpfn import _FPFNMixin
from models.vae3d_var import VARVQVAE3D


class VARVQVAE3D_ImgCond_FPFN(_FPFNMixin, _ImgCondMixin, VARVQVAE3D):
    """VAR-backbone FP/FN model: 2-channel ``[image, broken mask]`` in, ``[FP, FN]`` out.

    The multi-scale residual quantizer is inherited from ``VARVQVAE3D``; only the input
    width (2) and the final decoder head (2 output channels) differ from the clean-mask
    VAR tokenizer.
    """

    def __init__(self, *args, out_channels: int = 2, **kwargs) -> None:
        kwargs["in_channels"] = 2
        super().__init__(*args, **kwargs)
        self._shrink_output_head(out_channels)
