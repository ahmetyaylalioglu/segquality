"""FP/FN/TP triple-head on the VAR multi-scale tokenizer backbone (2026-10-20 direction).

Extends the Faz-3 VAR FP/FN model (:class:`models.vae3d_var_fpfn.VARVQVAE3D_ImgCond_FPFN`)
with the advisor's new **TP** head so the decoder emits the full error decomposition from the
``[grayscale CT, broken mask]`` input:

* channel 0 - **FP** logits (voxels to remove),
* channel 1 - **FN** logits (voxels to add),
* channel 2 - **TP** logits (voxels already correct, keep).

Each head is an independent sigmoid trained against its target from
:func:`data.fpfntp_targets.fpfntp_targets`. The body (encoder / VAR residual quantizer /
decoder) is inherited verbatim; only the final decoder head grows from 2 to 3 channels via the
existing ``_shrink_output_head`` hook. No protected file is touched.
"""
from __future__ import annotations

import torch

from models.vae3d_imgcond import _ImgCondMixin
from models.vae3d_var import VARVQVAE3D


class _FPFNTPMixin:
    """FP/FN/TP semantic accessor on top of a 3-channel image-conditioned backbone."""

    def predict_fpfntp(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(fp, fn, tp)`` each ``(B, 1, D, H, W)`` in ``[0, 1]`` (independent sigmoids)."""
        logits, _, _ = self.forward(x)
        probs = torch.sigmoid(logits)
        return probs[:, 0:1], probs[:, 1:2], probs[:, 2:3]


class VARVQVAE3D_ImgCond_FPFNTP(_FPFNTPMixin, _ImgCondMixin, VARVQVAE3D):
    """VAR-backbone FP/FN/TP model: 2-channel ``[image, broken mask]`` in, ``[FP, FN, TP]`` out.

    The multi-scale residual quantizer is inherited from ``VARVQVAE3D``; only the input width
    (2) and the final decoder head (3 output channels) differ from the clean-mask VAR tokenizer.
    """

    def __init__(self, *args, out_channels: int = 3, **kwargs) -> None:
        kwargs["in_channels"] = 2
        super().__init__(*args, **kwargs)
        self._shrink_output_head(out_channels)
