"""FP/FN dual-head image-conditioned 3D VQ-VAE (advisor direction G).

Same 2-channel ``[grayscale CT, broken mask]`` input as the E2-E7 image-conditioned line,
but the decoder emits **two** channels instead of one:

* channel 0 — **FP** logits (voxels to remove from the input mask),
* channel 1 — **FN** logits (voxels to add to the input mask).

Both heads are binary (independent sigmoids), each trained against its own target from
:func:`data.fpfn_targets.fpfn_targets`. Because the targets are defined *relative to the
input mask*, the model is forced to USE the mask channel (it cannot ignore it and still
predict what to remove/add) — the fix for the completion-regime confound.

The whole encoder/decoder/quantizer body is inherited from
:class:`models.vae3d_imgcond.ConvVQVAE3D_ImgCond` /
:class:`~models.vae3d_imgcond.ConvVQVAE3D_SimVQ_ImgCond`; the only change is the final
transpose-conv width, handled by the existing ``out_channels`` hook (set to 2 here). No
protected file is touched.

SDF-io heads (``*_FPFN_sdf_io``) are the continuous-representation analogue the advisor was
hopeful about: input becomes ``[image, SDF(broken mask)]`` and the two output channels are
the normalised SDFs of the FP and FN regions (restored to binary by the ``sdf < 0`` bridge),
mirroring the mask-only SDF-io line. Whatever the head, :meth:`predict_fpfn` returns binary
FP/FN masks so the eval / ASSD pipeline stays head-agnostic.
"""
from __future__ import annotations

import torch

from models.vae3d_imgcond import ConvVQVAE3D_ImgCond, ConvVQVAE3D_SimVQ_ImgCond
from models.vae3d_imgcond_sdf_io import _ImgCondSdfIoMixin
from models.vae3d_simvq import ConvVQVAE3D_SimVQ
from models.vae3d_vq import ConvVQVAE3D
from quality.sdf_paper import DELTA, normalized_to_binary


class _FPFNMixin:
    """Add the FP/FN semantic accessors on top of a 2-channel image-conditioned backbone."""

    def predict_fpfn(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(fp_prob, fn_prob)`` each ``(B, 1, D, H, W)`` in ``[0, 1]`` (sigmoid)."""
        logits, _, _ = self.forward(x)
        probs = torch.sigmoid(logits)
        return probs[:, 0:1], probs[:, 1:2]


class _FPFNSdfIoMixin(_ImgCondSdfIoMixin):
    """FP/FN accessors for the SDF-io head: predict a 2-channel SDF, bridge each to binary."""

    def predict_fpfn(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return binary ``(fp, fn)`` each ``(B, 1, D, H, W)`` via the ``sdf < 0`` bridge."""
        sdf = self.predict_sdf(x)                       # (B, 2, D, H, W) normalised SDF
        return normalized_to_binary(sdf[:, 0:1]), normalized_to_binary(sdf[:, 1:2])


class ConvVQVAE3D_ImgCond_FPFN(_FPFNMixin, ConvVQVAE3D_ImgCond):
    """Standard-VQ FP/FN model: 2-channel ``[image, broken mask]`` in, ``[FP, FN]`` out."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["out_channels"] = 2
        super().__init__(*args, **kwargs)


class ConvVQVAE3D_SimVQ_ImgCond_FPFN(_FPFNMixin, ConvVQVAE3D_SimVQ_ImgCond):
    """SimVQ FP/FN model: 2-channel ``[image, broken mask]`` in, ``[FP, FN]`` out."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["out_channels"] = 2
        super().__init__(*args, **kwargs)


class ConvVQVAE3D_ImgCond_FPFN_sdf_io(_FPFNSdfIoMixin, ConvVQVAE3D):
    """Standard-VQ SDF-io FP/FN: [image, SDF(broken)] in, 2-channel SDF [FP, FN] out."""

    def __init__(self, *args, delta: float = DELTA, **kwargs) -> None:
        kwargs["in_channels"] = 2
        super().__init__(*args, **kwargs)
        self._shrink_output_head(2)
        self.delta = float(delta)


class ConvVQVAE3D_SimVQ_ImgCond_FPFN_sdf_io(_FPFNSdfIoMixin, ConvVQVAE3D_SimVQ):
    """SimVQ SDF-io FP/FN: [image, SDF(broken)] in, 2-channel SDF [FP, FN] out."""

    def __init__(self, *args, delta: float = DELTA, **kwargs) -> None:
        kwargs["in_channels"] = 2
        super().__init__(*args, **kwargs)
        self._shrink_output_head(2)
        self.delta = float(delta)
