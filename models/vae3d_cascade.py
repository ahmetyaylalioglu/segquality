"""Cascade stage-2 refiner on the VAR backbone (Faz 4).

Advisor's WhatsApp drawing (2026-10-12): ``Img -> VAE -> [FP(-)] [FN(+)] -> 2nd VAE ->
[reference mask]``, with "only train this one" pointing at the SECOND VAE. So:

* **Stage-1** is the frozen FP/FN model (:class:`models.vae3d_var_fpfn.VARVQVAE3D_ImgCond_FPFN`,
  the Faz-3 winner). It consumes ``[grayscale CT, broken mask]`` and predicts ``[FP, FN]``,
  from which a ``corrected`` mask ``(broken AND NOT FP) OR FN`` follows. The image feeds ONLY
  this stage.
* **Stage-2** is a NEW refiner VAE (this module) that takes the stage-1 products
  ``[corrected, FP, FN]`` (3 channels, NO image — faithful to the drawing's two −/+ boxes plus
  their combination) and predicts a single-channel REFERENCE mask. Only stage-2 is trained;
  stage-1 stays frozen.

The question (advisor): does a 2nd VAE increase the produced reference mask's accuracy, i.e.
is ``Dice(refined, clean) > Dice(corrected, clean)``? Per the project rule we ALSO gate on the
QC metric, not accuracy alone.

Reuses the VAR backbone verbatim and the :class:`models.vae3d_imgcond._ImgCondMixin` output-head
rebuild (in=3, out=1). No protected file is touched.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from data.fpfn_targets import correct_mask
from models.vae3d_imgcond import _ImgCondMixin
from models.vae3d_var import VARVQVAE3D
from models.vae3d_var_fpfn import VARVQVAE3D_ImgCond_FPFN


class VARVQVAE3D_Refiner(_ImgCondMixin, VARVQVAE3D):
    """Stage-2 VAR refiner: 3-channel ``[corrected, FP, FN]`` in, 1-channel mask logits out."""

    def __init__(self, *args, in_channels: int = 3, out_channels: int = 1, **kwargs) -> None:
        kwargs["in_channels"] = int(in_channels)
        super().__init__(*args, **kwargs)
        self._shrink_output_head(out_channels)


class CascadeFPFN(nn.Module):
    """Frozen stage-1 FP/FN + trainable stage-2 refiner (advisor's 2-VAE cascade).

    ``forward`` is the stage-2 path only (so an optimizer over ``self.refiner.parameters()``
    trains exactly the second VAE). The stage-1 products are produced under ``no_grad`` by
    :meth:`stage1_products`. ``predict_reference`` runs the full cascade for GT-free inference.
    """

    def __init__(self, stage1: VARVQVAE3D_ImgCond_FPFN, refiner: VARVQVAE3D_Refiner) -> None:
        super().__init__()
        self.stage1 = stage1
        self.refiner = refiner
        for p in self.stage1.parameters():
            p.requires_grad_(False)
        self.stage1.eval()

    @property
    def in_channels(self) -> int:
        return self.refiner.in_channels

    @property
    def out_channels(self) -> int:
        return self.refiner.out_channels

    @property
    def param_count(self) -> int:
        return self.refiner.param_count

    @property
    def v_patch_nums(self) -> tuple[int, ...]:
        return self.refiner.v_patch_nums

    def train(self, mode: bool = True):  # keep stage-1 frozen in eval always
        super().train(mode)
        self.stage1.eval()
        return self

    @torch.no_grad()
    def stage1_products(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """From ``x = [image, broken]`` return ``(corrected, fp, fn)`` of the frozen stage-1."""
        fp, fn = self.stage1.predict_fpfn(x)                 # sigmoid prob maps (B,1,...)
        corrected = correct_mask(x[:, 1:2], fp, fn)          # (B,1,...), {0,1}
        return corrected, fp, fn

    @staticmethod
    def refiner_input(corrected: Tensor, fp: Tensor, fn: Tensor) -> Tensor:
        """Stack the stage-1 products into the 3-channel stage-2 input (no image)."""
        return torch.cat([corrected, fp, fn], dim=1)

    def forward(self, refiner_in: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        return self.refiner.forward(refiner_in)

    @torch.no_grad()
    def predict_reference(self, x: Tensor) -> Tensor:
        """Full cascade on ``x = [image, broken]`` -> refined reference mask probs (B,1,...)."""
        corrected, fp, fn = self.stage1_products(x)
        logits, _, _ = self.refiner.forward(self.refiner_input(corrected, fp, fn))
        return torch.sigmoid(logits)
