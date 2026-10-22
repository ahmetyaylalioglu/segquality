"""Two-stage VAR refiner with the TP head and a recurrent stage-2 (2026-10-20 direction).

The advisor's new cascade keeps the frozen stage-1 FP/FN model but promotes it to the full
error decomposition and adds a *recurrent, image-aware* stage-2 refiner:

* **Stage-1** is the frozen :class:`models.vae3d_var_fpfntp.VARVQVAE3D_ImgCond_FPFNTP`. It
  consumes ``[grayscale CT, broken mask]`` and predicts ``[FP, FN, TP]``.
* **Stage-2** is a NEW VAR refiner (:class:`VARVQVAE3D_RefinerTP`) that consumes
  ``[image, FP, FN, TP]`` (4 channels -- the image is kept so the refiner can "learn from the
  grayscale", per the advisor) and predicts refined ``[FP, FN, TP]``. Only stage-2 is trained.
* **Recurrence:** the refiner is applied for ``iters`` passes (advisor: 2nd and 3rd iteration),
  feeding its own ``[FP, FN, TP]`` output back as the next input alongside the SAME image, with
  shared weights. The training loss is summed over every iteration.

The refined clean mask is ``TP OR FN`` (:func:`data.fpfntp_targets.reconstruct_from_fntp`); the
metric of interest is how its Dice against the clean mask evolves across iterations. Reuses the
VAR backbone and the ``_ImgCondMixin`` head rebuild (in=4, out=3). No protected file is touched.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from models.vae3d_imgcond import _ImgCondMixin
from models.vae3d_var import VARVQVAE3D
from models.vae3d_var_fpfntp import VARVQVAE3D_ImgCond_FPFNTP


class VARVQVAE3D_RefinerTP(_ImgCondMixin, VARVQVAE3D):
    """Stage-2 VAR refiner: 4-channel ``[image, FP, FN, TP]`` in, 3-channel ``[FP, FN, TP]`` out.

    The decoder head is **zero-initialised** so the refiner predicts a RESIDUAL in logit space
    (:meth:`CascadeFPFNTP.refine_step`): at initialisation its delta is 0, so the refined maps
    equal the input maps and the recurrent cascade starts exactly at the stage-1 accuracy (it
    can only polish from there, instead of regenerating the mask from scratch and having to
    climb back up from the degenerate "predict everything" Dice).
    """

    def __init__(self, *args, in_channels: int = 4, out_channels: int = 3, **kwargs) -> None:
        kwargs["in_channels"] = int(in_channels)
        super().__init__(*args, **kwargs)
        self._shrink_output_head(out_channels)
        head = self.dec_stages[-1][0]          # final ConvTranspose3d (the residual delta head)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)


class CascadeFPFNTP(nn.Module):
    """Frozen stage-1 FP/FN/TP + trainable recurrent stage-2 refiner (advisor's 2-VAR cascade).

    ``forward`` runs the refiner for ``iters`` recurrent passes and returns the per-iteration
    ``[FP, FN, TP]`` probability maps (so the loss can be summed over iterations). The stage-1
    products are produced under ``no_grad`` by :meth:`stage1_products`. ``predict_reference``
    runs the full cascade for GT-free inference and returns the final refined probabilities.
    """

    def __init__(
        self,
        stage1: VARVQVAE3D_ImgCond_FPFNTP,
        refiner: VARVQVAE3D_RefinerTP,
        iters: int = 3,
    ) -> None:
        super().__init__()
        self.stage1 = stage1
        self.refiner = refiner
        self.iters = int(iters)
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
        """From ``x = [image, broken]`` return ``(fp, fn, tp)`` probs of the frozen stage-1."""
        return self.stage1.predict_fpfntp(x)

    @staticmethod
    def refiner_input(image: Tensor, fp: Tensor, fn: Tensor, tp: Tensor) -> Tensor:
        """Stack ``[image, FP, FN, TP]`` into the 4-channel stage-2 input (image is kept)."""
        return torch.cat([image, fp, fn, tp], dim=1)

    @staticmethod
    def _logit(p: Tensor, eps: float = 1e-4) -> Tensor:
        """Numerically-safe inverse sigmoid of a probability map."""
        p = p.clamp(eps, 1.0 - eps)
        return torch.log(p) - torch.log1p(-p)

    def refine_step(
        self, image: Tensor, fp: Tensor, fn: Tensor, tp: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """One RESIDUAL refiner pass: refined map = sigmoid(logit(input) + delta).

        The refiner (zero-init head) predicts a logit-space delta per channel; adding it to the
        input maps' logits means delta=0 reproduces the input exactly, so the pass starts as an
        identity and only polishes. ``vq`` is the refiner's VQ commitment term for this pass (so
        the training loop can add it and actually train the codebook).
        """
        delta, vq, _ = self.refiner.forward(self.refiner_input(image, fp, fn, tp))
        in_logits = torch.cat([self._logit(fp), self._logit(fn), self._logit(tp)], dim=1)
        probs = torch.sigmoid(in_logits + delta)
        return probs[:, 0:1], probs[:, 1:2], probs[:, 2:3], vq

    def forward(
        self, x: Tensor, iters: int | None = None
    ) -> tuple[list[tuple[Tensor, Tensor, Tensor]], Tensor]:
        """Run stage-1 then ``iters`` recurrent refiner passes on ``x = [image, broken]``.

        Returns ``(outputs, vq_total)`` where ``outputs`` is a list of ``(FP, FN, TP)``
        probability triples (one per refiner iteration, so the loss can be accumulated at EACH
        iteration) and ``vq_total`` is the summed VQ commitment term over the iterations. The
        image channel ``x[:, 0:1]`` is re-fed every pass; only the FP/FN/TP maps are updated.
        """
        n = self.iters if iters is None else int(iters)
        image = x[:, 0:1]
        fp, fn, tp = self.stage1_products(x)
        outputs: list[tuple[Tensor, Tensor, Tensor]] = []
        vq_total = image.new_zeros(())
        for _ in range(n):
            fp, fn, tp, vq = self.refine_step(image, fp, fn, tp)
            outputs.append((fp, fn, tp))
            vq_total = vq_total + vq
        return outputs, vq_total

    @torch.no_grad()
    def predict_reference(self, x: Tensor, iters: int | None = None) -> Tensor:
        """Full cascade on ``x = [image, broken]`` -> final refined clean-mask probs ``TP OR FN``."""
        outputs, _ = self.forward(x, iters=iters)
        _, fn, tp = outputs[-1]
        return torch.clamp(tp + fn, 0.0, 1.0)
