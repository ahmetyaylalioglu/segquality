"""3D PatchGAN discriminator for the VAR Eq.(5) adversarial term (Faz 2).

VAR's Stage-1 tokenizer is trained as a VQGAN: its loss (paper Eq. 5) is
``L = ||im - im_hat||_2 + ||f - f_hat||_2 + lambda_P * LPIPS + lambda_G * L_GAN``.
The advisor explicitly asked for the ``lambda_G * L_GAN`` term ("GAN hatasi").
LPIPS/feature-L2 target natural RGB images and are meaningless for binary organ
masks, so Faz 2 keeps our existing recon (BCE + soft-Dice) + VQ objective and adds
**only** the adversarial term, supplied by this discriminator.

This is the 3D analogue of the classic PatchGAN (Isola et al. 2017): a small fully
convolutional stack of stride-2 ``Conv3d`` blocks that outputs a grid of real/fake
logits, each judging a local receptive field ("patch") of the volume rather than a
single global verdict. Patch-level judgement pushes the generator toward locally
realistic mask boundaries, which is exactly where our reconstructions are softest.

Trained with the hinge objective (VQGAN's choice):
    D: 0.5 * (relu(1 - D(real)).mean() + relu(1 + D(fake)).mean())
    G: -D(fake).mean()
The training loop (``scripts/train_vae3d_var_gan.py``) owns the optimisation; this
module is a plain ``nn.Module`` so the existing model/eval code is untouched.
"""
from __future__ import annotations

from torch import Tensor, nn


class PatchGAN3D(nn.Module):
    """Fully-convolutional 3D patch discriminator.

    ``n_layers`` stride-2 blocks shrink the volume while widening channels, then a
    stride-1 block and a 1-channel head emit the patch-logit grid. With
    ``base_channels=32`` and ``n_layers=3`` on a 64^3 input the receptive field
    covers a local sub-volume, not the whole organ.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32, n_layers: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv3d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        ch = base_channels
        for _ in range(1, n_layers):
            nxt = min(ch * 2, base_channels * 8)
            layers += [
                nn.Conv3d(ch, nxt, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm3d(nxt, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = nxt
        nxt = min(ch * 2, base_channels * 8)
        layers += [
            nn.Conv3d(ch, nxt, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm3d(nxt, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nxt, 1, kernel_size=4, stride=1, padding=1),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Map a volume ``(B, in_channels, D, H, W)`` to a patch-logit grid."""
        return self.model(x)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
