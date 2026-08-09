"""A plain 3D convolutional VAE that learns the shape prior of clean organ volumes.

The volumetric analogue of :class:`models.vae.ConvVAE` (which is untouched). A
whole liver is one training example — a 64x64x64 single-channel binary volume —
so the model sees organ-level 3D shape, not per-slice 2D shape. Same design as 2D
(no attention/patches), one axis higher: ``Conv3d`` / ``BatchNorm3d`` /
``ConvTranspose3d``.

Every spatial/channel dimension is derived from ``latent_dim``, ``base_channels``
and ``input_size`` — nothing is hard-coded, so the same class scales from a 32^3
laptop smoke test to a 64^3 Colab run by changing one config line.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

# Channel multipliers over ``base_channels``. Length = number of stride-2
# down/up-sampling stages. Four stages take 64 -> 32 -> 16 -> 8 -> 4.
_CHANNEL_MULTS: tuple[int, ...] = (1, 2, 4, 8)
_BYTES_PER_FLOAT = 4


class ConvVAE3D(nn.Module):
    """Convolutional VAE for single-channel binary volumes.

    ``forward`` returns raw logits (no sigmoid) so the loss can use the
    numerically stable ``BCEWithLogitsLoss`` — identical contract to the 2D model.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        base_channels: int = 16,
        input_size: tuple[int, int, int] = (64, 64, 64),
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.base_channels = int(base_channels)
        self.input_size = (int(input_size[0]), int(input_size[1]), int(input_size[2]))
        self.in_channels = int(in_channels)

        n_stages = len(_CHANNEL_MULTS)
        divisor = 2**n_stages
        if any(s % divisor for s in self.input_size):
            raise ValueError(
                f"input_size {self.input_size} must be divisible by {divisor} "
                f"for {n_stages} stride-2 stages"
            )
        # Spatial size of the bottleneck feature map (computed, not hard-coded).
        self._feat_dhw = tuple(s // divisor for s in self.input_size)
        channels = [self.base_channels * m for m in _CHANNEL_MULTS]
        self._bottleneck_channels = channels[-1]
        self._flat_dim = self._bottleneck_channels * math.prod(self._feat_dhw)

        self.encoder = self._build_encoder(channels)
        self.fc_mu = nn.Linear(self._flat_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self._flat_dim, self.latent_dim)
        self.fc_decode = nn.Linear(self.latent_dim, self._flat_dim)
        self.decoder = self._build_decoder(channels)

    # -- construction helpers ----------------------------------------------
    def _build_encoder(self, channels: list[int]) -> nn.Sequential:
        layers: list[nn.Module] = []
        prev = self.in_channels
        for ch in channels:
            layers += [
                nn.Conv3d(prev, ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            prev = ch
        return nn.Sequential(*layers)

    def _build_decoder(self, channels: list[int]) -> nn.Sequential:
        layers: list[nn.Module] = []
        rev = list(reversed(channels))
        for prev, ch in zip(rev[:-1], rev[1:]):
            layers += [
                nn.ConvTranspose3d(prev, ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        # Final stage upsamples and maps to logits: no BatchNorm/activation.
        layers.append(
            nn.ConvTranspose3d(rev[-1], self.in_channels, kernel_size=4, stride=2, padding=1)
        )
        return nn.Sequential(*layers)

    # -- core API -----------------------------------------------------------
    def encode(self, x: Tensor) -> Tensor:
        """Return the posterior mean ``mu`` (the deterministic latent code)."""
        mu, _ = self._encode_dist(x)
        return mu

    def decode(self, z: Tensor) -> Tensor:
        """Map a latent code to reconstruction **logits** (no sigmoid)."""
        h = self.fc_decode(z)
        h = h.view(-1, self._bottleneck_channels, *self._feat_dhw)
        return self.decoder(h)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Sample ``z`` in training; return ``mu`` deterministically in eval."""
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self._encode_dist(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar

    def reconstruct(self, x: Tensor) -> Tensor:
        """Reconstruction **probabilities** (sigmoid applied), for eval/preview."""
        logits, _, _ = self.forward(x)
        return torch.sigmoid(logits)

    # -- introspection (3D models grow fast; surface size before OOM) -------
    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def estimate_forward_mb(self, batch_size: int) -> float:
        """Rough per-forward activation memory (MB) for ``batch_size`` volumes.

        Sums the element counts of the input plus every encoder/decoder feature
        map (both halves are symmetric) at float32 — a coarse OOM early-warning,
        not an exact figure (ignores autograd's saved-tensor bookkeeping).
        """
        elems = math.prod(self.input_size) * self.in_channels  # input volume
        spatial = list(self.input_size)
        for m in _CHANNEL_MULTS:
            spatial = [s // 2 for s in spatial]
            elems += self.base_channels * m * math.prod(spatial)
        elems *= 2  # encoder + symmetric decoder feature maps
        return elems * batch_size * _BYTES_PER_FLOAT / 1e6

    # -- internals ----------------------------------------------------------
    def _encode_dist(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)
