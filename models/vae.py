"""A plain 2D convolutional VAE that learns the shape prior of clean organ masks.

The model is deliberately simple: no attention, no patches. A 128x128 single-
channel binary mask fits without tiling, so global context is preserved by a
straight conv encoder/decoder. Trained only on clean ground-truth masks (see
``train.py``); at evaluation time the reconstruction error of a *degraded* mask
becomes the quality signal (Phase 6).

Every spatial/channel dimension is derived from ``latent_dim``, ``base_channels``
and ``input_size`` — nothing is hard-coded, so the same class scales to other
input sizes or capacities from config alone.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

# Channel multipliers over ``base_channels``. Length = number of stride-2
# down/up-sampling stages. Four stages take 128 -> 64 -> 32 -> 16 -> 8.
_CHANNEL_MULTS: tuple[int, ...] = (1, 2, 4, 8)


class ConvVAE(nn.Module):
    """Convolutional VAE for single-channel binary masks.

    ``forward`` returns raw logits (no sigmoid) so the loss can use the
    numerically stable ``BCEWithLogitsLoss``.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        base_channels: int = 32,
        input_size: tuple[int, int] = (128, 128),
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.base_channels = int(base_channels)
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.in_channels = int(in_channels)

        n_stages = len(_CHANNEL_MULTS)
        h, w = self.input_size
        if h % (2**n_stages) or w % (2**n_stages):
            raise ValueError(
                f"input_size {self.input_size} must be divisible by "
                f"{2**n_stages} for {n_stages} stride-2 stages"
            )
        # Spatial size of the bottleneck feature map (computed, not hard-coded).
        self._feat_hw = (h // 2**n_stages, w // 2**n_stages)
        channels = [self.base_channels * m for m in _CHANNEL_MULTS]
        self._bottleneck_channels = channels[-1]
        self._flat_dim = self._bottleneck_channels * self._feat_hw[0] * self._feat_hw[1]

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
                nn.Conv2d(prev, ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            prev = ch
        return nn.Sequential(*layers)

    def _build_decoder(self, channels: list[int]) -> nn.Sequential:
        layers: list[nn.Module] = []
        rev = list(reversed(channels))
        for prev, ch in zip(rev[:-1], rev[1:]):
            layers += [
                nn.ConvTranspose2d(prev, ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        # Final stage upsamples and maps to logits: no BatchNorm/activation.
        layers.append(
            nn.ConvTranspose2d(rev[-1], self.in_channels, kernel_size=4, stride=2, padding=1)
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
        h = h.view(-1, self._bottleneck_channels, *self._feat_hw)
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

    # -- internals ----------------------------------------------------------
    def _encode_dist(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)
