"""segquality models package: the shape-prior VAE.

Only the architecture lives here. Losses live in ``losses`` and the training
loop in ``train.py`` so the model stays a pure ``nn.Module`` with no training or
config-parsing concerns baked in.
"""
from __future__ import annotations

from models.vae import ConvVAE
from models.vae3d import ConvVAE3D

__all__ = ["ConvVAE", "ConvVAE3D"]
