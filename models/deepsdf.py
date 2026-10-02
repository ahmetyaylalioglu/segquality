"""DeepSDF auto-decoder — a continuous implicit shape prior (advisor's NIR/INR line).

Motivation (why a new model, nothing existing is touched)
---------------------------------------------------------
The VQ-VAE line reconstructs a mask through a *discrete* spatial codebook. The
codebook collapses (~13/512 codes used) and its latent is a fixed 4x4x4 grid, so
there is no smooth shape manifold to interpolate. DeepSDF (Park et al., 2019,
arXiv:1901.05103) replaces the grid with ONE continuous latent vector per shape and
a coordinate-MLP ``f(z, xyz) -> sdf``. There is no encoder and no codebook, so:

* codebook collapse is structurally impossible (no codes at all);
* the per-shape latent lives in a continuous R^L that can be interpolated /
  optimised, which is exactly the "parametric shape learning" direction the advisor
  pointed at (ImplicitAtlas, INR4MICCAI).

Auto-decoder training
---------------------
Latents are an ``nn.Embedding(num_shapes, latent_dim)`` initialised at N(0, 0.01)
(the paper's value). A training step feeds ``concat(z_i, xyz)`` for many sampled
points of shape ``i`` through an 8-layer 512-wide MLP with a mid-network skip
(the input is re-concatenated at ``skip_layers``) and weight-norm, and regresses the
truncated SDF. This model is pure ``torch`` and side-effect free, so it is unit
tested with tiny dims and no dataset.

The sign convention matches ``quality/sdf.py``: outside +, inside -, boundary ~0, so
the binary bridge ``sdf < 0 -> inside`` turns a decoded field back into a mask for
the identical Dice/correlation QC machinery.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DeepSDF(nn.Module):
    """Auto-decoder: per-shape latent embedding + coordinate MLP -> scalar SDF.

    Parameters
    ----------
    num_shapes:
        Number of training shapes; one latent row per shape.
    latent_dim:
        Length ``L`` of each shape's latent code.
    hidden_dim, num_layers:
        Width and depth of the coordinate MLP (paper: 512 x 8).
    skip_layers:
        Layer indices whose input is re-concatenated with the raw ``(z, xyz)``
        input (the paper's single mid-network skip at layer 4).
    dropout:
        Dropout probability inside the MLP (paper: 0.2).
    latent_init_std:
        Std of the N(0, .) latent initialisation (paper: 0.01).
    weight_norm:
        Apply weight normalisation to the MLP linears (paper: on).
    """

    def __init__(
        self,
        num_shapes: int,
        latent_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 8,
        skip_layers: tuple[int, ...] = (4,),
        dropout: float = 0.2,
        latent_init_std: float = 0.01,
        weight_norm: bool = True,
    ) -> None:
        super().__init__()
        self.num_shapes = int(num_shapes)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.skip_layers = set(int(s) for s in skip_layers)
        self.dropout_p = float(dropout)

        self.latents = nn.Embedding(self.num_shapes, self.latent_dim)
        nn.init.normal_(self.latents.weight, mean=0.0, std=float(latent_init_std))

        input_dim = self.latent_dim + 3
        self.input_dim = input_dim
        layers: list[nn.Module] = []
        for i in range(self.num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            if i in self.skip_layers:
                in_dim += input_dim
            lin = nn.Linear(in_dim, hidden_dim)
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            layers.append(lin)
        self.layers = nn.ModuleList(layers)
        self.out = nn.Linear(hidden_dim, 1)

    # -- core -----------------------------------------------------------------
    def decode(self, latents: Tensor, coords: Tensor) -> Tensor:
        """Predict SDF for explicit latents and coordinates.

        ``latents`` is ``(M, L)`` and ``coords`` is ``(M, 3)`` (one latent per
        point, already broadcast by the caller). Returns ``(M,)`` raw SDF.
        """
        x = torch.cat([latents, coords], dim=-1)
        h = x
        for i, lin in enumerate(self.layers):
            if i in self.skip_layers:
                h = torch.cat([h, x], dim=-1)
            h = lin(h)
            h = F.relu(h, inplace=True)
            if self.dropout_p > 0.0:
                h = F.dropout(h, p=self.dropout_p, training=self.training)
        return self.out(h).squeeze(-1)

    def forward(self, indices: Tensor, coords: Tensor) -> Tensor:
        """Predict SDF for shape ``indices`` (M,) at ``coords`` (M, 3) -> (M,)."""
        latents = self.latents(indices)
        return self.decode(latents, coords)

    # -- helpers --------------------------------------------------------------
    def latent_l2(self, indices: Tensor) -> Tensor:
        """Mean squared L2 norm of the referenced latents (the code regulariser)."""
        z = self.latents(indices)
        return (z * z).sum(dim=-1).mean()

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
