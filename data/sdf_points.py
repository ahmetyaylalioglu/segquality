"""Point samples for DeepSDF training — turn binary masks into ``(xyz, sdf)`` pairs.

DeepSDF regresses an SDF at scattered *points*, not on a full grid, so a shape must
be presented as a cloud of ``(coordinate, signed-distance)`` samples. This module
precomputes the truncated SDF grid once per shape (reusing ``quality/sdf.py`` — the
same sign convention as the rest of WP-3) and then serves random point batches.

Design choices
--------------
* **Precompute-once.** The EDT-based SDF of every shape is computed a single time in
  ``__init__`` and cached as a CPU ``float32`` stack; a training step is then a pure
  index-gather (fast, no SciPy in the loop).
* **Near-surface weighting.** The SDF carries information mostly *near the boundary*;
  the paper oversamples there. ``sample`` draws a ``near_surface_frac`` fraction from
  the voxels inside a thin ``surface_band`` and the rest uniformly, so the boundary
  is well constrained without ignoring the far field.
* **Normalised coordinates.** Voxel indices are mapped to ``[-1, 1]`` per axis so the
  MLP sees a resolution-independent domain. Decoding a full grid (for a Dice sanity
  check) uses the identical mapping via :func:`normalized_grid`.

Pure NumPy/Torch + ``quality.sdf``; no dataset dependency, so it is unit tested with a
synthetic cube.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from quality.sdf import TRUNCATION, signed_distance, truncate


def normalized_grid(shape: tuple[int, int, int]) -> Tensor:
    """Full ``(D*H*W, 3)`` grid of voxel-centre coordinates mapped to ``[-1, 1]``.

    Axis order matches the flattened SDF volume (``D`` slowest, ``W`` fastest), so
    ``normalized_grid(shape)[k]`` is the coordinate of ``sdf.reshape(-1)[k]``.
    """
    d, h, w = shape
    zs = torch.linspace(-1.0, 1.0, d)
    ys = torch.linspace(-1.0, 1.0, h)
    xs = torch.linspace(-1.0, 1.0, w)
    gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")
    return torch.stack([gz.reshape(-1), gy.reshape(-1), gx.reshape(-1)], dim=-1).contiguous()


class SDFPointSampler:
    """Precompute per-shape truncated SDF grids and serve random point batches.

    Parameters
    ----------
    masks:
        Iterable of ``(D, H, W)`` binary volumes (bool/float in {0, 1}).
    truncation:
        TSDF half-width (``quality/sdf.py``'s ``TRUNCATION`` by default).
    surface_band:
        Voxels with ``|sdf| <= surface_band`` form the near-surface pool.
    """

    def __init__(
        self,
        masks,
        truncation: float = TRUNCATION,
        surface_band: float = 2.0,
    ) -> None:
        arr = np.stack([np.asarray(m, dtype=np.float32) for m in masks])
        if arr.ndim != 4:
            raise ValueError(f"masks must stack to (N, D, H, W); got {arr.shape}")
        self.num_shapes = int(arr.shape[0])
        self.shape = (int(arr.shape[1]), int(arr.shape[2]), int(arr.shape[3]))
        self.truncation = float(truncation)

        sdf = np.empty(arr.shape, dtype=np.float32)
        for i in range(self.num_shapes):
            sdf[i] = truncate(signed_distance(arr[i]), self.truncation)
        self.sdf = torch.from_numpy(sdf)                 # (N, D, H, W)
        self.sdf_flat = self.sdf.view(self.num_shapes, -1)  # (N, P)
        self.n_vox = self.sdf_flat.shape[1]
        self.coords = normalized_grid(self.shape)        # (P, 3)

        band = (self.sdf_flat.abs() <= float(surface_band))
        self.surface_idx: list[Tensor] = [
            torch.nonzero(band[i], as_tuple=False).squeeze(1) for i in range(self.num_shapes)
        ]

    def __len__(self) -> int:
        return self.num_shapes

    def sample(
        self,
        shape_indices,
        n_points: int,
        near_surface_frac: float = 0.5,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sample ``n_points`` per shape in ``shape_indices``.

        Returns ``(idx, coords, sdf)`` flattened over shapes:
        ``idx`` ``(M,)`` long (shape id per point), ``coords`` ``(M, 3)``,
        ``sdf`` ``(M,)`` truncated target, where ``M = len(shape_indices)*n_points``.
        """
        if isinstance(shape_indices, Tensor):
            shape_indices = shape_indices.tolist()
        n_near = int(round(n_points * float(near_surface_frac)))
        n_far = n_points - n_near

        idx_out, coord_out, sdf_out = [], [], []
        for s in shape_indices:
            picks = []
            if n_near > 0:
                pool = self.surface_idx[s]
                if pool.numel() == 0:                    # empty/degenerate shape
                    picks.append(torch.randint(self.n_vox, (n_near,), generator=generator))
                else:
                    sel = torch.randint(pool.numel(), (n_near,), generator=generator)
                    picks.append(pool[sel])
            if n_far > 0:
                picks.append(torch.randint(self.n_vox, (n_far,), generator=generator))
            vox = torch.cat(picks) if picks else torch.empty(0, dtype=torch.long)
            coord_out.append(self.coords[vox])
            sdf_out.append(self.sdf_flat[s, vox])
            idx_out.append(torch.full((vox.numel(),), int(s), dtype=torch.long))

        return torch.cat(idx_out), torch.cat(coord_out), torch.cat(sdf_out)
