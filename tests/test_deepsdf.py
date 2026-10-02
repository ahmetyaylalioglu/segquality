"""Unit tests for the DeepSDF auto-decoder + its point sampler, synthetic cubes only.

Covers the two things a new-model contribution must guarantee before a full run:
* the model wires up (forward/decode shapes, latents, skip, param count, gradients);
* the sampler serves correctly-signed, correctly-shaped SDF point batches and its
  full-grid coordinates round-trip through the same ``sdf < 0`` bridge as WP-3.
No dataset is needed.
"""
from __future__ import annotations

import numpy as np
import torch

from data.sdf_points import SDFPointSampler, normalized_grid
from models.deepsdf import DeepSDF


def _cube(size: int = 16, half: int = 4) -> np.ndarray:
    c = size // 2
    vol = np.zeros((size, size, size), dtype=np.float32)
    vol[c - half:c + half, c - half:c + half, c - half:c + half] = 1.0
    return vol


def _tiny_model(num_shapes: int = 4) -> DeepSDF:
    return DeepSDF(num_shapes=num_shapes, latent_dim=8, hidden_dim=16,
                   num_layers=4, skip_layers=(2,), dropout=0.0)


# -- model -------------------------------------------------------------------
def test_forward_shapes() -> None:
    model = _tiny_model()
    idx = torch.tensor([0, 1, 2, 3, 0])
    coords = torch.randn(5, 3)
    out = model(idx, coords)
    assert out.shape == (5,)
    assert torch.isfinite(out).all()


def test_decode_matches_forward() -> None:
    """forward(idx, coords) must equal decode(latents(idx), coords)."""
    model = _tiny_model()
    model.eval()
    idx = torch.tensor([1, 3])
    coords = torch.randn(2, 3)
    with torch.no_grad():
        a = model(idx, coords)
        b = model.decode(model.latents(idx), coords)
    assert torch.allclose(a, b)


def test_skip_layer_input_dim() -> None:
    """The skip layer's linear must take hidden_dim + input_dim inputs."""
    model = _tiny_model()
    skip = next(iter(model.skip_layers))
    lin = model.layers[skip]
    # weight_norm wraps the weight; in_features is still exposed
    assert lin.in_features == model.hidden_dim + model.input_dim


def test_latents_and_param_count() -> None:
    model = _tiny_model(num_shapes=7)
    assert model.latents.weight.shape == (7, 8)
    assert model.param_count > 0
    # latent init is small (N(0, 0.01))
    assert model.latents.weight.std().item() < 0.1


def test_gradients_flow_to_latents_and_decoder() -> None:
    model = _tiny_model()
    idx = torch.tensor([0, 1, 2])
    coords = torch.randn(3, 3)
    loss = model(idx, coords).pow(2).mean() + 1e-3 * model.latent_l2(idx)
    loss.backward()
    assert model.latents.weight.grad is not None
    assert torch.isfinite(model.latents.weight.grad).all()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for n, p in model.named_parameters() if not n.startswith("latents"))


def test_latent_l2_is_nonnegative() -> None:
    model = _tiny_model()
    assert model.latent_l2(torch.tensor([0, 1])).item() >= 0.0


# -- sampler -----------------------------------------------------------------
def test_normalized_grid_bounds_and_order() -> None:
    grid = normalized_grid((4, 5, 6))
    assert grid.shape == (4 * 5 * 6, 3)
    assert float(grid.min()) == -1.0 and float(grid.max()) == 1.0


def test_sampler_shapes_and_count() -> None:
    sampler = SDFPointSampler([_cube(), _cube(half=3)])
    idx, coords, sdf = sampler.sample([0, 1], n_points=100, near_surface_frac=0.5)
    assert idx.shape == (200,) and coords.shape == (200, 3) and sdf.shape == (200,)
    assert set(idx.tolist()) == {0, 1}
    assert (idx == 0).sum().item() == 100


def test_sampler_sign_convention() -> None:
    """A point at the cube centre must read a negative (inside) SDF."""
    sampler = SDFPointSampler([_cube()])
    # centre voxel of a 16^3 grid, flattened
    d, h, w = sampler.shape
    c = d // 2
    flat = (c * h + c) * w + c
    assert sampler.sdf_flat[0, flat].item() < 0.0
    # truncation respected
    assert sampler.sdf.abs().max().item() <= sampler.truncation + 1e-6


def test_sampler_near_surface_values_are_small() -> None:
    """Near-surface-only draws should mostly have |sdf| within the band."""
    sampler = SDFPointSampler([_cube()], surface_band=2.0)
    _, _, sdf = sampler.sample([0], n_points=500, near_surface_frac=1.0)
    assert (sdf.abs() <= 2.0 + 1e-6).float().mean().item() > 0.95


def test_full_grid_bridge_roundtrip() -> None:
    """Decoding the stored SDF grid via sdf<0 recovers the input cube exactly."""
    cube = _cube()
    sampler = SDFPointSampler([cube])
    recovered = (sampler.sdf[0] < 0).numpy()
    assert np.array_equal(recovered, cube > 0.5)
