"""Unit tests for the SDF-input / SDF-output VQ-VAE (advisor request), tiny tensors.

Key invariants beyond WP-3: the model is architecturally identical to the base VQ-VAE
(same param_count — no new layers), the encoder input is the NORMALISED SDF in
``[-1, 1]`` (``to_sdf_input``), ``predict_sdf`` maps a *binary* input to a raw
continuous SDF (converting to SDF internally), and ``reconstruct`` is the binary
bridge ``sdf < 0 -> inside``. The paper-SDF helper's sign/normalisation is checked too.
"""
from __future__ import annotations

import numpy as np
import torch

from models.vae3d_vq import ConvVQVAE3D
from models.vae3d_vq_sdf_io import ConvVQVAE3D_sdf_io
from quality.sdf_paper import DELTA, normalized_sdf, normalized_to_binary

INPUT = (16, 16, 16)


def _base() -> ConvVQVAE3D:
    return ConvVQVAE3D(base_channels=8, num_embeddings=64, embedding_dim=16,
                       commitment_cost=0.25, input_size=INPUT, use_skip=False)


def _sdfio() -> ConvVQVAE3D_sdf_io:
    return ConvVQVAE3D_sdf_io(base_channels=8, num_embeddings=64, embedding_dim=16,
                              commitment_cost=0.25, input_size=INPUT, use_skip=False)


def _batch(b: int = 2) -> torch.Tensor:
    torch.manual_seed(0)
    return (torch.rand(b, 1, *INPUT) > 0.5).float()


def _cube() -> np.ndarray:
    vol = np.zeros((16, 16, 16), dtype=np.float32)
    vol[4:12, 4:12, 4:12] = 1.0
    return vol


# -- paper SDF helper --------------------------------------------------------
def test_normalized_sdf_sign_and_range() -> None:
    """Inside negative, outside positive, boundary ~0, values in [-1, 1]."""
    cube = _cube()
    sdf = normalized_sdf(cube)
    assert sdf.min() >= -1.0 - 1e-6 and sdf.max() <= 1.0 + 1e-6
    assert sdf[8, 8, 8] < 0.0        # deep inside -> negative
    assert sdf[0, 0, 0] > 0.0        # far outside -> positive
    # the bridge recovers the foreground
    assert np.array_equal(normalized_to_binary(sdf), cube > 0.5)


def test_normalized_sdf_is_scaled_by_delta() -> None:
    """Normalised = truncated / delta, so the deepest/farthest reach ±1 at δ."""
    cube = _cube()
    sdf = normalized_sdf(cube, delta=DELTA)
    assert abs(sdf.min()) <= 1.0 + 1e-6
    # at least one voxel saturates the truncation on the outside
    assert np.isclose(sdf.max(), 1.0, atol=1e-6)


# -- model -------------------------------------------------------------------
def test_sdfio_matches_base_param_count() -> None:
    """No new architecture -> identical parameter count to the base VQ-VAE."""
    assert _sdfio().param_count == _base().param_count


def test_to_sdf_input_is_normalised() -> None:
    model = _sdfio()
    x = _batch()
    sdf_in = model.to_sdf_input(x)
    assert sdf_in.shape == x.shape
    assert torch.isfinite(sdf_in).all()
    assert float(sdf_in.min()) >= -1.0 - 1e-6 and float(sdf_in.max()) <= 1.0 + 1e-6
    # a non-empty, non-full binary mask has both inside (neg) and outside (pos)
    assert float(sdf_in.min()) < 0.0 < float(sdf_in.max())


def test_predict_sdf_is_raw_continuous() -> None:
    model = _sdfio()
    x = _batch()
    sdf = model.predict_sdf(x)
    assert sdf.shape == x.shape
    assert torch.isfinite(sdf).all()
    assert float(sdf.min()) < 0.0     # unbounded regression output, not a probability


def test_reconstruct_is_binary_bridge() -> None:
    model = _sdfio()
    x = _batch()
    recon = model.reconstruct(x)
    assert recon.shape == x.shape
    assert set(torch.unique(recon).tolist()).issubset({0.0, 1.0})
    assert torch.equal(recon, (model.predict_sdf(x) < 0).float())


def test_backprop_through_sdf_autoencoder_reaches_encoder() -> None:
    import torch.nn.functional as F

    model = _sdfio()
    x = _batch()
    sdf = model.to_sdf_input(x)
    pred_sdf, vq_term, _ = model.forward(sdf)     # input == target
    loss = F.smooth_l1_loss(pred_sdf, sdf) + vq_term
    loss.backward()
    g = model.enc_stages[0][0].weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
