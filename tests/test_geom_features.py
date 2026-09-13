"""Unit tests for the geometric feature module (WP-2), all on synthetic volumes."""
from __future__ import annotations

import numpy as np
import torch

from quality.geom_features import (
    ALL_FEATURE_NAMES,
    FEATURE_GROUPS,
    compute_features_batch,
    compute_features_single,
    resolve_feature_names,
)


def _cube(size: int = 16, lo: int = 4, hi: int = 12) -> np.ndarray:
    v = np.zeros((size, size, size), dtype=bool)
    v[lo:hi, lo:hi, lo:hi] = True
    return v


# -- group / name resolution -------------------------------------------------
def test_resolve_order_is_stable_regardless_of_request_order() -> None:
    a = resolve_feature_names(["topology", "shape"])
    b = resolve_feature_names(["shape", "topology"])
    assert a == b == list(ALL_FEATURE_NAMES)


def test_resolve_topology_only() -> None:
    assert resolve_feature_names(["topology"]) == list(FEATURE_GROUPS["topology"])


def test_resolve_unknown_group_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        resolve_feature_names(["nope"])


# -- topology features -------------------------------------------------------
def test_single_blob_one_component_full_fraction() -> None:
    feats = compute_features_single(_cube())
    assert feats["n_components"] == 1.0
    assert feats["largest_cc_fraction"] == 1.0


def test_two_blobs_two_components() -> None:
    v = np.zeros((16, 16, 16), dtype=bool)
    v[2:5, 2:5, 2:5] = True     # blob A
    v[10:13, 10:13, 10:13] = True  # blob B, disconnected
    feats = compute_features_single(v)
    assert feats["n_components"] == 2.0
    # equal-size blobs -> largest holds half the foreground
    assert abs(feats["largest_cc_fraction"] - 0.5) < 1e-6


def test_empty_mask_defaults() -> None:
    feats = compute_features_single(np.zeros((16, 16, 16), dtype=bool))
    assert feats["n_components"] == 0.0
    assert feats["largest_cc_fraction"] == 0.0
    assert feats["volume_frac"] == 0.0
    assert feats["elongation"] == 1.0
    assert feats["centroid_z"] == feats["centroid_y"] == feats["centroid_x"] == 0.5


# -- shape features ----------------------------------------------------------
def test_volume_fraction() -> None:
    v = _cube(size=16, lo=4, hi=12)  # 8^3 foreground in 16^3
    feats = compute_features_single(v)
    assert abs(feats["volume_frac"] - (8**3) / (16**3)) < 1e-6


def test_centroid_of_centred_cube_is_half() -> None:
    feats = compute_features_single(_cube(size=16, lo=4, hi=12))
    for axis in ("centroid_z", "centroid_y", "centroid_x"):
        assert abs(feats[axis] - 0.5) < 1e-6


def test_elongated_box_more_elongated_than_cube() -> None:
    cube = compute_features_single(_cube(size=32, lo=8, hi=24))
    rod = np.zeros((32, 32, 32), dtype=bool)
    rod[2:30, 14:18, 14:18] = True  # long along z, thin in y/x
    rod_feats = compute_features_single(rod)
    assert abs(cube["elongation"] - 1.0) < 0.1
    assert rod_feats["elongation"] > cube["elongation"]
    assert rod_feats["elongation"] > 2.0


# -- batched tensor version --------------------------------------------------
def test_batch_matches_single_and_respects_column_order() -> None:
    names = resolve_feature_names(["topology", "shape"])
    v0 = _cube(size=16, lo=4, hi=12)
    v1 = np.zeros((16, 16, 16), dtype=bool)  # empty
    batch = torch.stack([
        torch.from_numpy(v0.astype(np.float32)),
        torch.from_numpy(v1.astype(np.float32)),
    ]).unsqueeze(1)  # (2,1,16,16,16)
    out = compute_features_batch(batch, names)
    assert out.shape == (2, len(names))
    ref0 = compute_features_single(v0)
    for j, name in enumerate(names):
        assert abs(float(out[0, j]) - ref0[name]) < 1e-5


def test_batch_topology_only_selects_two_columns() -> None:
    names = resolve_feature_names(["topology"])
    batch = (torch.rand(3, 1, 16, 16, 16) > 0.5).float()
    out = compute_features_batch(batch, names)
    assert out.shape == (3, 2)
    assert torch.isfinite(out).all()
