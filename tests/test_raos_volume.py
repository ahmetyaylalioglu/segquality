"""Tests for the 3D volume dataset (Phase 7), data-free.

A synthetic labelled volume is injected via ``volume_loader`` so ``__getitem__``
and the precomputed path are exercised without nibabel or raw data. Mirrors the
2D ``test_precomputed.py``: the nifti and precomputed paths must return identical
masks, and the precomputed path must never touch a volume loader.
"""
from __future__ import annotations

import numpy as np
import pytest

from data.raos_volume import RAOSVolumes, unique_patient_records
from data.volume_transforms import extract_organ_volume

SIZE = (16, 16, 16)
ORGAN_ID = 1


def _labelled_volume() -> np.ndarray:
    """A synthetic label volume with an off-centre liver (id=1) plus other ids."""
    vol = np.zeros((20, 24, 18), dtype=np.int16)
    vol[5:12, 6:16, 4:12] = ORGAN_ID
    vol[0:3, 0:3, 0:3] = 2  # a different organ, must be ignored
    return vol


def _sample_records() -> list[dict]:
    return [
        {"patient_id": "pA", "source_subset": "Set1", "original_split": "Tr"},
        {"patient_id": "pB", "source_subset": "Set1", "original_split": "Tr"},
    ]


def _nifti_ds() -> RAOSVolumes:
    vol = _labelled_volume()
    return RAOSVolumes(
        _sample_records(), root="/unused", organ_id=ORGAN_ID,
        output_size=SIZE, bbox_margin=0.25,
        volume_loader=lambda _path: vol,
    )


def test_getitem_returns_bool_mask_of_output_size() -> None:
    ds = _nifti_ds()
    sample = ds[0]
    assert sample["mask"].shape == SIZE and sample["mask"].dtype == bool
    assert sample["mask"].any()
    assert sample["patient_id"] == "pA"
    assert sample["meta"]["source_subset"] == "Set1"


def test_precomputed_matches_nifti_exactly() -> None:
    vol = _labelled_volume()
    expected = extract_organ_volume(vol == ORGAN_ID, SIZE, 0.25)
    stack = np.stack([expected.astype(np.uint8), expected.astype(np.uint8)])
    pre = RAOSVolumes(
        _sample_records(), root="/unused", organ_id=ORGAN_ID,
        output_size=SIZE, bbox_margin=0.25, volumes=stack,
    )
    assert np.array_equal(pre[0]["mask"], _nifti_ds()[0]["mask"])
    assert pre[0]["mask"].dtype == bool


def test_precomputed_never_calls_volume_loader() -> None:
    def _boom(_path):
        raise AssertionError("volume_loader must not be called in precomputed mode")

    stack = np.zeros((2, *SIZE), dtype=np.uint8)
    ds = RAOSVolumes(
        _sample_records(), root="/unused", organ_id=ORGAN_ID,
        output_size=SIZE, bbox_margin=0.25, volumes=stack, volume_loader=_boom,
    )
    _ = ds[0]  # must not raise


def test_patients_property() -> None:
    assert _nifti_ds().patients == ["pA", "pB"]


def test_unique_patient_records_collapses_slices() -> None:
    slices = [
        {"patient_id": "pA", "source_subset": "Set1", "original_split": "Tr", "slice_index": 1},
        {"patient_id": "pA", "source_subset": "Set1", "original_split": "Tr", "slice_index": 2},
        {"patient_id": "pB", "source_subset": "Set1", "original_split": "Ts", "slice_index": 9},
    ]
    recs = unique_patient_records(slices)
    assert [r["patient_id"] for r in recs] == ["pA", "pB"]
    assert recs[1]["original_split"] == "Ts"
    assert "slice_index" not in recs[0]  # collapsed to path metadata only
