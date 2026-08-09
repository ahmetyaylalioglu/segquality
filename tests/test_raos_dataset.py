"""Tests for data.raos_dataset.

The dataset's IO (index/splits/NIfTI reading) is separated from its core, so the
core (__getitem__, LRU cache, slice extraction) is tested here against a
synthetic in-memory volume via an injected ``volume_loader`` — no raw RAOS data
or nibabel required. The pure path/filter helpers are tested directly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from data.raos_dataset import (
    RAOSMaskSlices,
    filter_by_patients,
    label_path_for,
)


ORGAN_ID = 1
AXIS = 2
OUTPUT_SIZE = (128, 128)
BBOX_MARGIN = 0.25


def make_volume(n_slices: int = 5) -> np.ndarray:
    """A (40, 40, n_slices) volume with a liver-labelled disk on each slice."""
    vol = np.zeros((40, 40, n_slices), dtype=np.int16)
    yy, xx = np.ogrid[:40, :40]
    disk = (yy - 20) ** 2 + (xx - 20) ** 2 <= 10**2
    for s in range(n_slices):
        vol[:, :, s][disk] = ORGAN_ID
    return vol


def make_record(patient_id: str, slice_index: int = 0) -> dict:
    return {
        "patient_id": patient_id,
        "source_subset": "CancerImages(Set1)",
        "original_split": "Tr",
        "slice_index": slice_index,
        "organ_area": 317,
    }


def make_dataset(records, counter=None):
    """Dataset over an injected loader that returns a synthetic volume."""
    calls = counter if counter is not None else []

    def loader(path: Path) -> np.ndarray:
        calls.append(str(path))
        return make_volume()

    ds = RAOSMaskSlices(
        records,
        root=Path("/fake/root"),
        organ_id=ORGAN_ID,
        axis=AXIS,
        output_size=OUTPUT_SIZE,
        bbox_margin=BBOX_MARGIN,
        cache_size=2,
        volume_loader=loader,
    )
    return ds, calls


# --- pure helpers ----------------------------------------------------------
def test_label_path_reconstruction():
    rec = make_record("PID123")
    path = label_path_for(Path("/data/RAOS"), rec)
    assert path == Path("/data/RAOS/CancerImages(Set1)/labelsTr/PID123.nii.gz")


def test_filter_by_patients_keeps_order_and_membership():
    slices = [make_record("a", 0), make_record("b", 1), make_record("a", 2)]
    kept = filter_by_patients(slices, {"a"})
    assert [s["slice_index"] for s in kept] == [0, 2]


# --- dataset protocol ------------------------------------------------------
def test_len_matches_samples():
    ds, _ = make_dataset([make_record("a", 0), make_record("a", 1)])
    assert len(ds) == 2


def test_getitem_returns_binary_mask_and_metadata():
    ds, _ = make_dataset([make_record("PID9", 2)])
    sample = ds[0]
    mask = sample["mask"]
    assert mask.shape == OUTPUT_SIZE
    assert np.all(np.isin(np.unique(mask), (0, 1)))
    assert mask.any(), "liver disk should survive crop+resize"
    assert sample["patient_id"] == "PID9"
    assert sample["slice_index"] == 2
    assert sample["source_subset"] == "CancerImages(Set1)"


def test_transform_hook_is_applied():
    def add_flag(sample: dict) -> dict:
        sample["flagged"] = True
        return sample

    ds, _ = make_dataset([make_record("a", 0)])
    ds.transform = add_flag
    assert ds[0]["flagged"] is True


# --- LRU cache -------------------------------------------------------------
def test_same_patient_slices_hit_cache():
    # Two slices of the same patient -> volume read exactly once.
    ds, calls = make_dataset([make_record("same", 0), make_record("same", 1)])
    ds[0]
    ds[1]
    assert len(calls) == 1, f"expected 1 volume read, got {len(calls)}"


def test_cache_evicts_beyond_capacity():
    # cache_size=2: reading 3 distinct patients then re-reading the first must
    # trigger a reload (it was evicted).
    records = [make_record("p1"), make_record("p2"), make_record("p3")]
    ds, calls = make_dataset(records)
    ds[0]  # p1 load
    ds[1]  # p2 load
    ds[2]  # p3 load -> evicts p1
    assert len(calls) == 3
    ds[0]  # p1 reload
    assert len(calls) == 4


def test_patients_property_is_unique_sorted():
    ds, _ = make_dataset([make_record("b", 0), make_record("a", 0), make_record("b", 1)])
    assert ds.patients == ["a", "b"]
