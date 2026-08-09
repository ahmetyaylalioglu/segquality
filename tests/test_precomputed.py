"""Tests for the precomputed patch mode of RAOSMaskSlices.

No real RAOS data or nibabel is needed: a synthetic volume drives the NIfTI path
via an injected ``volume_loader``, and the precomputed path is fed the masks that
path produces. The key guarantee under test is *parity* — index ``i`` yields a
byte-identical sample whether served from a volume or from an in-memory patch
stack — which is exactly what ``scripts/precompute_patches.py`` asserts on real
data. (That real-data parity is verified by running the script; here we pin the
code path itself so a refactor can't silently diverge the two sources.)
"""
from __future__ import annotations

import numpy as np

from data.raos_dataset import RAOSMaskSlices

ORGAN_ID = 1
AXIS = 2
OUTPUT_SIZE = (32, 32)
BBOX_MARGIN = 0.25


def make_volume(n_slices: int = 5) -> np.ndarray:
    vol = np.zeros((40, 40, n_slices), dtype=np.int16)
    yy, xx = np.ogrid[:40, :40]
    for s in range(n_slices):
        r = 8 + s  # varying radius so slices differ
        vol[:, :, s][(yy - 20) ** 2 + (xx - 20) ** 2 <= r**2] = ORGAN_ID
    return vol


def make_samples(n: int) -> list[dict]:
    return [
        {
            "patient_id": "Set1_0001",
            "slice_index": s,
            "source_subset": "CancerImages(Set1)",
            "original_split": "Tr",
            "organ_area": int(100 + s),
        }
        for s in range(n)
    ]


def make_nifti_ds(n: int) -> RAOSMaskSlices:
    vol = make_volume(n)
    return RAOSMaskSlices(
        make_samples(n),
        root="/fake",
        organ_id=ORGAN_ID,
        axis=AXIS,
        output_size=OUTPUT_SIZE,
        bbox_margin=BBOX_MARGIN,
        volume_loader=lambda _p: vol,
    )


def stack_patches(ds: RAOSMaskSlices) -> np.ndarray:
    return np.stack([np.asarray(ds[i]["mask"], dtype=np.uint8) for i in range(len(ds))])


def test_precomputed_serves_patches_as_bool_with_full_metadata() -> None:
    n = 4
    patches = (np.random.default_rng(0).random((n, *OUTPUT_SIZE)) > 0.5).astype(np.uint8)
    ds = RAOSMaskSlices(
        make_samples(n),
        root="/fake",
        organ_id=ORGAN_ID,
        axis=AXIS,
        output_size=OUTPUT_SIZE,
        bbox_margin=BBOX_MARGIN,
        patches=patches,
    )
    assert len(ds) == n
    for i in range(n):
        sample = ds[i]
        assert sample["mask"].dtype == bool
        assert np.array_equal(sample["mask"], patches[i].astype(bool))
        assert set(sample) == {"mask", "patient_id", "slice_index", "source_subset", "organ_area"}
        assert sample["slice_index"] == i


def test_precomputed_matches_nifti_path_exactly() -> None:
    n = 5
    nifti = make_nifti_ds(n)
    patches = stack_patches(nifti)  # what precompute would write
    pre = RAOSMaskSlices(
        make_samples(n),
        root="/fake",
        organ_id=ORGAN_ID,
        axis=AXIS,
        output_size=OUTPUT_SIZE,
        bbox_margin=BBOX_MARGIN,
        patches=patches,
    )
    assert len(nifti) == len(pre)
    for i in range(n):
        a, b = nifti[i], pre[i]
        assert np.array_equal(a["mask"], b["mask"])
        assert a["mask"].dtype == b["mask"].dtype  # both bool
        assert a["patient_id"] == b["patient_id"]
        assert a["slice_index"] == b["slice_index"]
        assert a["organ_area"] == b["organ_area"]


def test_precomputed_does_not_call_volume_loader() -> None:
    n = 3
    calls = {"n": 0}

    def counting_loader(_p):  # pragma: no cover - should never run
        calls["n"] += 1
        return make_volume(n)

    patches = (np.random.default_rng(1).random((n, *OUTPUT_SIZE)) > 0.5).astype(np.uint8)
    ds = RAOSMaskSlices(
        make_samples(n),
        root="/fake",
        organ_id=ORGAN_ID,
        axis=AXIS,
        output_size=OUTPUT_SIZE,
        bbox_margin=BBOX_MARGIN,
        volume_loader=counting_loader,
        patches=patches,
    )
    _ = [ds[i] for i in range(n)]
    assert calls["n"] == 0  # precomputed path never touches a volume


def test_precomputed_patients_property() -> None:
    n = 3
    patches = np.zeros((n, *OUTPUT_SIZE), dtype=np.uint8)
    ds = RAOSMaskSlices(
        make_samples(n),
        root="/fake",
        organ_id=ORGAN_ID,
        axis=AXIS,
        output_size=OUTPUT_SIZE,
        bbox_margin=BBOX_MARGIN,
        patches=patches,
    )
    assert ds.patients == ["Set1_0001"]
