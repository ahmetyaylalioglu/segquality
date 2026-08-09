"""Real-liver mask-slice dataset built on the precomputed per-slice index.

``RAOSMaskSlices`` turns one entry of ``outputs/index_liver_{group}.json`` into a
resized 2D organ mask on demand. It never rescans the volumes: the index already
holds (patient, slice_index, bbox, ...), so construction is cheap and only the
NIfTI volume of a requested slice is read (and cached).

Design
------
* IO is separated from the core. :meth:`from_config` does all file access
  (reads the index, loads the patient split, resolves paths); ``__init__`` takes
  an explicit sample list plus geometry params and an injectable
  ``volume_loader``. That keeps ``__getitem__`` / the LRU cache unit-testable
  with a synthetic volume, no raw data or nibabel required.
* Cropping reuses :func:`data.transforms.extract_organ_patch` verbatim — the same
  bbox-centred crop+resize proven on real anatomy in the degradation work, not a
  re-implementation.
* No torch dependency: this is a plain ``__len__`` / ``__getitem__`` class so it
  drops into a ``DataLoader`` later without pulling torch in now (no model code
  exists yet, by design).
"""
from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from data.splits import load_splits
from data.transforms import extract_organ_patch

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NIFTI_SUFFIX = ".nii.gz"

VolumeLoader = Callable[[Path], np.ndarray]


def _default_volume_loader(path: Path) -> np.ndarray:
    """Load a NIfTI label volume as int16 (lazy dataobj, no float64 blow-up)."""
    import nibabel as nib

    return np.asanyarray(nib.load(str(path)).dataobj).astype(np.int16)


def index_path_for(config: dict[str, Any], organ: str, group: str) -> Path:
    """Resolve ``outputs/index_{organ}_{group}.json`` against the project root."""
    template = config["cache"]["index_path"].format(organ=organ)
    base = PROJECT_ROOT / template
    return base.with_name(f"{base.stem}_{group}{base.suffix}")


def patches_path_for(config: dict[str, Any], split: str, group: str = "train") -> Path:
    """Resolve the precomputed-patch archive ``outputs/patches/{split}.npz``.

    The directory comes from ``cache.patches_dir`` (defaulting to
    ``outputs/patches``). Only the train group is precomputed, so ``group`` is
    accepted for symmetry but not encoded in the filename.
    """
    patches_dir = config.get("cache", {}).get("patches_dir", "outputs/patches")
    return PROJECT_ROOT / patches_dir / f"{split}.npz"


def label_path_for(root: Path, record: dict[str, Any]) -> Path:
    """Reconstruct a slice's NIfTI path from its index record.

    The index stores ``source_subset`` and ``original_split`` (Tr/Val/Ts) rather
    than the full path; the label dir is ``labels`` + that split.
    """
    label_dir = "labels" + record["original_split"]
    return root / record["source_subset"] / label_dir / (record["patient_id"] + NIFTI_SUFFIX)


def filter_by_patients(
    slices: list[dict[str, Any]], patient_ids: set[str]
) -> list[dict[str, Any]]:
    """Keep only slice records whose patient is in ``patient_ids`` (order kept)."""
    return [s for s in slices if s["patient_id"] in patient_ids]


class RAOSMaskSlices:
    """A split of real organ mask slices, resized to ``output_size``.

    Indexable like a torch ``Dataset``: ``len(ds)`` slices, ``ds[i]`` returns a
    dict ``{"mask", "patient_id", "slice_index", "source_subset", "organ_area"}``.
    Volumes are read lazily and kept in a small LRU cache (consecutive slices of
    the same patient hit the cache instead of reopening the file).
    """

    def __init__(
        self,
        samples: list[dict[str, Any]],
        *,
        root: Path,
        organ_id: int,
        axis: int,
        output_size: tuple[int, int],
        bbox_margin: float,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache_size: int = 4,
        volume_loader: VolumeLoader | None = None,
        patches: np.ndarray | None = None,
    ) -> None:
        self.samples = samples
        self.root = Path(root)
        self.organ_id = int(organ_id)
        self.axis = int(axis)
        self.output_size = (int(output_size[0]), int(output_size[1]))
        self.bbox_margin = float(bbox_margin)
        self.transform = transform
        self.cache_size = int(cache_size)
        self.volume_loader = volume_loader or _default_volume_loader
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        # Precomputed mode: an in-memory (N, H, W) stack of already-extracted
        # patches aligned 1:1 with ``samples``. When set, ``__getitem__`` serves
        # from it and never touches a volume (see ``from_precomputed``). The NIfTI
        # path is untouched and remains the default (``patches is None``).
        self.patches = patches

    # -- IO-side constructor ------------------------------------------------
    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        split: str | None,
        group: str = "train",
        **kwargs: Any,
    ) -> "RAOSMaskSlices":
        """Build a split from the on-disk index + patient split file.

        ``split`` is one of train/val/test (filtered via ``splits.json``); pass
        ``None`` to use every patient in the index (e.g. the heldout group, which
        has no train/val/test split).

        If ``dataset.data_source == "precomputed"`` the split is served from the
        precomputed patch archive instead of the NIfTI volumes; the NIfTI path is
        the default and is otherwise unchanged.
        """
        source = config.get("dataset", {}).get("data_source", "nifti")
        if source == "precomputed":
            return cls.from_precomputed(config, split, group, **kwargs)

        organ = config["target"]
        index_path = index_path_for(config, organ, group)
        with index_path.open(encoding="utf-8") as fh:
            index = json.load(fh)
        meta = index["metadata"]
        samples = index["slices"]

        if split is not None:
            splits_path = PROJECT_ROOT / config["cache"]["splits_path"]
            patient_ids = set(load_splits(splits_path)[split])
            samples = filter_by_patients(samples, patient_ids)

        logger.info(
            "RAOSMaskSlices(group=%s, split=%s): %d slices from %d patients",
            group, split, len(samples), len({s["patient_id"] for s in samples}),
        )
        return cls(
            samples,
            root=Path(config["dataset"]["root"]).expanduser(),
            organ_id=int(meta["organ_id"]),
            axis=int(meta["axis"]),
            output_size=tuple(config["slicing"]["output_size"]),
            bbox_margin=float(config["slicing"]["bbox_margin"]),
            **kwargs,
        )

    @classmethod
    def from_precomputed(
        cls,
        config: dict[str, Any],
        split: str,
        group: str = "train",
        **kwargs: Any,
    ) -> "RAOSMaskSlices":
        """Build a split from the precomputed patch archive (no NIfTI reads).

        The archive (written by ``scripts/precompute_patches.py``) already holds
        every 128x128 patch plus its metadata, so this is memory-resident and
        fast. Sample order matches the NIfTI path exactly (both iterate the index
        in stored order), so a given index maps to the same mask in either mode.
        """
        archive_path = patches_path_for(config, split, group)
        with np.load(archive_path, allow_pickle=False) as data:
            masks = data["masks"]
            samples = [
                {
                    "patient_id": str(pid),
                    "slice_index": int(sidx),
                    "source_subset": str(subset),
                    "organ_area": int(area),
                }
                for pid, sidx, subset, area in zip(
                    data["patient_id"],
                    data["slice_index"],
                    data["source_subset"],
                    data["organ_area"],
                )
            ]
        logger.info(
            "RAOSMaskSlices(precomputed, split=%s): %d patches from %d patients",
            split, len(samples), len({s["patient_id"] for s in samples}),
        )
        return cls(
            samples,
            root=Path(config["dataset"]["root"]).expanduser(),
            organ_id=int(config["organs"][config["target"]]),
            axis=int(config["slicing"]["axis"]),
            output_size=tuple(config["slicing"]["output_size"]),
            bbox_margin=float(config["slicing"]["bbox_margin"]),
            patches=masks,
            **kwargs,
        )

    # -- dataset protocol ---------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.samples[index]
        if self.patches is not None:
            # Precomputed: the patch is already extracted; return a bool mask so
            # the sample is byte-identical to the NIfTI path (which returns bool).
            mask = self.patches[index].astype(bool)
        else:
            volume = self._get_volume(record)
            slice_2d = np.take(volume, record["slice_index"], axis=self.axis) == self.organ_id
            mask = extract_organ_patch(slice_2d, self.output_size, self.bbox_margin)
        sample = {
            "mask": mask,
            "patient_id": record["patient_id"],
            "slice_index": record["slice_index"],
            "source_subset": record["source_subset"],
            "organ_area": record.get("organ_area"),
        }
        return self.transform(sample) if self.transform else sample

    # -- helpers ------------------------------------------------------------
    @property
    def patients(self) -> list[str]:
        return sorted({s["patient_id"] for s in self.samples})

    def _get_volume(self, record: dict[str, Any]) -> np.ndarray:
        """Return the patient's volume, using an LRU cache of the last N volumes."""
        key = str(label_path_for(self.root, record))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        volume = self.volume_loader(Path(key))
        self._cache[key] = volume
        self._cache.move_to_end(key)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return volume
