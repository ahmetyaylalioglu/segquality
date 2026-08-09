"""Real-liver whole-*volume* dataset for the 3D VAE (Phase 7).

The volumetric sibling of :class:`data.raos_dataset.RAOSMaskSlices`. There, one
sample is a 2D slice; here, one sample is a whole liver **volume** — so 317 Set1
patients become 317 examples (vs ~16.5k slices in 2D). That scarcity is the core
3D challenge and is why the split must be reused verbatim.

Design (mirrors the 2D class)
-----------------------------
* IO is separated from the core: :meth:`from_config` reads the index + patient
  split and resolves label paths; ``__init__`` takes an explicit sample list plus
  an injectable ``volume_loader``, so ``__getitem__`` is unit-testable with a
  synthetic volume (no nibabel / raw data).
* The **same** patient split (``splits.json``) as 2D is used, so a given patient
  lands in the same split in both pipelines and the 2D-vs-3D comparison is fair.
* Cropping reuses :func:`data.volume_transforms.extract_organ_volume` (cube-crop
  + nearest resize), never a re-implementation.
* ``input_size`` is read from the single source ``vae3d.input_size`` — the model
  and the data resize target can never silently disagree.
"""
from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from data.raos_dataset import (
    PROJECT_ROOT,
    VolumeLoader,
    _default_volume_loader,
    filter_by_patients,
    index_path_for,
    label_path_for,
)
from data.splits import load_splits
from data.volume_transforms import extract_organ_volume

logger = logging.getLogger(__name__)


def volumes_path_for(config: dict[str, Any], split: str) -> Path:
    """Resolve the precomputed-volume archive ``{volumes_dir}/{split}.npz``."""
    volumes_dir = config.get("data", {}).get("volumes_dir", "outputs/volumes")
    return PROJECT_ROOT / volumes_dir / f"{split}.npz"


def _bbox_margin_3d(config: dict[str, Any]) -> float:
    """The 3D bbox margin: ``vae3d.bbox_margin`` if set, else 2D ``slicing.bbox_margin``.

    3D uses a *smaller* margin than 2D (0.10 vs 0.25): the whole organ is already
    captured so little context padding is needed, and a smaller margin raises the
    post-squash occupancy. The small non-zero pad is deliberate — it keeps a dilate
    degradation from spilling past the volume edge (which would corrupt the true
    Dice) and avoids clipping edge voxels on resize.
    """
    vae3d = config.get("vae3d", {})
    if "bbox_margin" in vae3d:
        return float(vae3d["bbox_margin"])
    return float(config["slicing"]["bbox_margin"])


def unique_patient_records(slices: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One record per patient (first occurrence), keeping the path metadata.

    The per-slice index lists every organ-bearing slice; for 3D we only need each
    patient's label file once (the whole volume is loaded and the organ extracted
    in 3D), so we collapse to the fields :func:`label_path_for` needs.
    """
    seen: dict[str, dict[str, Any]] = OrderedDict()
    for s in slices:
        pid = s["patient_id"]
        if pid not in seen:
            seen[pid] = {
                "patient_id": pid,
                "source_subset": s["source_subset"],
                "original_split": s["original_split"],
            }
    return list(seen.values())


class RAOSVolumes:
    """A split of real liver volumes, cube-cropped and resized to ``output_size``.

    Indexable like a torch ``Dataset``: ``ds[i]`` returns
    ``{"mask": (D,H,W) bool, "patient_id", "meta": {...}}``. Volumes are read
    lazily and kept in a small LRU cache; in precomputed mode they are served from
    an in-memory stack and no volume is ever read.
    """

    def __init__(
        self,
        samples: list[dict[str, Any]],
        *,
        root: Path,
        organ_id: int,
        output_size: tuple[int, int, int],
        bbox_margin: float,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache_size: int = 4,
        volume_loader: VolumeLoader | None = None,
        volumes: np.ndarray | None = None,
    ) -> None:
        self.samples = samples
        self.root = Path(root)
        self.organ_id = int(organ_id)
        self.output_size = (int(output_size[0]), int(output_size[1]), int(output_size[2]))
        self.bbox_margin = float(bbox_margin)
        self.transform = transform
        self.cache_size = int(cache_size)
        self.volume_loader = volume_loader or _default_volume_loader
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        # Precomputed mode: an (N, D, H, W) stack aligned 1:1 with ``samples``.
        self.volumes = volumes

    # -- IO-side constructor ------------------------------------------------
    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        split: str | None,
        group: str = "train",
        **kwargs: Any,
    ) -> "RAOSVolumes":
        """Build a split from the index + patient split (or the precomputed archive).

        ``dataset.data_source`` is 2D-specific; the 3D source is ``data.data_source``
        ("nifti" default, or "precomputed").
        """
        source = config.get("data", {}).get("data_source", "nifti")
        if source == "precomputed":
            return cls.from_precomputed(config, split, group, **kwargs)

        organ = config["target"]
        index_path = index_path_for(config, organ, group)
        with index_path.open(encoding="utf-8") as fh:
            index = json.load(fh)
        samples = unique_patient_records(index["slices"])

        if split is not None:
            splits_path = PROJECT_ROOT / config["cache"]["splits_path"]
            patient_ids = set(load_splits(splits_path)[split])
            samples = filter_by_patients(samples, patient_ids)

        logger.info(
            "RAOSVolumes(group=%s, split=%s): %d volumes (patients)",
            group, split, len(samples),
        )
        return cls(
            samples,
            root=Path(config["dataset"]["root"]).expanduser(),
            organ_id=int(config["organs"][config["target"]]),
            output_size=tuple(config["vae3d"]["input_size"]),
            bbox_margin=_bbox_margin_3d(config),
            **kwargs,
        )

    @classmethod
    def from_precomputed(
        cls,
        config: dict[str, Any],
        split: str,
        group: str = "train",
        **kwargs: Any,
    ) -> "RAOSVolumes":
        """Build a split from the precomputed-volume archive (no NIfTI reads)."""
        archive_path = volumes_path_for(config, split)
        with np.load(archive_path, allow_pickle=False) as data:
            volumes = data["volumes"]
            samples = [
                {
                    "patient_id": str(pid),
                    "source_subset": str(subset),
                    "original_split": str(osplit),
                }
                for pid, subset, osplit in zip(
                    data["patient_id"], data["source_subset"], data["original_split"]
                )
            ]
        logger.info(
            "RAOSVolumes(precomputed, split=%s): %d volumes", split, len(samples)
        )
        return cls(
            samples,
            root=Path(config["dataset"]["root"]).expanduser(),
            organ_id=int(config["organs"][config["target"]]),
            output_size=tuple(config["vae3d"]["input_size"]),
            bbox_margin=_bbox_margin_3d(config),
            volumes=volumes,
            **kwargs,
        )

    # -- dataset protocol ---------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.samples[index]
        if self.volumes is not None:
            mask = self.volumes[index].astype(bool)
        else:
            volume = self._get_volume(record)
            binary = volume == self.organ_id
            mask = extract_organ_volume(binary, self.output_size, self.bbox_margin)
        sample = {
            "mask": mask,
            "patient_id": record["patient_id"],
            "meta": {
                "source_subset": record["source_subset"],
                "original_split": record["original_split"],
            },
        }
        return self.transform(sample) if self.transform else sample

    # -- helpers ------------------------------------------------------------
    @property
    def patients(self) -> list[str]:
        return sorted({s["patient_id"] for s in self.samples})

    def _get_volume(self, record: dict[str, Any]) -> np.ndarray:
        """Return the patient's label volume, using a small LRU cache."""
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
