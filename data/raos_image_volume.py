"""Real liver volumes served WITH the aligned grayscale CT image (image-channel workstream).

The image-carrying sibling of :class:`data.raos_volume.RAOSVolumes`. It subclasses that
class (encoder/split/precompute machinery inherited verbatim, never edited) and adds a
second field ``image``: the CT cropped to the SAME liver bbox and resized to the same
64³ grid as the mask, so ``sample["image"]`` and ``sample["mask"]`` are voxel-aligned.

``ds[i]`` -> ``{"mask": (D,H,W) bool, "image": (D,H,W) float32 in [0,1], "patient_id", "meta"}``.

Two modes, matching the parent:
* **nifti** — the label volume gives both the mask (via ``extract_organ_volume``) and the
  bbox; the CT is loaded via :func:`data.raos_image.image_path_for` and cropped with
  :func:`data.raos_image.extract_organ_image` using that same full-res binary.
* **precomputed** — both arrays are read from ``{image_volumes_dir}/{split}.npz`` (keys
  ``volumes`` + ``images``, 1:1 aligned to ``patient_id``), no NIfTI read.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from data.raos_dataset import PROJECT_ROOT, VolumeLoader, filter_by_patients, index_path_for
from data.raos_image import WINDOW_HI, WINDOW_LO, extract_organ_image, image_path_for, load_ct_volume
from data.raos_volume import RAOSVolumes, _bbox_margin_3d, unique_patient_records
from data.splits import load_splits
from data.volume_transforms import extract_organ_volume

import json
import logging

logger = logging.getLogger(__name__)


def image_volumes_path_for(config: dict[str, Any], split: str) -> Path:
    """Resolve the precomputed image+mask archive ``{image_volumes_dir}/{split}.npz``."""
    image_dir = config.get("data", {}).get("image_volumes_dir", "outputs/image_volumes")
    return PROJECT_ROOT / image_dir / f"{split}.npz"


class RAOSImageVolumes(RAOSVolumes):
    """A split of liver volumes, each with its aligned grayscale CT image channel."""

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
        image_loader: VolumeLoader | None = None,
        volumes: np.ndarray | None = None,
        images: np.ndarray | None = None,
        window: tuple[float, float] = (WINDOW_LO, WINDOW_HI),
    ) -> None:
        # Parent handles mask machinery; we apply our own transform after adding image.
        super().__init__(
            samples, root=root, organ_id=organ_id, output_size=output_size,
            bbox_margin=bbox_margin, transform=None, cache_size=cache_size,
            volume_loader=volume_loader, volumes=volumes,
        )
        self.image_loader = image_loader or load_ct_volume
        self.images = images
        self.window = window
        self._img_transform = transform
        self._img_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

    # -- IO-side constructors ----------------------------------------------
    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        split: str | None,
        group: str = "train",
        **kwargs: Any,
    ) -> "RAOSImageVolumes":
        """Build from the index + patient split (nifti) or the precomputed archive."""
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
        logger.info("RAOSImageVolumes(group=%s, split=%s): %d volumes", group, split, len(samples))
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
    ) -> "RAOSImageVolumes":
        """Build from the precomputed image+mask archive (no NIfTI reads)."""
        archive_path = image_volumes_path_for(config, split)
        with np.load(archive_path, allow_pickle=False) as data:
            volumes = data["volumes"]
            images = data["images"]
            samples = [
                {"patient_id": str(pid), "source_subset": str(subset), "original_split": str(osplit)}
                for pid, subset, osplit in zip(
                    data["patient_id"], data["source_subset"], data["original_split"]
                )
            ]
        logger.info("RAOSImageVolumes(precomputed, split=%s): %d volumes", split, len(samples))
        return cls(
            samples,
            root=Path(config["dataset"]["root"]).expanduser(),
            organ_id=int(config["organs"][config["target"]]),
            output_size=tuple(config["vae3d"]["input_size"]),
            bbox_margin=_bbox_margin_3d(config),
            volumes=volumes,
            images=images,
            **kwargs,
        )

    # -- dataset protocol ---------------------------------------------------
    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.samples[index]
        if self.volumes is not None:
            mask = self.volumes[index].astype(bool)
            image = self.images[index].astype(np.float32)
        else:
            volume = self._get_volume(record)
            binary = volume == self.organ_id
            mask = extract_organ_volume(binary, self.output_size, self.bbox_margin)
            ct = self._get_ct(record)
            image = extract_organ_image(
                ct, binary, self.output_size, self.bbox_margin, window=self.window
            )
        sample = {
            "mask": mask,
            "image": image,
            "patient_id": record["patient_id"],
            "meta": {
                "source_subset": record["source_subset"],
                "original_split": record["original_split"],
            },
        }
        return self._img_transform(sample) if self._img_transform else sample

    # -- helpers ------------------------------------------------------------
    def _get_ct(self, record: dict[str, Any]) -> np.ndarray:
        """Return the patient's CT volume, using a small LRU cache."""
        key = str(image_path_for(self.root, record))
        if key in self._img_cache:
            self._img_cache.move_to_end(key)
            return self._img_cache[key]
        volume = self.image_loader(Path(key))
        self._img_cache[key] = volume
        self._img_cache.move_to_end(key)
        if len(self._img_cache) > self.cache_size:
            self._img_cache.popitem(last=False)
        return volume
