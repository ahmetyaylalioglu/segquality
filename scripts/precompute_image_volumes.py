"""Precompute every liver volume WITH its aligned CT image, once (image-channel workstream).

The image-carrying analogue of ``precompute_volumes.py``. Each patient's mask (via
``extract_organ_volume``) and CT (via ``extract_organ_image``, same bbox) are cube-cropped
and resized to ``vae3d.input_size`` and stored together in
``{data.image_volumes_dir}/{split}.npz`` (keys ``volumes`` uint8 + ``images`` float32,
1:1 aligned to ``patient_id``). With ``data.data_source: precomputed`` the
``RAOSImageVolumes`` dataset serves both from memory, no NIfTI reads at train time.

Because ``RAOSImageVolumes`` shares the exact index + split of ``RAOSVolumes``, the
patient order here is identical to the mask-only archive — so image-conditioned runs are
directly comparable to the mask-only experiments on the same volumes.

Usage
-----
    python scripts/precompute_image_volumes.py --config configs/vae3d.yaml
    python scripts/precompute_image_volumes.py --config configs/vae3d.yaml --splits val
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import load_config  # noqa: E402
from data.raos_image_volume import RAOSImageVolumes, image_volumes_path_for  # noqa: E402

logger = logging.getLogger("precompute_image_volumes")

DEFAULT_SPLITS = ("train", "val", "test")


def _nifti_config(config: dict) -> dict:
    """Shallow copy forced onto the NIfTI source (never read our own precomputed output)."""
    return {**config, "data": {**config.get("data", {}), "data_source": "nifti"}}


def build_split(config: dict, split: str) -> Path:
    """Extract every mask+image of ``split`` and write ``{split}.npz``. Returns the path."""
    ds = RAOSImageVolumes.from_config(_nifti_config(config), split=split, group="train")
    n = len(ds)
    d, h, w = ds.output_size
    volumes = np.zeros((n, d, h, w), dtype=np.uint8)
    images = np.zeros((n, d, h, w), dtype=np.float32)
    patient_id = np.empty(n, dtype=object)
    source_subset = np.empty(n, dtype=object)
    original_split = np.empty(n, dtype=object)

    for i in range(n):
        sample = ds[i]
        volumes[i] = sample["mask"].astype(np.uint8)
        images[i] = sample["image"].astype(np.float32)
        patient_id[i] = sample["patient_id"]
        source_subset[i] = sample["meta"]["source_subset"]
        original_split[i] = sample["meta"]["original_split"]
        logger.info("  %s: %d/%d volumes", split, i + 1, n)

    out_path = image_volumes_path_for(config, split)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        volumes=volumes,
        images=images,
        patient_id=patient_id.astype(str),
        source_subset=source_subset.astype(str),
        original_split=original_split.astype(str),
    )
    size_mb = out_path.stat().st_size / 1e6
    logger.info("wrote %s (%d volumes+images, %.1f MB)", out_path, n, size_mb)
    logger.info("  %s image intensity: min=%.3f median=%.3f max=%.3f",
                split, float(images.min()), float(np.median(images)), float(images.max()))
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute liver mask+CT-image volumes per split.")
    parser.add_argument("--config", default="configs/vae3d_vq_sdf_io.yaml",
                        help="path to config YAML (must be 64³ to match the mask experiments)")
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS),
                        help="splits to precompute (default: train val test)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logging.basicConfig(
        level=getattr(logging, config["logging"]["level"], logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for split in args.splits:
        logger.info("=== precomputing image split: %s ===", split)
        build_split(config, split)
    logger.info("done.")


if __name__ == "__main__":
    main()
