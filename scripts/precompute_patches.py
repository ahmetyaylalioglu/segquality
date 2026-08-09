"""Precompute every organ mask patch to a per-split archive, once.

Training reads slices in random order, which under the NIfTI path forces a full
volume re-read per access — the IO bottleneck that dominates epoch time. This
script extracts all 128x128 patches ahead of time (via the *same*
``extract_organ_patch`` used online, so masks are byte-identical) and stores them
as ``outputs/patches/{split}.npz``. With ``dataset.data_source: precomputed`` the
dataset then serves these from memory and training becomes compute-bound.

Each archive holds the patch stack plus per-patch metadata (patient_id,
slice_index, source_subset, organ_area) so evaluation stays traceable.

Usage
-----
    python scripts/precompute_patches.py --config configs/raos.yaml
    python scripts/precompute_patches.py --config configs/raos.yaml --splits train val
    python scripts/precompute_patches.py --config configs/raos.yaml --no-verify
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
from data.raos_dataset import RAOSMaskSlices, patches_path_for  # noqa: E402

logger = logging.getLogger("precompute_patches")

DEFAULT_SPLITS = ("train", "val", "test")
VERIFY_SAMPLES = 50  # random patches re-checked against the NIfTI path per split


def _nifti_config(config: dict) -> dict:
    """A shallow copy forced onto the NIfTI source (so precompute never reads its
    own precomputed output, whatever ``data_source`` the config declares)."""
    return {**config, "dataset": {**config["dataset"], "data_source": "nifti"}}


def build_split(config: dict, split: str) -> Path:
    """Extract every patch of ``split`` and write ``{split}.npz``. Returns the path."""
    ds = RAOSMaskSlices.from_config(_nifti_config(config), split=split, group="train")
    n = len(ds)
    h, w = ds.output_size
    masks = np.zeros((n, h, w), dtype=np.uint8)
    patient_id = np.empty(n, dtype=object)
    slice_index = np.zeros(n, dtype=np.int64)
    source_subset = np.empty(n, dtype=object)
    organ_area = np.zeros(n, dtype=np.int64)

    for i in range(n):
        sample = ds[i]
        masks[i] = sample["mask"].astype(np.uint8)
        patient_id[i] = sample["patient_id"]
        slice_index[i] = sample["slice_index"]
        source_subset[i] = sample["source_subset"]
        organ_area[i] = int(sample["organ_area"])
        if (i + 1) % 1000 == 0:
            logger.info("  %s: %d/%d patches", split, i + 1, n)

    out_path = patches_path_for(config, split, group="train")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        masks=masks,
        patient_id=patient_id.astype(str),
        slice_index=slice_index,
        source_subset=source_subset.astype(str),
        organ_area=organ_area,
    )
    size_mb = out_path.stat().st_size / 1e6
    logger.info("wrote %s (%d patches, %.1f MB)", out_path, n, size_mb)
    return out_path


def verify_split(config: dict, split: str, n_samples: int, seed: int) -> None:
    """Assert precomputed patches match the NIfTI path exactly on a random sample."""
    nifti = RAOSMaskSlices.from_config(_nifti_config(config), split=split, group="train")
    pre = RAOSMaskSlices.from_precomputed(config, split=split, group="train")
    assert len(nifti) == len(pre), f"{split}: length mismatch {len(nifti)} != {len(pre)}"

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(nifti), size=min(n_samples, len(nifti)), replace=False)
    mismatches = 0
    for i in map(int, idx):
        a, b = nifti[i], pre[i]
        if not np.array_equal(a["mask"], b["mask"]):
            mismatches += 1
        assert a["mask"].dtype == b["mask"].dtype, f"{split}[{i}]: dtype differs"
        assert a["patient_id"] == b["patient_id"], f"{split}[{i}]: patient_id differs"
        assert a["slice_index"] == b["slice_index"], f"{split}[{i}]: slice_index differs"
        assert a["organ_area"] == b["organ_area"], f"{split}[{i}]: organ_area differs"
    assert mismatches == 0, f"{split}: {mismatches}/{len(idx)} mask mismatches!"
    logger.info("verify %s: %d/%d sampled patches byte-identical to NIfTI path",
                split, len(idx), len(idx))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute organ mask patches per split.")
    parser.add_argument("--config", default="configs/raos.yaml", help="path to config YAML")
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS),
                        help="splits to precompute (default: train val test)")
    parser.add_argument("--no-verify", action="store_true", help="skip the NIfTI parity check")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logging.basicConfig(
        level=getattr(logging, config["logging"]["level"], logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    seed = int(config.get("seed", 42))
    for split in args.splits:
        logger.info("=== precomputing split: %s ===", split)
        build_split(config, split)
        if not args.no_verify:
            verify_split(config, split, VERIFY_SAMPLES, seed)
    logger.info("done.")


if __name__ == "__main__":
    main()
