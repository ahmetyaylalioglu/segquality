"""Precompute every liver volume to a per-split archive, once (3D, Phase 7).

The 3D analogue of ``precompute_patches.py``. Each patient's whole label volume
is cube-cropped and nearest-resized to ``vae3d.input_size`` (via the *same*
``extract_organ_volume`` used online, so volumes are byte-identical) and stored as
``{data.volumes_dir}/{split}.npz``. With ``data.data_source: precomputed`` the
dataset then serves these from memory.

3D-specific health check
-------------------------
3D nearest resize is far more lossy than 2D: a liver 40-80 slices tall is
squeezed to 64 (or 32). So besides the NIfTI parity check, this logs the
post-resize **occupancy** (filled voxels / total) distribution per split. A very
low minimum (<1%) means some patient's organ is melting away in the resize —
a cue to raise ``input_size`` or flag those patients before training on garbage.

Usage
-----
    python scripts/precompute_volumes.py --config configs/vae3d.yaml
    python scripts/precompute_volumes.py --config configs/vae3d.yaml --splits val
    python scripts/precompute_volumes.py --config configs/vae3d.yaml --no-verify
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
from data.raos_volume import RAOSVolumes, volumes_path_for  # noqa: E402

logger = logging.getLogger("precompute_volumes")

DEFAULT_SPLITS = ("train", "val", "test")
VERIFY_SAMPLES = 20      # random volumes re-checked against the NIfTI path per split
LOW_OCCUPANCY = 0.01     # occupancy below this flags a resize-melt risk


def _nifti_config(config: dict) -> dict:
    """A shallow copy forced onto the NIfTI source (so precompute never reads its
    own precomputed output, whatever ``data.data_source`` the config declares)."""
    return {**config, "data": {**config.get("data", {}), "data_source": "nifti"}}


def _log_occupancy(split: str, occupancy: np.ndarray, patient_id: np.ndarray) -> None:
    """Log the post-resize occupancy distribution and flag melting volumes."""
    logger.info(
        "  %s occupancy (filled/total): min=%.4f  median=%.4f  max=%.4f",
        split, float(occupancy.min()), float(np.median(occupancy)), float(occupancy.max()),
    )
    low = np.where(occupancy < LOW_OCCUPANCY)[0]
    if low.size:
        logger.warning(
            "  %s: %d/%d volumes below %.0f%% occupancy (resize-melt risk):",
            split, low.size, occupancy.size, LOW_OCCUPANCY * 100,
        )
        for i in map(int, low):
            logger.warning("    %s -> %.4f", patient_id[i], float(occupancy[i]))
    else:
        logger.info("  %s: all volumes >= %.0f%% occupancy (no resize-melt)",
                    split, LOW_OCCUPANCY * 100)


def build_split(config: dict, split: str) -> Path:
    """Extract every volume of ``split`` and write ``{split}.npz``. Returns the path."""
    ds = RAOSVolumes.from_config(_nifti_config(config), split=split, group="train")
    n = len(ds)
    d, h, w = ds.output_size
    volumes = np.zeros((n, d, h, w), dtype=np.uint8)
    patient_id = np.empty(n, dtype=object)
    source_subset = np.empty(n, dtype=object)
    original_split = np.empty(n, dtype=object)
    occupancy = np.zeros(n, dtype=np.float64)

    for i in range(n):
        sample = ds[i]
        mask = sample["mask"]
        volumes[i] = mask.astype(np.uint8)
        patient_id[i] = sample["patient_id"]
        source_subset[i] = sample["meta"]["source_subset"]
        original_split[i] = sample["meta"]["original_split"]
        occupancy[i] = float(mask.sum()) / mask.size
        logger.info("  %s: %d/%d volumes", split, i + 1, n)

    out_path = volumes_path_for(config, split)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        volumes=volumes,
        patient_id=patient_id.astype(str),
        source_subset=source_subset.astype(str),
        original_split=original_split.astype(str),
        occupancy=occupancy,
    )
    size_mb = out_path.stat().st_size / 1e6
    logger.info("wrote %s (%d volumes, %.1f MB)", out_path, n, size_mb)
    _log_occupancy(split, occupancy, patient_id)
    return out_path


def verify_split(config: dict, split: str, n_samples: int, seed: int) -> None:
    """Assert precomputed volumes match the NIfTI path exactly on a random sample."""
    nifti = RAOSVolumes.from_config(_nifti_config(config), split=split, group="train")
    pre = RAOSVolumes.from_precomputed(config, split=split, group="train")
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
    assert mismatches == 0, f"{split}: {mismatches}/{len(idx)} volume mismatches!"
    logger.info("verify %s: %d/%d sampled volumes byte-identical to NIfTI path",
                split, len(idx), len(idx))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute liver volumes per split (3D).")
    parser.add_argument("--config", default="configs/vae3d.yaml", help="path to config YAML")
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
