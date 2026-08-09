"""Precompute a per-slice index of the target organ so the dataset never opens a
full NIfTI volume in ``__getitem__``.

Builds two independent indexes — ``train`` (Set1) and ``heldout`` (Set2+Set3) —
then makes the patient-based train/val/test split over Set1 only. The split is
stored separately (outputs/splits.json); the split name is never embedded in the
index, so changing the split does not require rebuilding the index.

    python scripts/build_index.py --config configs/raos.yaml
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import load_config  # noqa: E402
from data.splits import get_or_make_splits  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("build_index")

NIFTI_SUFFIX = ".nii.gz"
_SANITY_PATIENTS = 3  # how many patients to log axis/spacing for


# --------------------------------------------------------------------------
# Path / file helpers
# --------------------------------------------------------------------------
def patient_id_from_path(path: Path) -> str:
    name = path.name
    return name[: -len(NIFTI_SUFFIX)] if name.endswith(NIFTI_SUFFIX) else path.stem


def original_split_from_dir(label_dir_name: str) -> str:
    if label_dir_name.lower().startswith("labels"):
        return label_dir_name[len("labels") :] or label_dir_name
    return label_dir_name


def index_path_for(config: dict[str, Any], organ: str, group: str) -> Path:
    """Derive outputs/index_{organ}_{group}.json from the config template."""
    template = config["cache"]["index_path"].format(organ=organ)
    base = PROJECT_ROOT / template
    return base.with_name(f"{base.stem}_{group}{base.suffix}")


def collect_label_files(
    root: Path, subsets: list[str], label_dirs: list[str]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for subset in subsets:
        for label_dir in label_dirs:
            folder = root / subset / label_dir
            if not folder.is_dir():
                continue
            for path in sorted(folder.glob(f"*{NIFTI_SUFFIX}")):
                records.append(
                    {
                        "source_subset": subset,
                        "original_split": original_split_from_dir(label_dir),
                        "path": path,
                        "patient_id": patient_id_from_path(path),
                    }
                )
    return records


# --------------------------------------------------------------------------
# Core indexing
# --------------------------------------------------------------------------
def _log_axis_sanity(
    patient_id: str, shape: tuple[int, ...], spacing: tuple[float, ...],
    axis: int, n_slices_with_organ: int,
) -> None:
    """Warn if the chosen slicing axis does not look like the axial (z) axis.

    The axial axis should have the largest spacing (z = 2.5/3.0 mm in RAOS vs
    ~0.977 mm in-plane). If the chosen axis is in-plane, we are slicing the wrong
    way — flagged loudly but not fatal.
    """
    logger.info(
        "  [axis-check] %s shape=%s spacing=%s | axis=%d: %d slices total, "
        "%d with organ, spacing[axis]=%.3f mm",
        patient_id, shape, tuple(round(s, 3) for s in spacing),
        axis, shape[axis], n_slices_with_organ, spacing[axis],
    )
    if spacing[axis] < max(spacing) - 1e-6:
        logger.warning(
            "  [axis-check] spacing[axis]=%.3f is NOT the largest (%.3f) — axis=%d "
            "may be in-plane, not axial. Verify slicing.axis!",
            spacing[axis], max(spacing), axis,
        )


def slice_records_for_volume(
    mask: np.ndarray, axis: int, min_area: int
) -> list[dict[str, Any]]:
    """Per-slice records for slices whose organ area >= min_area.

    ``bbox`` is in the slice's ORIGINAL (pre-resize) coordinates; the dataset
    loader crops+resizes from it. See metadata ``bbox_space: "original"``.
    """
    other_axes = tuple(i for i in range(mask.ndim) if i != axis)
    areas = mask.sum(axis=other_axes)
    keep = np.where(areas >= min_area)[0]

    records: list[dict[str, Any]] = []
    for slice_index in keep.tolist():
        slice_2d = np.take(mask, slice_index, axis=axis)
        ys, xs = np.where(slice_2d)
        bbox = [int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1]
        records.append(
            {
                "slice_index": int(slice_index),
                "organ_area": int(areas[slice_index]),
                "bbox": bbox,
            }
        )
    return records


def build_group_index(
    group: str,
    label_files: list[dict[str, Any]],
    organ: str,
    organ_id: int,
    axis: int,
    min_area: int,
    limit: int | None,
) -> dict[str, Any]:
    import nibabel as nib

    if limit is not None:
        label_files = label_files[:limit]

    logger.info(
        "\n=== Building '%s' index: %d patients, organ=%s(id=%d), axis=%d, "
        "min_area=%d ===",
        group, len(label_files), organ, organ_id, axis, min_area,
    )

    slices: list[dict[str, Any]] = []
    per_patient_counts: list[int] = []
    skipped: list[str] = []

    for i, rec in enumerate(label_files, start=1):
        try:
            img = nib.load(str(rec["path"]))
            volume = np.asanyarray(img.dataobj).astype(np.int16)
            spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
        except Exception as exc:  # noqa: BLE001 - report and skip corrupt files
            logger.warning("skipping unreadable volume %s: %s", rec["path"], exc)
            skipped.append(str(rec["path"]))
            continue

        mask = volume == organ_id
        vol_records = slice_records_for_volume(mask, axis, min_area)
        per_patient_counts.append(len(vol_records))

        if i <= _SANITY_PATIENTS:
            _log_axis_sanity(
                rec["patient_id"], volume.shape, spacing, axis, len(vol_records)
            )

        for r in vol_records:
            slices.append(
                {
                    "patient_id": rec["patient_id"],
                    "subset": group,
                    "source_subset": rec["source_subset"],
                    "original_split": rec["original_split"],
                    "slice_index": r["slice_index"],
                    "organ_area": r["organ_area"],
                    "bbox": r["bbox"],
                    "volume_shape": [int(s) for s in volume.shape],
                    "spacing": [round(s, 6) for s in spacing],
                }
            )
        if i % 25 == 0:
            logger.info("  progress: %d/%d patients", i, len(label_files))

    metadata = {
        "organ": organ,
        "organ_id": organ_id,
        "group": group,
        "axis": axis,
        "min_organ_area": min_area,
        "bbox_space": "original",  # bbox is pre-resize slice coordinates
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "n_patients": len(per_patient_counts),
        "n_slices": len(slices),
        "n_skipped": len(skipped),
        "skipped_files": skipped,
        "source_subsets": sorted({r["source_subset"] for r in label_files}),
    }
    return {"metadata": metadata, "slices": slices, "_counts": per_patient_counts}


def summarize(group: str, index: dict[str, Any]) -> None:
    counts = index.get("_counts") or _counts_from_slices(index["slices"])
    meta = index["metadata"]
    with_slices = [c for c in counts if c > 0]
    logger.info("\n=== Summary: '%s' index ===", group)
    logger.info("  patients            : %d", meta["n_patients"])
    logger.info("  patients w/ slices  : %d", len(with_slices))
    logger.info("  total slices        : %d", meta["n_slices"])
    if with_slices:
        logger.info(
            "  slices/patient      : min=%d median=%d max=%d",
            min(with_slices), int(statistics.median(with_slices)), max(with_slices),
        )
    logger.info("  skipped (unreadable): %d", meta["n_skipped"])


def _counts_from_slices(slices: list[dict[str, Any]]) -> list[int]:
    counts: dict[str, int] = {}
    for s in slices:
        counts[s["patient_id"]] = counts.get(s["patient_id"], 0) + 1
    return list(counts.values())


def write_index(path: Path, index: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": index["metadata"], "slices": index["slices"]}
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("wrote %s", path.resolve())


def load_index(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def process_group(
    group: str, subsets: list[str], config: dict[str, Any], root: Path,
    label_dirs: list[str], organ: str, organ_id: int, axis: int, min_area: int,
    force: bool, limit: int | None,
) -> dict[str, Any]:
    out_path = index_path_for(config, organ, group)
    if out_path.exists() and not force:
        logger.info("\n'%s' index exists at %s (use --force to rebuild)", group, out_path)
        index = load_index(out_path)
        summarize(group, index)
        return index

    label_files = collect_label_files(root, subsets, label_dirs)
    index = build_group_index(group, label_files, organ, organ_id, axis, min_area, limit)
    write_index(out_path, index)
    summarize(group, index)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true", help="rebuild existing indexes")
    parser.add_argument("--limit", type=int, default=None, help="max patients per group")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = config["dataset"]
    root = Path(dataset["root"]).expanduser()
    label_dirs = list(dataset["label_dirs"])

    organ = config["target"]
    organ_id = int(config["organs"][organ])
    axis = int(config["slicing"]["axis"])
    default_area = int(config["slicing"]["min_organ_area"])
    overrides = config["slicing"].get("min_organ_area_overrides", {})
    min_area = int(overrides.get(organ, default_area))

    if not root.is_dir():
        logger.error("dataset root does not exist: %s", root)
        raise SystemExit(1)

    train_index = process_group(
        "train", list(dataset["train_subsets"]), config, root, label_dirs,
        organ, organ_id, axis, min_area, args.force, args.limit,
    )
    process_group(
        "heldout", list(dataset["heldout_subsets"]), config, root, label_dirs,
        organ, organ_id, axis, min_area, args.force, args.limit,
    )

    # Split over Set1 (train group) patients only — Set2/Set3 stay an untouched
    # generalization set and must never enter train/val/test.
    train_patients = sorted({s["patient_id"] for s in train_index["slices"]})
    config["cache"]["splits_path"] = str(
        (PROJECT_ROOT / config["cache"]["splits_path"]).resolve()
    )
    splits = get_or_make_splits(train_patients, config)
    logger.info(
        "\n=== Split (Set1 only) ===\n  train=%d val=%d test=%d patients",
        len(splits["train"]), len(splits["val"]), len(splits["test"]),
    )


if __name__ == "__main__":
    main()
