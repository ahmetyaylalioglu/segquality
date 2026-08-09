"""Discover the RAOS dataset from disk — assume nothing, report everything.

Reports the folder structure and file-naming pattern, inspects a few random
label volumes in detail, builds a label inventory across the dataset, flags
missing organs and value anomalies, and finally prints a copy-paste-ready
``organs:`` YAML block. No organ id is hard-coded; every id comes from the data.

    python scripts/inspect_raos.py --config configs/raos.yaml --samples 5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("inspect_raos")

NIFTI_SUFFIX = ".nii.gz"
# A label value above this (or below 0, or non-integer) is treated as suspicious.
_MAX_PLAUSIBLE_LABEL = 200


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def patient_id_from_path(path: Path) -> str:
    """Patient id = filename with the ``.nii.gz`` suffix stripped.

    (Path.stem would only strip ``.gz`` and leave a trailing ``.nii``.)
    """
    name = path.name
    return name[: -len(NIFTI_SUFFIX)] if name.endswith(NIFTI_SUFFIX) else path.stem


def original_split_from_dir(label_dir_name: str) -> str:
    """'labelsTr' -> 'Tr', 'labelsVal' -> 'Val', else the raw name."""
    if label_dir_name.lower().startswith("labels"):
        return label_dir_name[len("labels") :] or label_dir_name
    return label_dir_name


def load_label_volume(path: Path) -> tuple[np.ndarray, tuple[float, ...], tuple[int, ...]]:
    """Load a label volume as int32 plus its voxel spacing and shape.

    Labels are integer; loading as float64 (the nibabel default) wastes time and
    memory over hundreds of volumes, so we cast to int32 immediately. The raw
    array is returned alongside so callers can detect non-integer residue.
    """
    import nibabel as nib

    img = nib.load(str(path))
    raw = np.asanyarray(img.dataobj)
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    return raw, spacing, tuple(int(s) for s in img.shape)


def _detect_value_anomalies(raw: np.ndarray, unique_raw: np.ndarray) -> list[str]:
    """Flag non-integer / negative / implausibly large label values."""
    flags: list[str] = []
    ints = unique_raw.astype(np.int64)
    if np.any(unique_raw != ints):
        flags.append("non-integer label values present (possible float residue)")
    if np.any(unique_raw < 0):
        flags.append("negative label values present")
    if np.any(unique_raw > _MAX_PLAUSIBLE_LABEL):
        big = unique_raw[unique_raw > _MAX_PLAUSIBLE_LABEL].tolist()
        flags.append(f"implausibly large label values: {big}")
    return flags


# --------------------------------------------------------------------------
# (a) structure discovery
# --------------------------------------------------------------------------
def discover_structure(root: Path, subsets: list[str]) -> dict[str, Any]:
    """List the folders actually present in each subset and their file counts."""
    report: dict[str, Any] = {}
    for subset in subsets:
        subset_dir = root / subset
        if not subset_dir.is_dir():
            logger.warning("subset folder not found: %s", subset_dir)
            report[subset] = {"exists": False, "folders": {}}
            continue
        folders = {}
        for child in sorted(p for p in subset_dir.iterdir() if p.is_dir()):
            n_nifti = sum(1 for _ in child.glob(f"*{NIFTI_SUFFIX}"))
            folders[child.name] = n_nifti
        report[subset] = {"exists": True, "folders": folders}
    return report


def collect_label_files(
    root: Path, subsets: list[str], label_dirs: list[str]
) -> list[dict[str, Any]]:
    """Gather every label file across the given subsets and label folders."""
    records: list[dict[str, Any]] = []
    for subset in subsets:
        for label_dir in label_dirs:
            folder = root / subset / label_dir
            if not folder.is_dir():
                continue
            for path in sorted(folder.glob(f"*{NIFTI_SUFFIX}")):
                records.append(
                    {
                        "subset": subset,
                        "original_split": original_split_from_dir(label_dir),
                        "label_dir": label_dir,
                        "path": path,
                        "patient_id": patient_id_from_path(path),
                    }
                )
    return records


def find_image_for_label(root: Path, rec: dict[str, Any]) -> Path | None:
    """Locate the image for a label, allowing the nnU-Net ``_0000`` channel suffix.

    RAOS is inconsistent: Tr/Val images share the label's exact name, while Ts
    images append ``_0000`` (nnU-Net channel 0). We only use labels this phase,
    but reporting this accurately avoids a misleading match count.
    """
    image_dir = rec["label_dir"].replace("labels", "images")
    folder = root / rec["subset"] / image_dir
    pid = rec["patient_id"]
    for name in (f"{pid}{NIFTI_SUFFIX}", f"{pid}_0000{NIFTI_SUFFIX}"):
        candidate = folder / name
        if candidate.is_file():
            return candidate
    return None


def match_images(root: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Report how many labels have a matching image, with examples."""
    matched = 0
    suffixed = 0
    examples: list[dict[str, str]] = []
    for rec in records:
        image = find_image_for_label(root, rec)
        if image is None:
            continue
        matched += 1
        if image.name.endswith(f"_0000{NIFTI_SUFFIX}"):
            suffixed += 1
        if len(examples) < 3:
            examples.append({"label": str(rec["path"]), "image": str(image)})
    return {
        "labels_total": len(records),
        "labels_with_image": matched,
        "images_with_0000_suffix": suffixed,
        "examples": examples,
    }


# --------------------------------------------------------------------------
# (b) detailed per-sample inspection
# --------------------------------------------------------------------------
def inspect_samples(
    root: Path, records: list[dict[str, Any]], n_samples: int, seed: int
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(records), size=min(n_samples, len(records)), replace=False)
    out: list[dict[str, Any]] = []
    for idx in chosen:
        rec = records[int(idx)]
        try:
            raw, spacing, shape = load_label_volume(rec["path"])
        except Exception as exc:  # noqa: BLE001 - report and skip corrupt files
            logger.warning("could not read %s: %s", rec["path"], exc)
            continue
        unique_raw, counts = np.unique(raw, return_counts=True)
        total = int(raw.size)
        image_path = find_image_for_label(root, rec)
        image_shape = None
        if image_path is not None:
            import nibabel as nib

            image_shape = tuple(int(s) for s in nib.load(str(image_path)).shape)
        out.append(
            {
                "patient_id": rec["patient_id"],
                "subset": rec["subset"],
                "shape": shape,
                "spacing_mm": spacing,
                "unique_labels": unique_raw.tolist(),
                "voxels_per_label": {
                    str(v): {"voxels": int(c), "percent": round(100 * c / total, 4)}
                    for v, c in zip(unique_raw, counts)
                },
                "image_shape": image_shape,
                "image_label_shape_match": image_shape == shape if image_shape else None,
                "value_anomalies": _detect_value_anomalies(raw, unique_raw),
            }
        )
    return out


# --------------------------------------------------------------------------
# (c) label inventory across the dataset
# --------------------------------------------------------------------------
def build_inventory(
    records: list[dict[str, Any]], limit: int | None
) -> dict[str, Any]:
    scanned = records if limit is None else records[:limit]
    if limit is None:
        logger.warning(
            "scanning all %d label volumes for the inventory — this may take "
            "several minutes; pass --limit N to scan a subset",
            len(scanned),
        )

    voxels_by_label: dict[int, list[int]] = defaultdict(list)
    patients_by_label: dict[int, set[str]] = defaultdict(set)
    all_patients: set[str] = set()
    anomalies: dict[str, list[str]] = {}
    skipped: list[str] = []

    for i, rec in enumerate(scanned, start=1):
        try:
            raw, _, _ = load_label_volume(rec["path"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("skipping unreadable volume %s: %s", rec["path"], exc)
            skipped.append(str(rec["path"]))
            continue
        pid = rec["patient_id"]
        all_patients.add(pid)
        unique_raw, counts = np.unique(raw, return_counts=True)
        flags = _detect_value_anomalies(raw, unique_raw)
        if flags:
            anomalies[str(rec["path"])] = flags
        for value, count in zip(unique_raw.astype(np.int64), counts):
            if value == 0:  # background
                continue
            voxels_by_label[int(value)].append(int(count))
            patients_by_label[int(value)].add(pid)
        if i % 25 == 0:
            logger.info("  inventory progress: %d/%d volumes", i, len(scanned))

    n_patients = len(all_patients)
    inventory = {}
    for label_id in sorted(voxels_by_label):
        vox = voxels_by_label[label_id]
        inventory[label_id] = {
            "present_in_patients": len(patients_by_label[label_id]),
            "total_patients": n_patients,
            "median_voxels": int(np.median(vox)),
            "min_voxels": int(np.min(vox)),
            "max_voxels": int(np.max(vox)),
        }
    return {
        "n_patients_scanned": n_patients,
        "n_volumes_scanned": len(scanned),
        "labels": inventory,
        "value_anomalies": anomalies,
        "skipped_files": skipped,
    }


def find_missing_organ_labels(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    """Labels present in some but not all patients (expected for Set3)."""
    flagged = []
    for label_id, info in inventory["labels"].items():
        if info["present_in_patients"] < info["total_patients"]:
            flagged.append(
                {
                    "label_id": label_id,
                    "present_in": info["present_in_patients"],
                    "total": info["total_patients"],
                }
            )
    return flagged


# --------------------------------------------------------------------------
# (d) reporting + organs YAML block
# --------------------------------------------------------------------------
def patient_id_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    """One file per patient is expected; flag if that does not hold."""
    ids = [r["patient_id"] for r in records]
    n_unique = len(set(ids))
    n_files = len(ids)
    return {
        "n_files": n_files,
        "n_unique_patients": n_unique,
        "one_to_one": n_files == n_unique,
    }


def suggest_target(inventory: dict[str, Any]) -> int | None:
    """Pick the largest organ present in every patient (most consistent)."""
    labels = inventory["labels"]
    if not labels:
        return None
    full = {
        lid: info
        for lid, info in labels.items()
        if info["present_in_patients"] == info["total_patients"]
    }
    pool = full or labels
    return max(pool, key=lambda lid: pool[lid]["median_voxels"])


def build_organs_yaml_block(inventory: dict[str, Any]) -> str:
    labels = inventory["labels"]
    lines = ["organs:"]
    for label_id in sorted(labels):
        info = labels[label_id]
        lines.append(
            f"  organ_{label_id:02d}: {label_id}"
            f"    # median {info['median_voxels']} voxels, "
            f"{info['present_in_patients']}/{info['total_patients']} patients"
        )
    target = suggest_target(inventory)
    if target is not None:
        lines.append(
            f"target: organ_{target:02d}  "
            f"# suggested: largest organ present in all patients"
        )
    return "\n".join(lines)


def log_structure(structure: dict[str, Any]) -> None:
    logger.info("\n=== Folder structure (depth 2, file counts) ===")
    for subset, info in structure.items():
        if not info["exists"]:
            logger.info("  %s: MISSING", subset)
            continue
        logger.info("  %s/", subset)
        for folder, count in info["folders"].items():
            logger.info("    %-12s %6d files", folder, count)


def log_inventory(inventory: dict[str, Any]) -> None:
    logger.info(
        "\n=== Label inventory (%d volumes, %d patients) ===",
        inventory["n_volumes_scanned"],
        inventory["n_patients_scanned"],
    )
    logger.info("  %-8s %-16s %-14s %-12s", "label", "patients", "median_vox", "range")
    for label_id, info in inventory["labels"].items():
        logger.info(
            "  %-8d %-16s %-14d %d..%d",
            label_id,
            f"{info['present_in_patients']}/{info['total_patients']}",
            info["median_voxels"],
            info["min_voxels"],
            info["max_voxels"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to raos.yaml")
    parser.add_argument("--limit", type=int, default=None, help="max volumes for inventory")
    parser.add_argument("--samples", type=int, default=5, help="detailed random volumes")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = config["dataset"]
    root = Path(dataset["root"]).expanduser()
    subsets = list(dataset["train_subsets"]) + list(dataset["heldout_subsets"])
    label_dirs = list(dataset["label_dirs"])
    out_dir = PROJECT_ROOT / config.get("output_dir", "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Health check: fail fast before scanning hundreds of volumes ------
    logger.info("=== Health check ===")
    logger.info("  resolved dataset.root : %s", root.resolve())
    logger.info("  root exists           : %s", root.exists())
    if not root.is_dir():
        logger.error("dataset root does not exist or is not a directory — aborting")
        raise SystemExit(1)

    structure = discover_structure(root, subsets)
    log_structure(structure)
    total_nifti = sum(
        sum(info["folders"].values()) for info in structure.values() if info["exists"]
    )
    if total_nifti == 0:
        logger.error(
            "found 0 .nii.gz files across subsets %s — check dataset.root / "
            "label_dirs; not scanning empty folders",
            subsets,
        )
        raise SystemExit(1)

    records = collect_label_files(root, subsets, label_dirs)
    if not records:
        logger.error(
            "no label files found in label_dirs %s — the folder-naming assumption "
            "may be wrong; check the structure listing above",
            label_dirs,
        )
        raise SystemExit(1)
    logger.info("  first label file      : %s", records[0]["path"].name)
    logger.info(
        "  -> patient id          : %s (verify this id-extraction looks right)",
        records[0]["patient_id"],
    )

    naming = match_images(root, records)
    logger.info(
        "\n=== Naming / image-label matching ===\n"
        "  %d label files, %d have a matching image (%d via nnU-Net _0000 suffix)",
        naming["labels_total"],
        naming["labels_with_image"],
        naming["images_with_0000_suffix"],
    )
    for ex in naming["examples"]:
        logger.info("  e.g. label=%s\n       image=%s", ex["label"], ex["image"])

    pid_report = patient_id_report(records)
    logger.info(
        "\n=== Patient-id extraction ===\n"
        "  %d files -> %d unique patient ids (one-to-one: %s)",
        pid_report["n_files"],
        pid_report["n_unique_patients"],
        pid_report["one_to_one"],
    )
    if not pid_report["one_to_one"]:
        logger.warning(
            "  file count != patient count: the id-extraction pattern needs review "
            "(a patient may have multiple scans, or the pattern is wrong)"
        )

    samples = inspect_samples(root, records, args.samples, seed=int(config["seed"]))
    logger.info("\n=== Sample volumes (%d) ===", len(samples))
    for s in samples:
        logger.info(
            "  %s [%s] shape=%s spacing=%s labels=%s img_match=%s%s",
            s["patient_id"],
            s["subset"],
            s["shape"],
            tuple(round(v, 3) for v in s["spacing_mm"]),
            s["unique_labels"],
            s["image_label_shape_match"],
            f" ANOMALIES={s['value_anomalies']}" if s["value_anomalies"] else "",
        )

    inventory = build_inventory(records, args.limit)
    log_inventory(inventory)

    missing = find_missing_organ_labels(inventory)
    if missing:
        logger.info("\n=== Missing-organ labels (present in some, not all) ===")
        for m in missing:
            logger.info("  label %d: %d/%d patients", m["label_id"], m["present_in"], m["total"])
    if inventory["value_anomalies"]:
        logger.warning(
            "\n=== VALUE ANOMALIES in %d volume(s) ===", len(inventory["value_anomalies"])
        )
        for path, flags in inventory["value_anomalies"].items():
            logger.warning("  %s: %s", path, flags)
    if inventory["skipped_files"]:
        logger.warning(
            "\nskipped %d unreadable volume(s)", len(inventory["skipped_files"])
        )

    organs_block = build_organs_yaml_block(inventory)
    logger.info(
        "\n=== Copy this into configs/raos.yaml "
        "(fill real organ names from RAOS docs) ===\n%s",
        organs_block,
    )

    inspection_path = out_dir / "raos_inspection.json"
    payload = {
        "root": str(root),
        "structure": structure,
        "naming": naming,
        "patient_id_report": pid_report,
        "samples": samples,
        "inventory": inventory,
        "missing_organ_labels": missing,
        "organs_yaml_block": organs_block,
    }
    with inspection_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    logger.info("\nwrote %s", inspection_path.resolve())


if __name__ == "__main__":
    main()
