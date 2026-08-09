"""Per-split dataset statistics + sanity checks for the liver mask slices.

Reads only the precomputed index + patient split (no volume loading), so it is
fast. Beyond counts it surfaces two balance checks that matter for a fair
evaluation later:

* slices-per-patient ratio per split — if val/test differ a lot from train, the
  patient shuffle happened to pile big-liver (many-slice) patients into one
  split;
* organ-area distribution per split — if one split holds systematically
  larger/smaller liver cross-sections, that is a distribution shift that would
  bias the VAE's test-time score.

Both are reported as numbers (and flagged if beyond a tolerance); the decision to
re-split is left to a human.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import load_config  # noqa: E402
from data.raos_dataset import RAOSMaskSlices  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("dataset_stats")

SPLIT_NAMES = ("train", "val", "test")
# A split's slices/patient or median area may drift this fraction from train's
# before we flag it as a possibly-unbalanced shuffle worth a human look.
RATIO_WARN_FRAC = 0.20
AREA_WARN_FRAC = 0.20


def _summary(values: list[int]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def split_stats(samples: list[dict[str, Any]], patients: list[str]) -> dict[str, Any]:
    n_slices = len(samples)
    n_patients = len(patients)
    per_patient = Counter(s["patient_id"] for s in samples)
    areas = [int(s["organ_area"]) for s in samples]
    return {
        "n_patients": n_patients,
        "n_slices": n_slices,
        "slices_per_patient": round(n_slices / n_patients, 2) if n_patients else 0.0,
        "slices_per_patient_dist": _summary(list(per_patient.values())),
        "organ_area": _summary(areas),
    }


def assert_no_leakage(patient_sets: dict[str, set[str]]) -> None:
    names = list(patient_sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap = patient_sets[a] & patient_sets[b]
            assert not overlap, f"patient leakage between {a} and {b}: {sorted(overlap)[:3]}"
    logger.info("leakage check: OK (train/val/test patient sets are disjoint)")


def balance_flags(stats: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """Flag splits whose ratio/median-area drift from train beyond tolerance."""
    ratio_flags: list[str] = []
    area_flags: list[str] = []
    train = stats["train"]
    base_ratio = train["slices_per_patient"]
    base_area = train["organ_area"]["median"]
    for name in ("val", "test"):
        r = stats[name]["slices_per_patient"]
        if base_ratio and abs(r - base_ratio) / base_ratio > RATIO_WARN_FRAC:
            ratio_flags.append(
                f"{name} slices/patient={r} vs train={base_ratio} "
                f"(>{RATIO_WARN_FRAC:.0%} drift)"
            )
        a = stats[name]["organ_area"]["median"]
        if base_area and abs(a - base_area) / base_area > AREA_WARN_FRAC:
            area_flags.append(
                f"{name} median area={a:.0f} vs train={base_area:.0f} "
                f"(>{AREA_WARN_FRAC:.0%} drift)"
            )
    return {"ratio_flags": ratio_flags, "area_flags": area_flags}


def log_table(stats: dict[str, dict[str, Any]]) -> None:
    logger.info("\n=== per-split statistics (liver) ===")
    logger.info(
        "  %-6s %9s %9s %14s | %-24s | %-24s",
        "split", "patients", "slices", "slices/patient",
        "slices/patient (min/med/max)", "organ area (min/med/max)",
    )
    for name in SPLIT_NAMES:
        s = stats[name]
        spp = s["slices_per_patient_dist"]
        area = s["organ_area"]
        logger.info(
            "  %-6s %9d %9d %14.2f | %6.0f /%6.0f /%6.0f | %6.0f /%6.0f /%6.0f",
            name, s["n_patients"], s["n_slices"], s["slices_per_patient"],
            spp["min"], spp["median"], spp["max"],
            area["min"], area["median"], area["max"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--group", default="train", help="index group to read")
    args = parser.parse_args()

    config = load_config(args.config)
    organ = config["target"]
    logger.info("=== dataset_stats: organ=%s group=%s ===", organ, args.group)

    stats: dict[str, dict[str, Any]] = {}
    patient_sets: dict[str, set[str]] = {}
    for name in SPLIT_NAMES:
        ds = RAOSMaskSlices.from_config(config, split=name, group=args.group)
        stats[name] = split_stats(ds.samples, ds.patients)
        patient_sets[name] = set(ds.patients)

    assert_no_leakage(patient_sets)
    log_table(stats)

    flags = balance_flags(stats)
    logger.info("\n=== balance checks ===")
    if not flags["ratio_flags"] and not flags["area_flags"]:
        logger.info("  slices/patient and organ-area distributions look balanced.")
    for msg in flags["ratio_flags"] + flags["area_flags"]:
        logger.info("  [FLAG] %s", msg)

    payload = {
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "organ": organ,
        "group": args.group,
        "splits": stats,
        "leakage_ok": True,
        "balance_flags": flags,
    }
    out_path = PROJECT_ROOT / config["output_dir"] / "stats.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("\nwrote %s", out_path.resolve())


if __name__ == "__main__":
    main()
