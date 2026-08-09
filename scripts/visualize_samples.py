"""Advisor-facing figure: real liver masks + synthetic degradations.

Produces two PNGs from the SAME rows, supporting two different sentences:

* ``samples_liver_train.png``           — plain degraded masks. "This is exactly
  what the VAE will see: only the (possibly wrong) mask, no ground truth."
* ``samples_liver_train_annotated.png`` — same masks with a thin GT contour on
  top. "This is what the degradation changed: erosion ate here, a false positive
  was added there." (compare fill against the GT boundary.)

Rows are sampled to be *representative* of liver geometry (small tip slices,
typical mid slices, large body slices) rather than random, so the figure shows
the degrader working across shapes, not one monotonous disk. Columns are fixed
target Dice levels [GT, 0.90, 0.70, 0.50], but each cell's title reports the
*achieved* Dice and the degradation type actually chosen (honest, per the eval
design). Degradation types are left to the sampler — a row may show erosion,
another a false positive; that natural variety is the point.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import load_config  # noqa: E402
from data.degradations import SyntheticDegrader  # noqa: E402
from data.raos_dataset import RAOSMaskSlices  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("visualize_samples")

TARGET_DICE = (0.90, 0.70, 0.50)   # degradation columns (GT is a 4th, leftmost)
ROWS_PER_TERTILE = 2               # small / medium / large -> 6 rows total
CONTOUR_COLOR = "#ff2d2d"          # thin GT boundary overlay (annotated figure)


def _ensure_plt() -> Any:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        logger.info("matplotlib missing -> installing it (only)")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def pick_representative_rows(
    dataset: RAOSMaskSlices, per_tertile: int, seed: int
) -> list[int]:
    """Sample ``per_tertile`` slice indices from each organ-area tertile.

    Ordering the returned rows small -> medium -> large makes the size range
    visible; sampling within a tertile is deterministic given ``seed``.
    """
    areas = np.array([int(s["organ_area"]) for s in dataset.samples])
    order = np.argsort(areas)
    thirds = np.array_split(order, 3)  # low, mid, high area groups
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for group in thirds:
        take = min(per_tertile, len(group))
        picks = rng.choice(group, size=take, replace=False)
        chosen.extend(int(i) for i in sorted(picks, key=lambda i: areas[i]))
    return chosen


def build_rows(
    dataset: RAOSMaskSlices, row_indices: list[int], degrader: SyntheticDegrader, seed: int
) -> list[dict[str, Any]]:
    """For each row: the GT mask and one degraded (mask, type, dice) per target."""
    rows: list[dict[str, Any]] = []
    for r, idx in enumerate(row_indices):
        sample = dataset[idx]
        gt = np.asarray(sample["mask"]) > 0
        cells = []
        for c, target in enumerate(TARGET_DICE):
            rng = np.random.default_rng(seed + 1000 * r + c)
            record = degrader.sample_record(
                gt, rng, patient_id=sample["patient_id"],
                slice_index=sample["slice_index"], target_dice=target,
            )
            degraded = np.asarray(degrader.apply_record(gt, record)) > 0
            cells.append({
                "mask": degraded,
                "type": record.degradation_type,
                "dice": record.true_dice,
            })
        rows.append({
            "gt": gt,
            "patient_id": sample["patient_id"],
            "slice_index": sample["slice_index"],
            "organ_area": sample["organ_area"],
            "cells": cells,
        })
    return rows


def _row_label(row: dict[str, Any]) -> str:
    pid = row["patient_id"]
    short = f"...{pid[-8:]}" if len(pid) > 8 else pid
    return f"{short}\nslice {row['slice_index']}\narea {row['organ_area']} px"


def render(rows: list[dict[str, Any]], out_path: Path, annotated: bool) -> None:
    plt = _ensure_plt()
    n_rows = len(rows)
    n_cols = len(TARGET_DICE) + 1  # GT + degradation columns
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.6 * n_cols, 2.9 * n_rows), squeeze=False
    )
    subtitle = "with GT contour (what the degradation changed)" if annotated \
        else "plain masks (what the VAE sees)"
    fig.suptitle(
        f"RAOS liver — synthetic degradations for evaluation\n{subtitle}",
        fontsize=14, fontweight="bold",
    )

    for r, row in enumerate(rows):
        # Column 0: ground truth.
        ax = axes[r][0]
        ax.imshow(row["gt"], cmap="gray", interpolation="nearest")
        ax.set_title("GT (ground truth)", fontsize=9)
        ax.set_ylabel(_row_label(row), fontsize=8, rotation=0,
                      ha="right", va="center", labelpad=38)
        ax.set_xticks([]); ax.set_yticks([])

        # Degradation columns.
        for c, cell in enumerate(row["cells"], start=1):
            ax = axes[r][c]
            ax.imshow(cell["mask"], cmap="gray", interpolation="nearest")
            if annotated:
                ax.contour(row["gt"].astype(float), levels=[0.5],
                           colors=CONTOUR_COLOR, linewidths=1.2)
            ax.set_title(f"{cell['type']}\nDice = {cell['dice']:.2f}", fontsize=9)
            ax.axis("off")

    fig.tight_layout(rect=(0.04, 0.0, 1.0, 0.96))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    logger.info("wrote %s", out_path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = PROJECT_ROOT / config["output_dir"]
    logger.info("=== visualize_samples: organ=%s ===", config["target"])

    dataset = RAOSMaskSlices.from_config(config, split="train", group="train")
    row_indices = pick_representative_rows(dataset, ROWS_PER_TERTILE, args.seed)
    gt_masks = [np.asarray(dataset[i]["mask"]) > 0 for i in row_indices]

    # Calibrate on the real row masks so type selection uses geometry-consistent
    # Dice ranges (keeps boundary_noise light -> no scattered-noise panels).
    degrader = SyntheticDegrader.from_config(config)
    degrader.calibrate(gt_masks)

    rows = build_rows(dataset, row_indices, degrader, args.seed)
    logger.info("selected %d representative rows (areas: %s)", len(rows),
                [r["organ_area"] for r in rows])

    render(rows, out_dir / "samples_liver_train.png", annotated=False)
    render(rows, out_dir / "samples_liver_train_annotated.png", annotated=True)


if __name__ == "__main__":
    main()
