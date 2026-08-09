"""Standalone visual demo of the mask degradations — no RAOS data required.

Generates organ-like synthetic shapes (ellipse, two disjoint components, an
indented irregular shape), degrades each toward a set of target Dice values, and
writes two figures:

* ``outputs/demo_degradations.png``      — shapes x (original + degraded) grid
* ``outputs/demo_dice_distribution.png`` — Dice histogram of 1000 random pairs

Paths are resolved against the project root, so it writes to the same place no
matter the current working directory.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import load_config  # noqa: E402
from data.degradations import SyntheticDegrader, make_degraded_pair  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("demo_degradations")

CANVAS = (128, 128)
TARGET_DICES = [0.95, 0.85, 0.70, 0.55]
N_HISTOGRAM = 1000


def _ensure_matplotlib():
    """Import matplotlib, installing *only* matplotlib if it is missing."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        logger.info("matplotlib not found — installing it (only matplotlib)")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


# --------------------------------------------------------------------------
# Synthetic organ-like shapes
# --------------------------------------------------------------------------
def _ellipse(shape, cy, cx, ry, rx) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0


def _disk(shape, cy, cx, r) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2


def make_shapes() -> dict[str, np.ndarray]:
    """Three organ-like binary masks on the shared canvas."""
    ellipse = _ellipse(CANVAS, 64, 64, 42, 28)

    two_components = _disk(CANVAS, 42, 42, 16) | _disk(CANVAS, 88, 88, 16)

    # Indented "bean": an ellipse with a concave bite taken out of one side.
    bean = _ellipse(CANVAS, 64, 58, 44, 32) & ~_disk(CANVAS, 64, 94, 24)

    return {
        "ellipse": ellipse,
        "two_components": two_components,
        "irregular_bean": bean,
    }


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def render_grid(shapes, degrader, out_path, plt) -> None:
    n_rows = len(shapes)
    n_cols = 1 + len(TARGET_DICES)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.4), squeeze=False
    )
    rng = np.random.default_rng(0)

    for row, (name, mask) in enumerate(shapes.items()):
        axes[row][0].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[row][0].set_title(f"{name}\noriginal (GT)", fontsize=9)
        axes[row][0].set_ylabel(name, fontsize=9)
        for col, target in enumerate(TARGET_DICES, start=1):
            record = degrader.sample_record(
                mask, rng, patient_id=name, slice_index=0, target_dice=target
            )
            degraded = degrader.apply_record(mask, record)
            axes[row][col].imshow(degraded, cmap="gray", vmin=0, vmax=1)
            axes[row][col].set_title(
                f"{record.degradation_type}\nDice={record.true_dice:.2f} "
                f"(t={target:.2f})",
                fontsize=9,
            )

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Synthetic mask degradations (true Dice under each)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def render_histogram(shape, out_path, plt) -> None:
    rng = np.random.default_rng(0)
    dices = np.array([make_degraded_pair(shape, rng)[1] for _ in range(N_HISTOGRAM)])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(dices, bins=np.linspace(0.3, 1.0, 30), color="#4C72B0", edgecolor="white")
    ax.axvline(
        np.median(dices),
        color="crimson",
        linestyle="--",
        label=f"median = {np.median(dices):.2f}",
    )
    ax.set_xlabel("true Dice")
    ax.set_ylabel("count")
    ax.set_title(f"Dice distribution of {N_HISTOGRAM} random degradations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "raos.yaml"),
        help="config used for degradation params and output_dir",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    degrader = SyntheticDegrader.from_config(config)

    out_dir = PROJECT_ROOT / config.get("output_dir", "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    plt = _ensure_matplotlib()
    shapes = make_shapes()

    grid_path = out_dir / "demo_degradations.png"
    hist_path = out_dir / "demo_dice_distribution.png"
    render_grid(shapes, degrader, grid_path, plt)
    render_histogram(shapes["irregular_bean"], hist_path, plt)

    logger.info("wrote %s", grid_path.resolve())
    logger.info("wrote %s", hist_path.resolve())
    print("\nGenerated files:")
    print(f"  {grid_path.resolve()}")
    print(f"  {hist_path.resolve()}")


if __name__ == "__main__":
    main()
