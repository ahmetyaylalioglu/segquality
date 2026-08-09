"""Measure the degradation Dice-targeting quality on REAL liver slices, so the
"A fix" (chaining + reachable-type selection) can be judged apples-to-apples.

Everything is driven by a single ``--master-seed`` so the *input* plan (which
mask, which target Dice, which per-sample seed) is byte-identical between the
``before`` run (unmodified degrader) and the ``after`` run (fixed degrader). The
only thing that changes between the two runs is the degrader's behaviour.

Workflow (see Job 2 plan):
    # 1. BEFORE touching data/degradations.py:
    python scripts/verify_degradation_fix.py --config configs/raos.yaml --tag before
    # 2. apply the fix, then:
    python scripts/verify_degradation_fix.py --config configs/raos.yaml --tag after

Outputs
-------
* outputs/dice_metrics_{tag}.json    metrics + per-sample achieved/target arrays
* outputs/dice_distribution_fixed.png (after only) before-vs-after histogram
* outputs/chain_degradation_check.png (after only) low-Dice chained examples

The masks come from ``outputs/index_liver_train.json`` (real liver, resized to
``slicing.output_size``) so the reachable Dice ranges match the training data's
geometry — synthetic ellipses would report different floors.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import label as cc_label

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import load_config  # noqa: E402
from data.degradations import SyntheticDegrader, dice, make_degraded_pair  # noqa: E402
from data.transforms import extract_organ_patch  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("verify_degradation_fix")

NIFTI_SUFFIX = ".nii.gz"
N_MASKS = 40          # real liver slices sampled from the index
N_SAMPLES = 1000      # degradation attempts making up the 1000-sample plan
TARGET_BANDS = [(0.40, 0.55), (0.55, 0.70), (0.70, 0.85), (0.85, 0.99)]
N_CHAIN_EXAMPLES = 6  # low-Dice chained masks drawn for the visual check


# --------------------------------------------------------------------------
# Real liver mask loading (reconstructs paths from the index records)
# --------------------------------------------------------------------------
def label_path_for(root: Path, record: dict[str, Any]) -> Path:
    label_dir = "labels" + record["original_split"]
    return root / record["source_subset"] / label_dir / (record["patient_id"] + NIFTI_SUFFIX)


def load_liver_masks(
    config: dict[str, Any], index_path: Path, n_masks: int, seed: int
) -> list[np.ndarray]:
    """Sample ``n_masks`` real liver slices, resized to slicing.output_size.

    Slices are chosen deterministically from the index; volumes are loaded once
    per patient (grouped) to avoid reopening the same NIfTI.
    """
    import nibabel as nib

    with index_path.open(encoding="utf-8") as fh:
        index = json.load(fh)
    slices = index["slices"]
    organ_id = int(index["metadata"]["organ_id"])
    axis = int(index["metadata"]["axis"])

    root = Path(config["dataset"]["root"]).expanduser()
    output_size = tuple(config["slicing"]["output_size"])
    bbox_margin = float(config["slicing"]["bbox_margin"])

    rng = np.random.default_rng(seed)
    chosen_idx = rng.choice(len(slices), size=n_masks, replace=False)
    chosen = [slices[int(i)] for i in chosen_idx]

    by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in chosen:
        by_patient[rec["patient_id"]].append(rec)

    masks: list[np.ndarray] = []
    for patient_id, recs in by_patient.items():
        path = label_path_for(root, recs[0])
        volume = np.asanyarray(nib.load(str(path)).dataobj)
        for rec in recs:
            slice_2d = np.take(volume, rec["slice_index"], axis=axis) == organ_id
            masks.append(extract_organ_patch(slice_2d, output_size, bbox_margin))
    logger.info("loaded %d real liver masks from %d patients", len(masks), len(by_patient))
    return masks


# --------------------------------------------------------------------------
# Deterministic sampling plan (identical before/after for a fixed master seed)
# --------------------------------------------------------------------------
def build_plan(
    n_masks: int, n_samples: int, target_range: tuple[float, float], master_seed: int
) -> list[dict[str, Any]]:
    """(mask_idx, target_dice, sample_seed) triples — independent of the degrader."""
    rng = np.random.default_rng(master_seed)
    lo, hi = target_range
    plan = []
    for _ in range(n_samples):
        plan.append(
            {
                "mask_idx": int(rng.integers(n_masks)),
                "target_dice": float(rng.uniform(lo, hi)),
                "sample_seed": int(rng.integers(0, 2**31 - 1)),
            }
        )
    return plan


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def fragmentation_stats(masks: list[np.ndarray]) -> dict[str, Any]:
    """Connected-component fragmentation over a list of degraded masks.

    A real liver is one component; controlled errors (a removed chunk) stay a
    few large pieces, whereas boundary-noise speckle shatters into many tiny
    ones. We separate these with two numbers: the fraction of masks with >3
    components (fragmented at all) and, among those, the mean area fraction held
    by the LARGEST component (near 1.0 = one-big-plus-crumbs = realistic; low =
    scattered pixel noise).
    """
    n_components: list[int] = []
    largest_frac_when_frag: list[float] = []
    for m in masks:
        binary = np.asarray(m) > 0
        labeled, n = cc_label(binary)
        n_components.append(int(n))
        if n > 3:
            sizes = np.bincount(labeled.ravel())[1:]  # drop background label 0
            largest_frac_when_frag.append(float(sizes.max() / sizes.sum()))
    arr = np.asarray(n_components)
    return {
        "frac_gt3_components": float((arr > 3).mean()),
        "mean_components": float(arr.mean()),
        "max_components": int(arr.max()),
        "mean_largest_frac_when_frag": (
            float(np.mean(largest_frac_when_frag)) if largest_frac_when_frag else 1.0
        ),
        "n_fragmented": int((arr > 3).sum()),
    }


def band_of(target: float) -> str:
    for lo, hi in TARGET_BANDS:
        if lo <= target < hi or (hi == TARGET_BANDS[-1][1] and target == hi):
            return f"{lo:.2f}-{hi:.2f}"
    return f"{TARGET_BANDS[-1][0]:.2f}-{TARGET_BANDS[-1][1]:.2f}"


def compute_metrics(
    targets: list[float], achieved: list[float], tol: float
) -> dict[str, Any]:
    t = np.asarray(targets)
    a = np.asarray(achieved)
    diff = np.abs(a - t)
    reached = diff <= tol

    bands: dict[str, Any] = {}
    for lo, hi in TARGET_BANDS:
        key = f"{lo:.2f}-{hi:.2f}"
        mask = np.array([band_of(x) == key for x in targets])
        n = int(mask.sum())
        bands[key] = {
            "count": n,
            "target_reached_frac": float(reached[mask].mean()) if n else 0.0,
            "mean_abs_diff": float(diff[mask].mean()) if n else 0.0,
        }
    return {
        "outside_tol_frac": float((~reached).mean()),
        "mean_abs_diff": float(diff.mean()),
        "median_achieved": float(np.median(a)),
        "target_reached_frac": float(reached.mean()),
        "bands": bands,
    }


# --------------------------------------------------------------------------
# Run the plan through the degrader
# --------------------------------------------------------------------------
def run_before(
    masks: list[np.ndarray], plan: list[dict[str, Any]]
) -> tuple[list[float], list[float]]:
    """Unmodified degrader: only (mask, target, seed) -> achieved Dice."""
    targets, achieved = [], []
    for item in plan:
        mask = masks[item["mask_idx"]]
        rng = np.random.default_rng(item["sample_seed"])
        _, ach = make_degraded_pair(mask, rng, target_dice=item["target_dice"])
        targets.append(item["target_dice"])
        achieved.append(ach)
    return targets, achieved


def run_after(
    config: dict[str, Any], masks: list[np.ndarray], plan: list[dict[str, Any]]
) -> tuple[list[float], list[float], list[int], list[dict[str, Any]], dict[str, tuple[float, float]]]:
    """Fixed degrader via sample_record so chain depth is observable.

    The degrader is calibrated on the SAME real-liver masks it runs on, so
    target-reachable type selection uses ranges consistent with the eval data.
    Returns (targets, achieved, chain_depths, low_dice_chain_examples, type_ranges).
    """
    degrader = SyntheticDegrader.from_config(config)
    ranges = degrader.calibrate(masks)
    targets, achieved, depths = [], [], []
    degraded_masks: list[np.ndarray] = []
    examples: list[dict[str, Any]] = []
    for item in plan:
        mask = masks[item["mask_idx"]]
        rng = np.random.default_rng(item["sample_seed"])
        record = degrader.sample_record(
            mask, rng, patient_id="verify", slice_index=item["mask_idx"],
            target_dice=item["target_dice"],
        )
        degraded = degrader.apply_record(mask, record)
        depth = len(record.steps) if record.steps else 1
        targets.append(item["target_dice"])
        achieved.append(record.true_dice)
        depths.append(depth)
        degraded_masks.append(np.asarray(degraded) > 0)
        if depth > 1 and 0.40 <= record.true_dice <= 0.50 and len(examples) < N_CHAIN_EXAMPLES:
            examples.append(
                {
                    "mask": mask,
                    "degraded": degraded,
                    "true_dice": record.true_dice,
                    "types": "+".join(s["type"] for s in record.steps),
                }
            )
    return targets, achieved, depths, examples, ranges, degraded_masks


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def log_metrics(tag: str, metrics: dict[str, Any]) -> None:
    logger.info("\n=== %s metrics (real liver, %d samples) ===", tag, N_SAMPLES)
    logger.info("  outside +/-tol frac : %.4f", metrics["outside_tol_frac"])
    logger.info("  mean |target-ach|   : %.4f", metrics["mean_abs_diff"])
    logger.info("  median achieved     : %.4f", metrics["median_achieved"])
    logger.info("  target_reached frac : %.4f", metrics["target_reached_frac"])
    logger.info("  per target band     : reached_frac (n, mean|diff|)")
    for band, b in metrics["bands"].items():
        logger.info(
            "    %-11s : %.4f  (n=%d, |diff|=%.4f)",
            band, b["target_reached_frac"], b["count"], b["mean_abs_diff"],
        )


def log_type_ranges(ranges: dict[str, tuple[float, float]]) -> None:
    logger.info("\n=== per-type reachable Dice range (real liver) ===")
    for name, (lo, hi) in ranges.items():
        logger.info("    %-16s [%.3f, %.3f]", name, lo, hi)


def log_fragmentation(frag: dict[str, Any]) -> None:
    logger.info("\n=== fragmentation (connected components on degraded masks) ===")
    logger.info("  frac >3 components  : %.4f  (n=%d)",
                frag["frac_gt3_components"], frag["n_fragmented"])
    logger.info("  mean components     : %.3f  (max=%d)",
                frag["mean_components"], frag["max_components"])
    logger.info("  largest-piece frac when fragmented : %.3f",
                frag["mean_largest_frac_when_frag"])


def log_chain_stats(depths: list[int]) -> None:
    arr = np.asarray(depths)
    logger.info("\n=== chaining ===")
    logger.info("  chained (depth>1)   : %.4f", float((arr > 1).mean()))
    hist = {int(d): int((arr == d).sum()) for d in sorted(set(depths))}
    logger.info("  depth histogram     : %s", hist)


def _ensure_matplotlib() -> Any:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        logger.info("matplotlib missing -> installing it (only)")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_distribution(before: list[float], after: list[float], out_path: Path) -> None:
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0.3, 1.0, 36)
    ax.hist(before, bins=bins, alpha=0.55, label="before", color="tab:red")
    ax.hist(after, bins=bins, alpha=0.55, label="after", color="tab:green")
    ax.set_xlabel("achieved true Dice")
    ax.set_ylabel("count")
    ax.set_title("Achieved Dice distribution (real liver, same 1000-sample plan)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", out_path.resolve())


def plot_chain_examples(examples: list[dict[str, Any]], out_path: Path) -> None:
    plt = _ensure_matplotlib()
    if not examples:
        logger.warning("no chained low-Dice examples found -> skipping %s", out_path.name)
        return
    n = len(examples)
    fig, axes = plt.subplots(n, 2, figsize=(5, 2.4 * n))
    if n == 1:
        axes = axes[None, :]
    for i, ex in enumerate(examples):
        axes[i, 0].imshow(ex["mask"], cmap="gray")
        axes[i, 0].set_title("original liver")
        axes[i, 1].imshow(ex["degraded"], cmap="gray")
        axes[i, 1].set_title(f"chained ({ex['types']})\nDice={ex['true_dice']:.3f}")
        for j in range(2):
            axes[i, j].axis("off")
    fig.suptitle("Chained low-Dice masks — still a broken liver, not noise?")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", out_path.resolve())


# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tag", required=True, choices=["before", "after"])
    parser.add_argument("--master-seed", type=int, default=1234)
    parser.add_argument("--n-masks", type=int, default=N_MASKS)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument(
        "--no-boundary-fix", action="store_true",
        help="lift the boundary_noise light-only restriction (measure the OFF state)",
    )
    args = parser.parse_args()

    if args.no_boundary_fix:
        import data.degradations as _deg
        _deg._LIGHT_ONLY_TYPES = ()  # measure pre-fix fragmentation from fixed code
        logger.info(">>> boundary_noise restriction DISABLED (OFF state)")

    config = load_config(args.config)
    tol = float(config["degradation"]["dice_tolerance"])
    target_range = tuple(config["degradation"]["target_dice_range"])
    index_path = PROJECT_ROOT / config["cache"]["index_path"].format(organ=config["target"])
    index_path = index_path.with_name(f"{index_path.stem}_train{index_path.suffix}")
    out_dir = PROJECT_ROOT / config["output_dir"]

    logger.info(
        "=== verify degradation fix: tag=%s master_seed=%d masks=%d samples=%d tol=%.3f ===",
        args.tag, args.master_seed, args.n_masks, args.n_samples, tol,
    )
    masks = load_liver_masks(config, index_path, args.n_masks, seed=args.master_seed)
    plan = build_plan(args.n_masks, args.n_samples, target_range, args.master_seed)

    if args.tag == "before":
        targets, achieved = run_before(masks, plan)
        metrics = compute_metrics(targets, achieved, tol)
        log_metrics("before", metrics)
        payload = {
            "tag": "before", "master_seed": args.master_seed,
            "n_masks": args.n_masks, "n_samples": args.n_samples, "tolerance": tol,
            **metrics, "targets": targets, "achieved": achieved,
        }
    else:
        targets, achieved, depths, examples, ranges, degraded_masks = run_after(
            config, masks, plan
        )
        metrics = compute_metrics(targets, achieved, tol)
        frag = fragmentation_stats(degraded_masks)
        log_metrics("after", metrics)
        log_type_ranges(ranges)
        log_chain_stats(depths)
        log_fragmentation(frag)
        chain_frac = float((np.asarray(depths) > 1).mean())
        suffix = "_nofix" if args.no_boundary_fix else ""
        payload = {
            "tag": "after", "boundary_fix": not args.no_boundary_fix,
            "master_seed": args.master_seed,
            "n_masks": args.n_masks, "n_samples": args.n_samples, "tolerance": tol,
            **metrics, "chain_frac": chain_frac, "fragmentation": frag,
            "type_ranges": {k: list(v) for k, v in ranges.items()},
            "targets": targets, "achieved": achieved,
        }
        plot_chain_examples(examples, out_dir / f"chain_degradation_check{suffix}.png")
        before_path = out_dir / "dice_metrics_before.json"
        if not args.no_boundary_fix and before_path.exists():
            before = json.loads(before_path.read_text(encoding="utf-8"))["achieved"]
            plot_distribution(before, achieved, out_dir / "dice_distribution_fixed.png")
        elif not args.no_boundary_fix:
            logger.warning("dice_metrics_before.json missing -> skipping overlay plot")

        out_path = out_dir / f"dice_metrics_after{suffix}.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("\nwrote %s", out_path.resolve())
        return

    out_path = out_dir / f"dice_metrics_{args.tag}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("\nwrote %s", out_path.resolve())


if __name__ == "__main__":
    main()
