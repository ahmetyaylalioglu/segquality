"""Phase 8: does the 3D shape-VAE's reconstruction error track a volume's true Dice?

The volumetric mirror of ``scripts/evaluate_correlation.py``. Takes a trained
ConvVAE3D checkpoint and, on the held-out test split (patient-based, never seen in
training), builds an evaluation set of clean liver *volumes* plus controlled 3D
degradations of known true Dice, scores every volume with each quality score in
``quality.score`` (which is already dimension-agnostic), and reports how well each
score correlates with true Dice — globally and stratified by Dice band.

Two things are load-bearing and differ from the 2D script:

* **3D primitive set.** The degrader is built with ``primitives=PRIMITIVES_3D``.
  The default ``primitives=None`` would fall back to the 2D primitives, which
  would try to carve a 2D disk out of a 3D volume — a silent shape error. The run
  logs the active primitive set up front so this can be verified.
* **Squashed-space consistency.** The precomputed ``test.npz`` volumes are the
  already-squashed 64^3 masks ``RAOSVolumes`` serves. Degrading those directly and
  measuring Dice on them keeps degradation, VAE reconstruction and true-Dice all
  in the same coordinate frame (the 3D form of the 2D ``space:"resized"`` rule).

Small-sample caveat (write it into the comparison, not just here): there are only
47 test patients. ``--n-per-mask 5`` yields ~282 evaluation points, which lowers
the *statistical* noise of each rho but does NOT add patient diversity — the
correlation still rests on 47 livers, unlike the 2D run's 9800 slice points.

Outputs
-------
* console: the active 3D primitive set, the post-calibration [dice_min, dice_max]
  per primitive, and a correlation table (global + per band) for every score
* ``outputs/correlation_scatter_3d.png``  — true Dice vs the best score
* ``outputs/score_comparison_3d.png``     — |Spearman rho| per score
* ``outputs/correlation_results_3d.json`` — all rho / r / p / n, global and banded

Usage
-----
    python scripts/evaluate_correlation_3d.py \
        --config configs/vae3d_full.yaml \
        --checkpoint outputs/checkpoints/vae3d_best_latent64.pt \
        --split test --n-per-mask 5 --data-source precomputed --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data import load_config  # noqa: E402
from data.degradations import SyntheticDegrader  # noqa: E402
from data.degradations_3d import PRIMITIVES_3D  # noqa: E402
from data.raos_volume import RAOSVolumes  # noqa: E402
from models.vae3d import ConvVAE3D  # noqa: E402
from quality.score import (  # noqa: E402
    DICE_BANDS,
    SCORE_DIRECTIONS,
    SCORE_FNS,
    banded_correlation,
    compute_all_scores,
    correlation,
)
from train import resolve_device, set_seed  # noqa: E402

logger = logging.getLogger("evaluate_correlation_3d")

N_CALIB_MASKS = 40   # real test volumes used to calibrate degradation Dice ranges
# Volumes are ~D times heavier than slices; a smaller scoring batch keeps a 64^3
# forward within laptop/Colab GPU memory (scoring is no-grad, so this is loose).
SCORE_BATCH = 32


# --------------------------------------------------------------------------
# model + data
# --------------------------------------------------------------------------
def load_model(checkpoint: Path, config: dict[str, Any], device: torch.device) -> ConvVAE3D:
    """Load the ConvVAE3D, asserting the checkpoint's latent_dim matches ``config``.

    The architecture is rebuilt from the checkpoint's OWN config (so input_size /
    base_channels always match the saved weights, e.g. a 64^3 full run); only the
    latent_dim is cross-checked against the passed config to catch an obvious
    config/checkpoint mismatch early.
    """
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ckpt_vae3d = ckpt["config"]["vae3d"]
    ckpt_latent = int(ckpt_vae3d["latent_dim"])
    cfg_latent = int(config["vae3d"]["latent_dim"])
    if ckpt_latent != cfg_latent:
        raise SystemExit(
            f"latent_dim mismatch: checkpoint={ckpt_latent} but config={cfg_latent}. "
            f"Pass the config whose latent_dim matches this checkpoint."
        )
    model = ConvVAE3D(
        latent_dim=ckpt_latent,
        base_channels=int(ckpt_vae3d["base_channels"]),
        input_size=tuple(ckpt_vae3d["input_size"]),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    logger.info(
        "loaded %s (latent_dim=%d, input_size=%s, epoch=%s)",
        checkpoint.name, ckpt_latent, tuple(ckpt_vae3d["input_size"]), ckpt.get("epoch"),
    )
    return model


def load_clean_volumes(
    config: dict[str, Any], split: str, limit: int | None, seed: int
) -> list[np.ndarray]:
    """Every clean test liver as a (D,H,W) bool volume, optionally capped to ``limit``.

    Volumes come from ``RAOSVolumes`` in whatever ``data.data_source`` the config
    selects; in precomputed mode these are the already-squashed 64^3 masks, which
    is what keeps the whole evaluation in one coordinate frame.
    """
    ds = RAOSVolumes.from_config(config, split=split)
    n = len(ds)
    indices = list(range(n))
    if limit and limit < n:
        rng = np.random.default_rng(seed)
        indices = sorted(int(i) for i in rng.choice(n, size=limit, replace=False))
    logger.info("clean volumes: %d of %d test livers", len(indices), n)
    return [np.asarray(ds[i]["mask"], dtype=bool) for i in indices]


def calibrate_degrader(
    config: dict[str, Any], clean_masks: list[np.ndarray], seed: int
) -> SyntheticDegrader:
    """Calibrate 3D degradation Dice ranges on a subset of the *test* volumes.

    Built with ``primitives=PRIMITIVES_3D`` (NOT the default 2D set). Reachable
    ranges are geometry-dependent and differ from 2D — a 3D blob perturbs a
    smaller volume *fraction* than a 2D disk of the same radius — so the measured
    [dice_min, dice_max] per primitive is logged here for inspection.
    """
    degrader = SyntheticDegrader.from_config(config, primitives=PRIMITIVES_3D)
    assert degrader.primitives is PRIMITIVES_3D, "degrader is not using the 3D primitives"
    rng = np.random.default_rng(seed)
    k = min(N_CALIB_MASKS, len(clean_masks))
    idx = rng.choice(len(clean_masks), size=k, replace=False)
    ranges = degrader.calibrate([clean_masks[int(i)] for i in idx])
    logger.info("calibrated 3D degrader on %d test volumes; reachable Dice ranges:", k)
    for name, (lo, hi) in ranges.items():
        logger.info("  %-18s [%.3f, %.3f]", name, lo, hi)
    return degrader


def build_eval_set(
    clean_masks: list[np.ndarray],
    degrader: SyntheticDegrader,
    n_per_mask: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clean volumes (true Dice 1.0) plus ``n_per_mask`` 3D degradations each.

    Returns ``(masks (M,D,H,W) bool, true_dice (M,), is_clean (M,) bool)``. A single
    seeded RNG drives every degradation in order, so the whole set is reproducible.
    """
    rng = np.random.default_rng(seed)
    masks: list[np.ndarray] = []
    dice_vals: list[float] = []
    is_clean: list[bool] = []
    for m in clean_masks:
        masks.append(m.astype(bool))
        dice_vals.append(1.0)
        is_clean.append(True)
        for _ in range(n_per_mask):
            degraded, true_d = degrader.make_pair(m, rng, target_dice=None)
            masks.append(np.asarray(degraded) > 0)
            dice_vals.append(float(true_d))
            is_clean.append(False)
    logger.info("eval set: %d volumes (%d clean + %d degraded)",
                len(masks), sum(is_clean), len(masks) - sum(is_clean))
    return np.stack(masks), np.asarray(dice_vals), np.asarray(is_clean)


# --------------------------------------------------------------------------
# scoring + stats
# --------------------------------------------------------------------------
def score_masks(model, masks: np.ndarray, device: torch.device) -> dict[str, np.ndarray]:
    """Run every quality score over all volumes in batches. ``{name: (M,) array}``.

    ``unsqueeze(1)`` turns (B,D,H,W) into the (B,1,D,H,W) the 3D model expects —
    the only shape difference from the 2D scorer.
    """
    accum: dict[str, list[np.ndarray]] = {name: [] for name in SCORE_FNS}
    for start in range(0, len(masks), SCORE_BATCH):
        chunk = masks[start:start + SCORE_BATCH]
        x = torch.from_numpy(chunk).float().unsqueeze(1).to(device)
        for name, arr in compute_all_scores(model, x).items():
            accum[name].append(arr)
    return {name: np.concatenate(parts) for name, parts in accum.items()}


def summarise(scores: dict[str, np.ndarray], true_dice: np.ndarray) -> dict[str, Any]:
    """Global + banded correlation for each score. Returns a JSON-ready dict."""
    result: dict[str, Any] = {}
    for name, values in scores.items():
        result[name] = {
            "direction": SCORE_DIRECTIONS[name],
            "global": correlation(values, true_dice),
            "bands": banded_correlation(values, true_dice),
        }
    return result


def best_score_name(summary: dict[str, Any]) -> str:
    """Score with the largest |global Spearman rho| (NaNs treated as 0)."""
    def strength(name: str) -> float:
        rho = summary[name]["global"]["spearman"]
        return 0.0 if np.isnan(rho) else abs(rho)
    return max(summary, key=strength)


def print_table(summary: dict[str, Any]) -> None:
    """Print the global + per-band correlation table to the console."""
    band_labels = [f"[{lo:.2f}-{hi:.2f}]" for lo, hi in DICE_BANDS]
    header = f"{'score':<12}{'dir':>4}{'global rho':>12}{'pearson':>10}{'n':>7}   " \
             + "".join(f"{lbl:>14}" for lbl in band_labels)
    print("\n" + header)
    print("-" * len(header))
    for name, info in summary.items():
        g = info["global"]
        row = (f"{name:<12}{info['direction']:>+4}"
               f"{g['spearman']:>12.3f}{g['pearson']:>10.3f}{g['n']:>7}   ")
        for band in info["bands"]:
            rho = band["spearman"]
            cell = "   n/a" if np.isnan(rho) else f"{rho:>+6.3f}"
            row += f"{cell:>14}"
        print(row)
    print()


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------
def plot_scatter(
    true_dice: np.ndarray,
    scores: np.ndarray,
    is_clean: np.ndarray,
    name: str,
    stats: dict[str, float],
    out_path: Path,
) -> None:
    """True Dice vs the best score: clean vs degraded coloured, bands marked."""
    fig, ax = plt.subplots(figsize=(9, 6))
    deg = ~is_clean
    ax.scatter(true_dice[deg], scores[deg], s=12, alpha=0.5, c="#c1121f",
               label=f"degraded (n={int(deg.sum())})")
    ax.scatter(true_dice[is_clean], scores[is_clean], s=24, alpha=0.7, c="#1d3557",
               label=f"clean, Dice=1.0 (n={int(is_clean.sum())})")
    for _, hi in DICE_BANDS[:-1]:
        ax.axvline(hi, color="grey", lw=0.6, ls="--", alpha=0.6)
    order = np.argsort(true_dice)
    coeffs = np.polyfit(true_dice, scores, 1)
    ax.plot(true_dice[order], np.polyval(coeffs, true_dice[order]),
            color="black", lw=1.2, label="linear trend")
    ax.set_xlabel("true Dice")
    ax.set_ylabel(f"{name} score")
    direction = "higher = better" if SCORE_DIRECTIONS[name] > 0 else "higher = worse"
    ax.set_title(
        f"3D — best score: {name} ({direction})\n"
        f"Spearman rho = {stats['spearman']:.3f}   "
        f"Pearson r = {stats['pearson']:.3f}   n = {stats['n']}"
    )
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def plot_score_comparison(summary: dict[str, Any], out_path: Path) -> None:
    """Bar chart of |global Spearman rho| per score — which score wins in 3D."""
    names = list(summary)
    rhos = [abs(summary[n]["global"]["spearman"]) if not np.isnan(summary[n]["global"]["spearman"])
            else 0.0 for n in names]
    order = np.argsort(rhos)[::-1]
    names = [names[i] for i in order]
    rhos = [rhos[i] for i in order]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, rhos, color="#457b9d")
    for bar, rho in zip(bars, rhos):
        ax.text(bar.get_x() + bar.get_width() / 2, rho + 0.01, f"{rho:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("|Spearman rho| vs true Dice")
    ax.set_ylim(0, 1)
    ax.set_title("3D quality-score ranking (higher = better correlation)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correlate 3D VAE reconstruction error with true Dice."
    )
    parser.add_argument("--config", default="configs/vae3d.yaml", help="path to config YAML")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/vae3d_best_latent64.pt",
                        help="trained ConvVAE3D checkpoint")
    parser.add_argument("--split", default="test", help="split to evaluate on")
    parser.add_argument("--n-per-mask", type=int, default=5,
                        help="degraded versions per clean volume (default 5 -> ~282 points)")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of clean volumes for a quick check")
    parser.add_argument("--data-source", choices=("nifti", "precomputed"), default=None,
                        help="override config data.data_source (use 'precomputed' for test.npz)")
    parser.add_argument("--seed", type=int, default=42, help="master seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logging.basicConfig(
        level=getattr(logging, config["logging"]["level"], logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.data_source is not None:
        config.setdefault("data", {})["data_source"] = args.data_source
    logger.info("data source: %s", config.get("data", {}).get("data_source", "nifti"))
    # State the active degradation primitive set up front so it can be verified:
    # this MUST be the 3D set, else 2D primitives would be applied to volumes.
    logger.info("degradation primitive set: 3D (%s)", ", ".join(PRIMITIVES_3D))

    set_seed(args.seed)
    device = resolve_device(config["train"]["device"])

    model = load_model(Path(args.checkpoint), config, device)
    clean_masks = load_clean_volumes(config, args.split, args.limit, args.seed)
    degrader = calibrate_degrader(config, clean_masks, args.seed)
    masks, true_dice, is_clean = build_eval_set(
        clean_masks, degrader, args.n_per_mask, args.seed
    )

    scores = score_masks(model, masks, device)
    summary = summarise(scores, true_dice)
    print_table(summary)

    best = best_score_name(summary)
    logger.info("best score by |global rho|: %s", best)

    out_dir = PROJECT_ROOT / "outputs"
    plot_scatter(true_dice, scores[best], is_clean, best,
                 summary[best]["global"], out_dir / "correlation_scatter_3d.png")
    plot_score_comparison(summary, out_dir / "score_comparison_3d.png")

    results = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "n_per_mask": args.n_per_mask,
        "seed": args.seed,
        "n_patients": int(is_clean.sum()),
        "n_total": int(len(true_dice)),
        "n_clean": int(is_clean.sum()),
        "n_degraded": int((~is_clean).sum()),
        "primitive_set": "3D",
        "calibrated_ranges": {name: list(rng) for name, rng in (degrader.type_ranges or {}).items()},
        "best_score": best,
        "scores": summary,
    }
    results_path = out_dir / "correlation_results_3d.json"
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("wrote %s", results_path)


if __name__ == "__main__":
    main()
