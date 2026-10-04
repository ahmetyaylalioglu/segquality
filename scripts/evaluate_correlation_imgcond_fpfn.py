"""Does the FP/FN dual-head model's predicted error track true Dice? (direction G)

The FP/FN counterpart of :mod:`scripts.evaluate_correlation_imgcond`. The model consumes
``[grayscale CT, broken mask]`` and predicts ``[FP, FN]``; the GT-free quality scores are
therefore derived from the predicted corrections, not from a reconstruction:

* ``est_dice``     — estimated Dice of the input mask vs the unknown clean shape (+1).
* ``error_volume`` — ``|FP| + |FN|`` predicted error voxels (-1, higher = worse).
* ``fp_volume`` / ``fn_volume`` — the two error components alone (-1).

The protocol reuses :mod:`evaluate_correlation_3d` (test split, ``PRIMITIVES_3D``,
calibration, banded correlation, table) and the image-aware clean-volume loader from
:mod:`scripts.evaluate_correlation_imgcond`. Because FP/FN directly target the boundary, we
ALSO report **ASSD** (:mod:`quality.surface_metrics`) of the input mask vs clean and of the
model's corrected mask vs clean — the numeric edge-error improvement the advisor asked for.

Non-clobbering: writes ``correlation_results_imgcond_fpfn_{tag}.json`` and
``*_imgcond_fpfn_{tag}.png``.

Usage
-----
    python scripts/evaluate_correlation_imgcond_fpfn.py --quantizer vq \
        --config configs/vae3d_imgcond_fpfn.yaml \
        --checkpoint outputs/checkpoints/vae3d_imgcond_fpfn_vq_best.pt \
        --tag g_vq --split test --n-per-mask 5 --seed 42
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
for p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from data import load_config  # noqa: E402
from data.degradations_3d import PRIMITIVES_3D  # noqa: E402
from data.fpfn_targets import correct_mask, error_volume, estimated_dice  # noqa: E402
from models.vae3d_imgcond_fpfn import (  # noqa: E402
    ConvVQVAE3D_ImgCond_FPFN,
    ConvVQVAE3D_ImgCond_FPFN_sdf_io,
    ConvVQVAE3D_SimVQ_ImgCond_FPFN,
    ConvVQVAE3D_SimVQ_ImgCond_FPFN_sdf_io,
)
from quality.score import banded_correlation, correlation  # noqa: E402
from quality.sdf_paper import DELTA  # noqa: E402
from quality.surface_metrics import assd  # noqa: E402
from train import resolve_device, set_seed  # noqa: E402

import evaluate_correlation_3d as ev3d  # noqa: E402
import evaluate_correlation_imgcond as evimg  # noqa: E402

logger = logging.getLogger("evaluate_correlation_imgcond_fpfn")

SCORE_BATCH = ev3d.SCORE_BATCH

FPFN_MODELS = {
    ("binary", "vq"): ConvVQVAE3D_ImgCond_FPFN,
    ("binary", "simvq"): ConvVQVAE3D_SimVQ_ImgCond_FPFN,
    ("sdfio", "vq"): ConvVQVAE3D_ImgCond_FPFN_sdf_io,
    ("sdfio", "simvq"): ConvVQVAE3D_SimVQ_ImgCond_FPFN_sdf_io,
}


# --------------------------------------------------------------------------
# GT-free FP/FN scores (computed on x = [image, broken mask])
# --------------------------------------------------------------------------
@torch.no_grad()
def est_dice_score(model, x: torch.Tensor) -> torch.Tensor:
    fp, fn = model.predict_fpfn(x)
    return estimated_dice(x[:, 1:2], fp, fn)


@torch.no_grad()
def error_volume_score(model, x: torch.Tensor) -> torch.Tensor:
    fp, fn = model.predict_fpfn(x)
    return error_volume(fp, fn)


@torch.no_grad()
def fp_volume_score(model, x: torch.Tensor) -> torch.Tensor:
    fp, fn = model.predict_fpfn(x)
    return (fp > 0.5).float().sum(dim=tuple(range(1, fp.dim())))


@torch.no_grad()
def fn_volume_score(model, x: torch.Tensor) -> torch.Tensor:
    fp, fn = model.predict_fpfn(x)
    return (fn > 0.5).float().sum(dim=tuple(range(1, fn.dim())))


def build_score_registry() -> tuple[dict, dict]:
    """Return ``(score_fns, directions)`` for the FP/FN GT-free quality scores."""
    fns = {
        "est_dice": est_dice_score,
        "error_volume": error_volume_score,
        "fp_volume": fp_volume_score,
        "fn_volume": fn_volume_score,
    }
    dirs = {"est_dice": +1, "error_volume": -1, "fp_volume": -1, "fn_volume": -1}
    return fns, dirs


def build_eval_set_fpfn(
    clean_masks: list[np.ndarray], clean_images: list[np.ndarray],
    degrader, n_per_mask: int, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Clean volumes + ``n_per_mask`` degradations each, carrying the patient image AND the
    clean reference mask per row (needed for ASSD). Returns ``(broken, images, clean_ref,
    true_dice, is_clean)``.
    """
    rng = np.random.default_rng(seed)
    broken: list[np.ndarray] = []
    images: list[np.ndarray] = []
    clean_ref: list[np.ndarray] = []
    dice_vals: list[float] = []
    is_clean: list[bool] = []
    for m, img in zip(clean_masks, clean_images):
        mb = m.astype(bool)
        broken.append(mb); images.append(img); clean_ref.append(mb)
        dice_vals.append(1.0); is_clean.append(True)
        for _ in range(n_per_mask):
            degraded, true_d = degrader.make_pair(m, rng, target_dice=None)
            broken.append(np.asarray(degraded) > 0); images.append(img); clean_ref.append(mb)
            dice_vals.append(float(true_d)); is_clean.append(False)
    logger.info("eval set: %d volumes (%d clean + %d degraded), image+clean-ref fixed per patient",
                len(broken), sum(is_clean), len(broken) - sum(is_clean))
    return (np.stack(broken), np.stack(images), np.stack(clean_ref),
            np.asarray(dice_vals), np.asarray(is_clean))


def score_masks_fpfn(
    model, broken: np.ndarray, images: np.ndarray, score_fns: dict, device: torch.device
) -> dict[str, np.ndarray]:
    """Run every GT-free score over ``x = [image, broken mask]`` (B,2,D,H,W) in batches."""
    model.eval()
    accum: dict[str, list[np.ndarray]] = {name: [] for name in score_fns}
    for start in range(0, len(broken), SCORE_BATCH):
        m = broken[start:start + SCORE_BATCH].astype(np.float32)
        img = images[start:start + SCORE_BATCH].astype(np.float32)
        x = torch.from_numpy(np.stack([img, m], axis=1)).to(device)   # ch0=image, ch1=broken
        for name, fn in score_fns.items():
            accum[name].append(fn(model, x).cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in accum.items()}


@torch.no_grad()
def compute_assd_improvement(
    model, broken: np.ndarray, images: np.ndarray, clean_ref: np.ndarray,
    is_clean: np.ndarray, device: torch.device,
) -> dict[str, float]:
    """Mean ASSD of the input mask vs clean and of the corrected mask vs clean (degraded rows).

    The advisor's numeric edge-error check: FP/FN corrections should pull the boundary back
    toward the clean surface, lowering ASSD. Averaged over the DEGRADED rows (clean rows have
    ASSD 0 by construction and would dilute the effect). NaNs (empty surfaces) are skipped.
    """
    model.eval()
    assd_in: list[float] = []
    assd_corr: list[float] = []
    for start in range(0, len(broken), SCORE_BATCH):
        sl = slice(start, start + SCORE_BATCH)
        m = broken[sl].astype(np.float32)
        img = images[sl].astype(np.float32)
        x = torch.from_numpy(np.stack([img, m], axis=1)).to(device)
        fp, fn = model.predict_fpfn(x)
        corrected = correct_mask(x[:, 1:2], fp, fn).cpu().numpy()[:, 0]   # (b,D,H,W)
        for j in range(m.shape[0]):
            if is_clean[start + j]:
                continue
            gt = clean_ref[start + j].astype(bool)
            assd_in.append(assd(broken[start + j].astype(bool), gt))
            assd_corr.append(assd(corrected[j] > 0.5, gt))
    mean_in = float(np.nanmean(assd_in)) if assd_in else float("nan")
    mean_corr = float(np.nanmean(assd_corr)) if assd_corr else float("nan")
    return {
        "assd_input_mean": mean_in,
        "assd_corrected_mean": mean_corr,
        "assd_reduction": mean_in - mean_corr,
        "n_degraded": int(len(assd_in)),
    }


def summarise(scores: dict[str, np.ndarray], true_dice: np.ndarray, directions: dict) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, values in scores.items():
        result[name] = {
            "direction": directions[name],
            "global": correlation(values, true_dice),
            "bands": banded_correlation(values, true_dice),
        }
    return result


def load_fpfn_model(checkpoint: Path, head: str, quantizer: str, device: torch.device):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    v = ckpt["config"]["vae3d_vq"]
    delta = float(ckpt["config"].get("vae3d_vq_sdf_io", {}).get("delta", DELTA))
    kwargs: dict[str, Any] = dict(
        base_channels=int(v["base_channels"]),
        num_embeddings=int(v["num_embeddings"]),
        embedding_dim=int(v["embedding_dim"]),
        commitment_cost=float(v["commitment_cost"]),
        input_size=tuple(v["input_size"]),
        use_skip=bool(v.get("use_skip", False)),
    )
    if head == "sdfio":
        kwargs["delta"] = delta
    model = FPFN_MODELS[(head, quantizer)](**kwargs)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    logger.info("loaded %s (%s head=%s quantizer=%s in=%d out=%d epoch=%s)",
                checkpoint.name, type(model).__name__, head, quantizer,
                model.in_channels, model.out_channels, ckpt.get("epoch"))
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Correlate FP/FN predicted error with true Dice (dir G).")
    p.add_argument("--config", default="configs/vae3d_imgcond_fpfn.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--head", choices=("binary", "sdfio"), default="binary")
    p.add_argument("--quantizer", choices=("vq", "simvq"), default="vq")
    p.add_argument("--tag", default=None, help="output filename tag (default: head_quantizer)")
    p.add_argument("--split", default="test")
    p.add_argument("--n-per-mask", type=int, default=5)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--data-source", choices=("nifti", "precomputed"), default="precomputed")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    config = load_config(args.config)
    logging.basicConfig(
        level=getattr(logging, config.get("logging", {}).get("level", "INFO"), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config.setdefault("data", {})["data_source"] = args.data_source
    logger.info("data source: %s | primitives: 3D (%s) | head=%s quantizer=%s",
                config["data"]["data_source"], ", ".join(PRIMITIVES_3D), args.head, args.quantizer)

    set_seed(args.seed)
    device = resolve_device(config["train"]["device"])

    ckpt_path = args.checkpoint
    if not Path(ckpt_path).is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    model = load_fpfn_model(Path(ckpt_path), args.head, args.quantizer, device)

    clean_masks, clean_images = evimg.load_clean_image_volumes(config, args.split, args.limit, args.seed)
    degrader = ev3d.calibrate_degrader(config, clean_masks, args.seed)
    broken, images, clean_ref, true_dice, is_clean = build_eval_set_fpfn(
        clean_masks, clean_images, degrader, args.n_per_mask, args.seed)

    score_fns, directions = build_score_registry()
    scores = score_masks_fpfn(model, broken, images, score_fns, device)
    summary = summarise(scores, true_dice, directions)
    ev3d.print_table(summary)

    best = ev3d.best_score_name(summary)
    est_rho = summary["est_dice"]["global"]["spearman"]
    logger.info("best score by |global rho|: %s | est_dice rho=%.4f", best, est_rho)

    assd_stats = compute_assd_improvement(model, broken, images, clean_ref, is_clean, device)
    logger.info("ASSD (degraded rows, n=%d): input=%.3f -> corrected=%.3f (reduction=%.3f)",
                assd_stats["n_degraded"], assd_stats["assd_input_mean"],
                assd_stats["assd_corrected_mean"], assd_stats["assd_reduction"])

    out_dir = PROJECT_ROOT / "outputs"
    tag = args.tag or f"{args.head}_{args.quantizer}"
    evimg._plot_scatter_local(true_dice, scores[best], is_clean, best, directions,
                              summary[best]["global"], out_dir / f"correlation_scatter_imgcond_fpfn_{tag}.png")
    ev3d.plot_score_comparison(summary, out_dir / f"score_comparison_imgcond_fpfn_{tag}.png")

    results = {
        "checkpoint": str(args.checkpoint),
        "tag": tag,
        "model": type(model).__name__,
        "head": args.head,
        "quantizer": args.quantizer,
        "split": args.split,
        "n_per_mask": args.n_per_mask,
        "seed": args.seed,
        "n_total": int(len(true_dice)),
        "n_clean": int(is_clean.sum()),
        "n_degraded": int((~is_clean).sum()),
        "primitive_set": "3D",
        "calibrated_ranges": {name: list(rng) for name, rng in (degrader.type_ranges or {}).items()},
        "best_score": best,
        "est_dice_rho": float(est_rho),
        "assd": assd_stats,
        "scores": summary,
    }
    results_path = out_dir / f"correlation_results_imgcond_fpfn_{tag}.json"
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("wrote %s", results_path)


if __name__ == "__main__":
    main()
