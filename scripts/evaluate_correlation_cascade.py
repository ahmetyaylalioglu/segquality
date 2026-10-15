"""Faz 4: does the cascade 2nd VAE increase the produced reference mask's accuracy, and does
it give a usable GT-free quality signal?

Stage-1 = frozen VAR FP/FN (``vae3d_g_var_best.pt``); stage-2 = the trained refiner
(``vae3d_h_cascade_best.pt``). The pipeline turns ``[image, broken]`` into a refined reference
mask. Two questions, reported side by side (test split, ``PRIMITIVES_3D``, seed 42):

1. **Accuracy (advisor's question):** mean ``Dice(refined, clean)`` vs ``Dice(corrected, clean)``
   and the same for ASSD, over the DEGRADED rows — does the 2nd VAE beat the stage-1 corrected
   mask? ``accuracy_gain = dice_refined - dice_corrected``.
2. **QC (project gate):** correlate GT-free scores with true Dice:
   * ``est_dice``       — stage-1's estimate (unchanged by cascade; the Faz-3 signal, baseline).
   * ``cascade_agree``  — ``Dice(corrected, refined)`` (+1): how much the shape-prior had to move
     the mask; low agreement = low-quality input. A NEW cascade-specific GT-free score.
   * ``refine_delta``   — ``|refined XOR corrected|`` voxel count (-1): the dual of the above.

Reuses the Faz-3 FP/FN eval machinery (``evaluate_correlation_imgcond_fpfn`` eval set, scoring,
summarise, ASSD) and the ``evaluate_correlation_3d`` calibration/table. Non-clobbering outputs
tagged ``h_cascade`` by default.

Usage
-----
    python scripts/evaluate_correlation_cascade.py \
        --config configs/vae3d_cascade.yaml \
        --refiner outputs/checkpoints/vae3d_h_cascade_best.pt \
        --tag h_cascade --split test --n-per-mask 5 --seed 42
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
from data.fpfn_targets import estimated_dice  # noqa: E402
from models.vae3d_cascade import CascadeFPFN, VARVQVAE3D_Refiner  # noqa: E402
from models.vae3d_var_fpfn import VARVQVAE3D_ImgCond_FPFN  # noqa: E402
from quality.score import banded_correlation, correlation  # noqa: E402
from quality.surface_metrics import assd  # noqa: E402
from train import resolve_device, set_seed  # noqa: E402

import evaluate_correlation_3d as ev3d  # noqa: E402
import evaluate_correlation_imgcond as evimg  # noqa: E402
import evaluate_correlation_imgcond_fpfn as evf  # noqa: E402

logger = logging.getLogger("evaluate_correlation_cascade")

SCORE_BATCH = ev3d.SCORE_BATCH


def _var_kwargs(v: dict[str, Any]) -> dict[str, Any]:
    return dict(
        base_channels=int(v["base_channels"]),
        num_embeddings=int(v["num_embeddings"]),
        embedding_dim=int(v["embedding_dim"]),
        input_size=tuple(v["input_size"]),
        v_patch_nums=tuple(v["v_patch_nums"]),
        using_znorm=bool(v["using_znorm"]),
        beta=float(v["beta"]),
        quant_resi=float(v["quant_resi"]),
    )


def load_cascade(config: dict[str, Any], refiner_ckpt: Path, device: torch.device) -> CascadeFPFN:
    v = config["vae3d_var"]
    ccfg = config["cascade"]
    stage1 = VARVQVAE3D_ImgCond_FPFN(**_var_kwargs(v))
    s1_path = ccfg["stage1_checkpoint"]
    if not Path(s1_path).is_absolute():
        s1_path = PROJECT_ROOT / s1_path
    s1 = torch.load(s1_path, map_location="cpu", weights_only=False)
    stage1.load_state_dict(s1["model_state"])
    refiner = VARVQVAE3D_Refiner(in_channels=int(ccfg.get("refiner_in_channels", 3)), **_var_kwargs(v))
    r = torch.load(refiner_ckpt, map_location="cpu", weights_only=False)
    refiner.load_state_dict(r["model_state"])
    model = CascadeFPFN(stage1, refiner).to(device)
    model.eval()
    logger.info("cascade loaded: stage1=%s (ep%s, frozen) + refiner=%s (ep%s) | refiner in=%d out=%d",
                Path(s1_path).name, s1.get("epoch"), Path(refiner_ckpt).name, r.get("epoch"),
                model.in_channels, model.out_channels)
    return model


def _reduce_dims(x: torch.Tensor) -> tuple[int, ...]:
    return tuple(range(1, x.dim()))


@torch.no_grad()
def _hard_dice_batch(a: torch.Tensor, b: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Per-sample hard Dice between two {0,1} tensors (B,1,...)."""
    af = (a > 0.5).float()
    bf = (b > 0.5).float()
    dims = _reduce_dims(af)
    inter = (af * bf).sum(dim=dims)
    card = af.sum(dim=dims) + bf.sum(dim=dims)
    return (2.0 * inter + eps) / (card + eps)


# --------------------------------------------------------------------------
# GT-free cascade scores, computed on x = [image, broken mask]
# --------------------------------------------------------------------------
@torch.no_grad()
def est_dice_score(model: CascadeFPFN, x: torch.Tensor) -> torch.Tensor:
    fp, fn = model.stage1.predict_fpfn(x)
    return estimated_dice(x[:, 1:2], fp, fn)


@torch.no_grad()
def cascade_agree_score(model: CascadeFPFN, x: torch.Tensor) -> torch.Tensor:
    """Dice(corrected, refined): how much the shape-prior agreed with the stage-1 mask (+1)."""
    corrected, fp, fn = model.stage1_products(x)
    logits, _, _ = model.refiner.forward(model.refiner_input(corrected, fp, fn))
    refined = (torch.sigmoid(logits) > 0.5).float()
    return _hard_dice_batch(corrected, refined)


@torch.no_grad()
def refine_delta_score(model: CascadeFPFN, x: torch.Tensor) -> torch.Tensor:
    """|refined XOR corrected| voxel count (-1, higher = more correction = worse input)."""
    corrected, fp, fn = model.stage1_products(x)
    logits, _, _ = model.refiner.forward(model.refiner_input(corrected, fp, fn))
    refined = (torch.sigmoid(logits) > 0.5).float()
    c = (corrected > 0.5).float()
    dims = _reduce_dims(c)
    return (refined - c).abs().sum(dim=dims)


def build_score_registry() -> tuple[dict, dict]:
    fns = {
        "est_dice": est_dice_score,
        "cascade_agree": cascade_agree_score,
        "refine_delta": refine_delta_score,
    }
    dirs = {"est_dice": +1, "cascade_agree": +1, "refine_delta": -1}
    return fns, dirs


def score_masks_cascade(model, broken, images, score_fns, device) -> dict[str, np.ndarray]:
    model.eval()
    accum: dict[str, list] = {name: [] for name in score_fns}
    for start in range(0, len(broken), SCORE_BATCH):
        m = broken[start:start + SCORE_BATCH].astype(np.float32)
        img = images[start:start + SCORE_BATCH].astype(np.float32)
        x = torch.from_numpy(np.stack([img, m], axis=1)).to(device)
        for name, fn in score_fns.items():
            accum[name].append(fn(model, x).cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in accum.items()}


def summarise(scores, true_dice, directions) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, values in scores.items():
        result[name] = {
            "direction": directions[name],
            "global": correlation(values, true_dice),
            "bands": banded_correlation(values, true_dice),
        }
    return result


@torch.no_grad()
def compute_accuracy(model: CascadeFPFN, broken, images, clean_ref, is_clean, device) -> dict[str, float]:
    """Mean Dice & ASSD of stage-1 corrected vs clean and cascade refined vs clean (degraded rows)."""
    model.eval()
    dice_corr: list[float] = []
    dice_ref: list[float] = []
    assd_corr: list[float] = []
    assd_ref: list[float] = []
    for start in range(0, len(broken), SCORE_BATCH):
        m = broken[start:start + SCORE_BATCH].astype(np.float32)
        img = images[start:start + SCORE_BATCH].astype(np.float32)
        x = torch.from_numpy(np.stack([img, m], axis=1)).to(device)
        corrected, fp, fn = model.stage1_products(x)
        logits, _, _ = model.refiner.forward(model.refiner_input(corrected, fp, fn))
        refined = (torch.sigmoid(logits) > 0.5).cpu().numpy()[:, 0]
        corr_np = (corrected > 0.5).cpu().numpy()[:, 0]
        for j in range(m.shape[0]):
            if is_clean[start + j]:
                continue
            gt = clean_ref[start + j].astype(bool)
            c = corr_np[j].astype(bool)
            r = refined[j].astype(bool)
            dice_corr.append(float(_dice_np(c, gt)))
            dice_ref.append(float(_dice_np(r, gt)))
            assd_corr.append(assd(c, gt))
            assd_ref.append(assd(r, gt))
    return {
        "dice_corrected_mean": float(np.mean(dice_corr)) if dice_corr else float("nan"),
        "dice_refined_mean": float(np.mean(dice_ref)) if dice_ref else float("nan"),
        "dice_gain": float(np.mean(dice_ref) - np.mean(dice_corr)) if dice_ref else float("nan"),
        "assd_corrected_mean": float(np.nanmean(assd_corr)) if assd_corr else float("nan"),
        "assd_refined_mean": float(np.nanmean(assd_ref)) if assd_ref else float("nan"),
        "assd_reduction": float(np.nanmean(assd_corr) - np.nanmean(assd_ref)) if assd_ref else float("nan"),
        "n_degraded": int(len(dice_ref)),
    }


def _dice_np(a: np.ndarray, b: np.ndarray, eps: float = 1.0) -> float:
    inter = np.logical_and(a, b).sum()
    card = a.sum() + b.sum()
    return (2.0 * inter + eps) / (card + eps)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cascade 2nd-VAE accuracy + GT-free QC (dir H, Faz 4).")
    p.add_argument("--config", default="configs/vae3d_cascade.yaml")
    p.add_argument("--refiner", required=True, help="stage-2 refiner checkpoint")
    p.add_argument("--tag", default="h_cascade")
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
    logger.info("data source: %s | primitives: 3D (%s) | cascade 2-VAE",
                config["data"]["data_source"], ", ".join(PRIMITIVES_3D))

    set_seed(args.seed)
    device = resolve_device(config["train"]["device"])

    ref_path = args.refiner
    if not Path(ref_path).is_absolute():
        ref_path = PROJECT_ROOT / ref_path
    model = load_cascade(config, Path(ref_path), device)

    clean_masks, clean_images = evimg.load_clean_image_volumes(config, args.split, args.limit, args.seed)
    degrader = ev3d.calibrate_degrader(config, clean_masks, args.seed)
    broken, images, clean_ref, true_dice, is_clean = evf.build_eval_set_fpfn(
        clean_masks, clean_images, degrader, args.n_per_mask, args.seed)

    score_fns, directions = build_score_registry()
    scores = score_masks_cascade(model, broken, images, score_fns, device)
    summary = summarise(scores, true_dice, directions)
    ev3d.print_table(summary)

    best = ev3d.best_score_name(summary)
    est_rho = summary["est_dice"]["global"]["spearman"]
    logger.info("best QC score by |global rho|: %s | est_dice rho=%.4f", best, est_rho)

    acc = compute_accuracy(model, broken, images, clean_ref, is_clean, device)
    logger.info("ACCURACY (degraded rows, n=%d): Dice corrected=%.4f -> refined=%.4f (gain=%+.4f)",
                acc["n_degraded"], acc["dice_corrected_mean"], acc["dice_refined_mean"], acc["dice_gain"])
    logger.info("ASSD (degraded rows): corrected=%.3f -> refined=%.3f (reduction=%+.3f)",
                acc["assd_corrected_mean"], acc["assd_refined_mean"], acc["assd_reduction"])

    out_dir = PROJECT_ROOT / "outputs"
    tag = args.tag
    evimg._plot_scatter_local(true_dice, scores[best], is_clean, best, directions,
                              summary[best]["global"], out_dir / f"correlation_scatter_cascade_{tag}.png")
    ev3d.plot_score_comparison(summary, out_dir / f"score_comparison_cascade_{tag}.png")

    results: dict[str, Any] = {
        "refiner_checkpoint": str(args.refiner),
        "stage1_checkpoint": config["cascade"]["stage1_checkpoint"],
        "tag": tag,
        "model": "CascadeFPFN",
        "backbone": "var",
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
        "accuracy": acc,
        "scores": summary,
    }
    results_path = out_dir / f"correlation_results_cascade_{tag}.json"
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("wrote %s", results_path)


if __name__ == "__main__":
    main()
