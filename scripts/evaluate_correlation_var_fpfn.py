"""Does the VAR-backbone FP/FN model's predicted error track true Dice? (dir G, Faz 3)

A thin sibling of :mod:`scripts.evaluate_correlation_imgcond_fpfn`: the ONLY change is the
model built / loaded — :class:`models.vae3d_var_fpfn.VARVQVAE3D_ImgCond_FPFN` (VAR
multi-scale residual quantizer) from a ``vae3d_var`` checkpoint block, instead of the
plain-VQ ``ConvVQVAE3D_ImgCond_FPFN`` from a ``vae3d_vq`` block. Every GT-free score
(``est_dice``, ``error_volume``, ``fp/fn_volume``), the banded correlation protocol, the
calibrated eval set, the corrected-mask ASSD improvement, the table, and all outputs are
reused verbatim from :mod:`evaluate_correlation_imgcond_fpfn` because they only touch the
head-agnostic ``predict_fpfn``.

Binary head only (the established best FP/FN head), matching ``train_var_fpfn.py``.

Non-clobbering: writes ``correlation_results_imgcond_fpfn_{tag}.json`` and
``*_imgcond_fpfn_{tag}.png`` (default tag ``g_var``).

Usage
-----
    python scripts/evaluate_correlation_var_fpfn.py \
        --config configs/vae3d_var_fpfn.yaml \
        --checkpoint outputs/checkpoints/vae3d_g_var_best.pt \
        --tag g_var --split test --n-per-mask 5 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from data import load_config  # noqa: E402
from data.degradations_3d import PRIMITIVES_3D  # noqa: E402
from models.vae3d_var_fpfn import VARVQVAE3D_ImgCond_FPFN  # noqa: E402
from train import resolve_device, set_seed  # noqa: E402

import evaluate_correlation_3d as ev3d  # noqa: E402
import evaluate_correlation_imgcond as evimg  # noqa: E402
import evaluate_correlation_imgcond_fpfn as evf  # noqa: E402

logger = logging.getLogger("evaluate_correlation_var_fpfn")


def load_var_fpfn_model(checkpoint: Path, device: torch.device) -> VARVQVAE3D_ImgCond_FPFN:
    """Build :class:`VARVQVAE3D_ImgCond_FPFN` from a ``vae3d_var`` checkpoint block."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    v = ckpt["config"]["vae3d_var"]
    model = VARVQVAE3D_ImgCond_FPFN(
        base_channels=int(v["base_channels"]),
        num_embeddings=int(v["num_embeddings"]),
        embedding_dim=int(v["embedding_dim"]),
        input_size=tuple(v["input_size"]),
        v_patch_nums=tuple(v["v_patch_nums"]),
        using_znorm=bool(v["using_znorm"]),
        beta=float(v["beta"]),
        quant_resi=float(v["quant_resi"]),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    logger.info("loaded %s (%s | VAR backbone | in=%d out=%d | scales=%s | epoch=%s)",
                checkpoint.name, type(model).__name__, model.in_channels, model.out_channels,
                model.v_patch_nums, ckpt.get("epoch"))
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Correlate VAR-backbone FP/FN predicted error with true Dice (dir G, Faz 3).")
    p.add_argument("--config", default="configs/vae3d_var_fpfn.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tag", default="g_var", help="output filename tag")
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
    logger.info("data source: %s | primitives: 3D (%s) | VAR backbone binary head",
                config["data"]["data_source"], ", ".join(PRIMITIVES_3D))

    set_seed(args.seed)
    device = resolve_device(config["train"]["device"])

    ckpt_path = args.checkpoint
    if not Path(ckpt_path).is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    model = load_var_fpfn_model(Path(ckpt_path), device)

    clean_masks, clean_images = evimg.load_clean_image_volumes(config, args.split, args.limit, args.seed)
    degrader = ev3d.calibrate_degrader(config, clean_masks, args.seed)
    broken, images, clean_ref, true_dice, is_clean = evf.build_eval_set_fpfn(
        clean_masks, clean_images, degrader, args.n_per_mask, args.seed)

    score_fns, directions = evf.build_score_registry()
    scores = evf.score_masks_fpfn(model, broken, images, score_fns, device)
    summary = evf.summarise(scores, true_dice, directions)
    ev3d.print_table(summary)

    best = ev3d.best_score_name(summary)
    est_rho = summary["est_dice"]["global"]["spearman"]
    logger.info("best score by |global rho|: %s | est_dice rho=%.4f", best, est_rho)

    assd_stats = evf.compute_assd_improvement(model, broken, images, clean_ref, is_clean, device)
    logger.info("ASSD (degraded rows, n=%d): input=%.3f -> corrected=%.3f (reduction=%.3f)",
                assd_stats["n_degraded"], assd_stats["assd_input_mean"],
                assd_stats["assd_corrected_mean"], assd_stats["assd_reduction"])

    out_dir = PROJECT_ROOT / "outputs"
    tag = args.tag
    evimg._plot_scatter_local(true_dice, scores[best], is_clean, best, directions,
                              summary[best]["global"], out_dir / f"correlation_scatter_imgcond_fpfn_{tag}.png")
    ev3d.plot_score_comparison(summary, out_dir / f"score_comparison_imgcond_fpfn_{tag}.png")

    results: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "tag": tag,
        "model": type(model).__name__,
        "head": "binary",
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
        "assd": assd_stats,
        "scores": summary,
    }
    results_path = out_dir / f"correlation_results_imgcond_fpfn_{tag}.json"
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("wrote %s", results_path)


if __name__ == "__main__":
    main()
