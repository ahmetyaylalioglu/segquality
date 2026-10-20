"""Evaluation for the 2026-10-20 direction: how does Dice change across refiner iterations?

Stage-1 = frozen VAR FP/FN/TP (``vae3d_g_var_tp_best.pt``); stage-2 = the trained recurrent
refiner (``vae3d_h_cascade_tp_best.pt``). The pipeline turns ``[image, broken]`` into refined
``[FP, FN, TP]`` at each of ``iters`` passes, and the reconstructed clean mask is ``TP OR FN``.

Per the advisor, the metric of interest is **Dice, tracked across iterations** -- NOT
correlation. On the test split (``PRIMITIVES_3D``, seed 42, degraded rows only) this reports:

* ``base_dice``   -- stage-1 reconstructed-mask Dice vs clean (the accuracy to beat),
* ``dice_iter{i}`` -- mean Dice vs clean after refiner iteration ``i`` (i = 1..iters),
* ``gain_iter{i}`` -- ``dice_iter{i} - base_dice``.

Reuses the FP/FN eval set construction (``evaluate_correlation_imgcond`` /
``_imgcond_fpfn``). Non-clobbering outputs tagged ``h_cascade_tp`` by default.

Usage
-----
    python scripts/evaluate_cascade_tp.py \
        --config configs/vae3d_cascade_tp.yaml \
        --refiner outputs/checkpoints/vae3d_h_cascade_tp_best.pt \
        --tag h_cascade_tp --split test --n-per-mask 5 --seed 42
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
import matplotlib.pyplot as plt  # noqa: E402

from data import load_config  # noqa: E402
from data.degradations_3d import PRIMITIVES_3D  # noqa: E402
from data.fpfntp_targets import reconstruct_from_fntp  # noqa: E402
from models.vae3d_cascade_tp import CascadeFPFNTP, VARVQVAE3D_RefinerTP  # noqa: E402
from models.vae3d_var_fpfntp import VARVQVAE3D_ImgCond_FPFNTP  # noqa: E402
from quality.surface_metrics import assd  # noqa: E402
from train import resolve_device, set_seed  # noqa: E402

import evaluate_correlation_3d as ev3d  # noqa: E402
import evaluate_correlation_imgcond as evimg  # noqa: E402
import evaluate_correlation_imgcond_fpfn as evf  # noqa: E402

logger = logging.getLogger("evaluate_cascade_tp")

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


def load_cascade(config: dict[str, Any], refiner_ckpt: Path, device: torch.device) -> CascadeFPFNTP:
    v = config["vae3d_var"]
    ccfg = config["cascade"]
    stage1 = VARVQVAE3D_ImgCond_FPFNTP(**_var_kwargs(v))
    s1_path = ccfg["stage1_checkpoint"]
    if not Path(s1_path).is_absolute():
        s1_path = PROJECT_ROOT / s1_path
    s1 = torch.load(s1_path, map_location="cpu", weights_only=False)
    stage1.load_state_dict(s1["model_state"])
    refiner = VARVQVAE3D_RefinerTP(in_channels=int(ccfg.get("refiner_in_channels", 4)), **_var_kwargs(v))
    r = torch.load(refiner_ckpt, map_location="cpu", weights_only=False)
    refiner.load_state_dict(r["model_state"])
    iters = int(ccfg.get("iters", 3))
    model = CascadeFPFNTP(stage1, refiner, iters=iters).to(device)
    model.eval()
    logger.info("cascade loaded: stage1=%s (ep%s, frozen) + refiner=%s (ep%s) | iters=%d | in=%d out=%d",
                Path(s1_path).name, s1.get("epoch"), Path(refiner_ckpt).name, r.get("epoch"),
                iters, model.in_channels, model.out_channels)
    return model


def _dice_np(a: np.ndarray, b: np.ndarray, eps: float = 1.0) -> float:
    inter = np.logical_and(a, b).sum()
    card = a.sum() + b.sum()
    return (2.0 * inter + eps) / (card + eps)


@torch.no_grad()
def dice_per_iteration(
    model: CascadeFPFNTP, broken, images, clean_ref, is_clean, device,
) -> dict[str, Any]:
    """Mean Dice (and ASSD) of the reconstructed mask at stage-1 and each refiner iteration.

    Evaluated over the DEGRADED rows only (the advisor cares about correcting broken masks).
    """
    model.eval()
    iters = model.iters
    dice_base: list[float] = []
    dice_iters: dict[int, list[float]] = {i: [] for i in range(1, iters + 1)}
    assd_base: list[float] = []
    assd_iters: dict[int, list[float]] = {i: [] for i in range(1, iters + 1)}
    for start in range(0, len(broken), SCORE_BATCH):
        m = broken[start:start + SCORE_BATCH].astype(np.float32)
        img = images[start:start + SCORE_BATCH].astype(np.float32)
        x = torch.from_numpy(np.stack([img, m], axis=1)).to(device)
        fp0, fn0, tp0 = model.stage1_products(x)
        base = (reconstruct_from_fntp(fn0, tp0) > 0.5).cpu().numpy()[:, 0]
        outputs, _ = model.forward(x)
        recon_iters = {
            i: (reconstruct_from_fntp(fn_i, tp_i) > 0.5).cpu().numpy()[:, 0]
            for i, (_, fn_i, tp_i) in enumerate(outputs, start=1)
        }
        for j in range(m.shape[0]):
            if is_clean[start + j]:
                continue
            gt = clean_ref[start + j].astype(bool)
            b = base[j].astype(bool)
            dice_base.append(_dice_np(b, gt))
            assd_base.append(assd(b, gt))
            for i in range(1, iters + 1):
                r = recon_iters[i][j].astype(bool)
                dice_iters[i].append(_dice_np(r, gt))
                assd_iters[i].append(assd(r, gt))
    base_mean = float(np.mean(dice_base)) if dice_base else float("nan")
    result: dict[str, Any] = {
        "n_degraded": int(len(dice_base)),
        "base_dice": base_mean,
        "base_assd": float(np.nanmean(assd_base)) if assd_base else float("nan"),
    }
    for i in range(1, iters + 1):
        d = float(np.mean(dice_iters[i])) if dice_iters[i] else float("nan")
        result[f"dice_iter{i}"] = d
        result[f"gain_iter{i}"] = d - base_mean
        result[f"assd_iter{i}"] = float(np.nanmean(assd_iters[i])) if assd_iters[i] else float("nan")
    return result


def plot_dice_curve(result: dict[str, Any], iters: int, path: Path) -> None:
    xs = list(range(0, iters + 1))
    ys = [result["base_dice"]] + [result[f"dice_iter{i}"] for i in range(1, iters + 1)]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(xs, ys, marker="o")
    ax.set_xticks(xs)
    ax.set_xticklabels(["stage-1"] + [f"iter {i}" for i in range(1, iters + 1)])
    ax.set_ylabel("mean Dice vs clean")
    ax.set_title("Refiner Dice across iterations (degraded rows)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dice-per-iteration eval for the recurrent TP refiner (2026-10-20).")
    p.add_argument("--config", default="configs/vae3d_cascade_tp.yaml")
    p.add_argument("--refiner", required=True, help="stage-2 refiner checkpoint")
    p.add_argument("--tag", default="h_cascade_tp")
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
    logger.info("data source: %s | primitives: 3D (%s) | recurrent TP cascade",
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

    result = dice_per_iteration(model, broken, images, clean_ref, is_clean, device)
    iters = model.iters
    logger.info("Dice across iterations (degraded rows, n=%d):", result["n_degraded"])
    logger.info("  stage-1 base = %.4f (ASSD %.3f)", result["base_dice"], result["base_assd"])
    for i in range(1, iters + 1):
        logger.info("  iter %d       = %.4f (gain %+.4f, ASSD %.3f)",
                    i, result[f"dice_iter{i}"], result[f"gain_iter{i}"], result[f"assd_iter{i}"])

    out_dir = PROJECT_ROOT / "outputs"
    plot_dice_curve(result, iters, out_dir / f"dice_per_iteration_{args.tag}.png")

    payload: dict[str, Any] = {
        "refiner_checkpoint": str(args.refiner),
        "stage1_checkpoint": config["cascade"]["stage1_checkpoint"],
        "tag": args.tag,
        "model": "CascadeFPFNTP",
        "backbone": "var",
        "iters": iters,
        "split": args.split,
        "n_per_mask": args.n_per_mask,
        "seed": args.seed,
        "n_total": int(len(true_dice)),
        "n_degraded": int((~is_clean).sum()),
        "primitive_set": "3D",
        "calibrated_ranges": {name: list(rng) for name, rng in (degrader.type_ranges or {}).items()},
        "dice_per_iteration": result,
    }
    results_path = out_dir / f"cascade_tp_results_{args.tag}.json"
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("wrote %s", results_path)


if __name__ == "__main__":
    main()
