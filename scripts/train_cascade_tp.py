"""Stage-2 of the 2026-10-20 direction: train the recurrent image-aware TP refiner.

Stage-1 = the FROZEN VAR FP/FN/TP model (``vae3d_g_var_tp_best.pt``) turns ``[image, broken]``
into ``[FP, FN, TP]``. Stage-2 (trained here) is a VAR refiner taking ``[image, FP, FN, TP]``
(image kept so it "learns from the grayscale", per the advisor) and predicting refined
``[FP, FN, TP]``; it is applied RECURRENTLY for ``iters`` passes (advisor: 2nd and 3rd
iteration), feeding its output back with the SAME image and shared weights. Only the stage-2
refiner is optimized.

Loss (:mod:`losses.fpfntp_losses`) is computed at EACH iteration: per-map BCE on FP/FN/TP plus
image-aware MSE terms (each map x image and x GT, per-map and combined ``TP+FN`` reconstruction).
The val metric is the Dice of ``TP_pred OR FN_pred`` vs clean at the FINAL iteration; per-epoch
logs also show the Dice at every iteration so the advisor's "how does Dice change at iter 2/3"
question is visible during training. The checkpoint stores only the stage-2 refiner.

Usage
-----
    python scripts/train_cascade_tp.py --config configs/vae3d_cascade_tp.yaml --tag h_cascade_tp
    python scripts/train_cascade_tp.py --config configs/vae3d_cascade_tp.yaml --tag h_cascade_tp \
        --epochs 2 --limit 24 --num-workers 0   # smoke
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data import load_config  # noqa: E402
from data.fpfn_targets import correct_mask  # noqa: E402
from data.fpfntp_targets import fpfntp_targets, reconstruct_from_fntp  # noqa: E402
from losses.fpfntp_losses import FPFNTPLossWeights, cascade_loss  # noqa: E402
from models.vae3d_cascade_tp import CascadeFPFNTP, VARVQVAE3D_RefinerTP  # noqa: E402
from models.vae3d_var_fpfntp import VARVQVAE3D_ImgCond_FPFNTP  # noqa: E402
from train import improved, resolve_device, save_checkpoint, set_seed  # noqa: E402
from train_imgcond import _hard_dice  # noqa: E402
from train_imgcond_fpfn import make_loaders  # noqa: E402
from train_vae3d_vq import plot_curves_vq  # noqa: E402

logger = logging.getLogger("train_cascade_tp")


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


def _loss_weights(config: dict[str, Any]) -> FPFNTPLossWeights:
    lw = config.get("loss", {})
    return FPFNTPLossWeights(
        bce=float(lw.get("bce", 1.0)),
        dice=float(lw.get("dice", 1.0)),
        mse_per_map_image=float(lw.get("mse_per_map_image", 1.0)),
        mse_per_map_gt=float(lw.get("mse_per_map_gt", 1.0)),
        mse_combined_image=float(lw.get("mse_combined_image", 1.0)),
        mse_combined_gt=float(lw.get("mse_combined_gt", 1.0)),
    )


def build_cascade(config: dict[str, Any], device: torch.device) -> CascadeFPFNTP:
    """Load the frozen stage-1 FP/FN/TP model and build a fresh recurrent stage-2 refiner."""
    v = config["vae3d_var"]
    ccfg = config["cascade"]
    stage1 = VARVQVAE3D_ImgCond_FPFNTP(**_var_kwargs(v))
    ckpt_path = ccfg["stage1_checkpoint"]
    if not Path(ckpt_path).is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    stage1.load_state_dict(ckpt["model_state"])
    logger.info("stage-1 (frozen) loaded from %s (epoch=%s)", Path(ckpt_path).name, ckpt.get("epoch"))
    refiner = VARVQVAE3D_RefinerTP(
        in_channels=int(ccfg.get("refiner_in_channels", 4)), **_var_kwargs(v))
    iters = int(ccfg.get("iters", 3))
    return CascadeFPFNTP(stage1, refiner, iters=iters).to(device)


def run_epoch(
    model: CascadeFPFNTP, loader, device: torch.device, weights: FPFNTPLossWeights,
    *, optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """One pass: frozen stage-1 -> recurrent refiner; loss summed over iterations.

    Val reports ``dice_iterN`` (N = 1..iters) = Dice of ``TP OR FN`` vs clean at each iteration,
    ``base_dice`` = the stage-1 reconstructed-mask Dice, and ``recon_dice`` = the final iteration.
    """
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = defaultdict(float)
    dice_iter_sum: dict[int, float] = defaultdict(float)
    base_sum = 0.0
    n = 0
    torch.set_grad_enabled(training)
    for x, target in loader:
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        broken = x[:, 1:2]
        image = x[:, 0:1]
        clean = correct_mask(broken, target[:, 0:1], target[:, 1:2])
        fp_t, fn_t, tp_t = fpfntp_targets(broken, clean)

        outputs, vq_total = model.forward(x)
        maps_loss, per_iter = cascade_loss(outputs, fp_t, fn_t, tp_t, image, clean, weights)
        total = maps_loss + vq_total
        if training:
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()

        bs = x.size(0)
        n += bs
        totals["total"] += total.item() * bs
        totals["recon"] += maps_loss.item() * bs
        totals["vq"] += float(vq_total.item()) * bs
        # aggregate first-iteration components for logging
        for k, v in per_iter[0].items():
            totals[k] += float(v.item()) * bs
        if not training:
            with torch.no_grad():
                fp0, fn0, tp0 = model.stage1_products(x)
                base_sum += _hard_dice(reconstruct_from_fntp(fn0, tp0), clean) * bs
                for i, (_, fn_i, tp_i) in enumerate(outputs, start=1):
                    recon_mask = reconstruct_from_fntp(fn_i, tp_i)
                    dice_iter_sum[i] += _hard_dice(recon_mask, clean) * bs
    torch.set_grad_enabled(True)
    metrics = {k: totals[k] / n for k in totals}
    if not training:
        for i, s in dice_iter_sum.items():
            metrics[f"dice_iter{i}"] = s / n
        metrics["base_dice"] = base_sum / n
        metrics["recon_dice"] = metrics[f"dice_iter{len(dice_iter_sum)}"]
    # keep plot_curves_vq happy (needs total/recon/vq + val recon_dice/blur/perplexity)
    metrics.setdefault("dice", 0.0)
    metrics.setdefault("perplexity", 0.0)
    metrics.setdefault("codebook_used", 0.0)
    if not training:
        metrics.setdefault("blur", 0.0)
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the recurrent image-aware TP refiner (stage-2, 2026-10-20 direction).")
    p.add_argument("--config", default="configs/vae3d_cascade_tp.yaml")
    p.add_argument("--primitive-set", choices=("mixed", "partial"), default="mixed")
    p.add_argument("--tag", default="h_cascade_tp")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    config = load_config(args.config)
    logging.basicConfig(level=getattr(logging, config["logging"]["level"], logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config.setdefault("data", {})["data_source"] = config.get("data", {}).get("data_source", "precomputed")
    if args.epochs is not None:
        config["train"]["epochs"] = int(args.epochs)
    if args.num_workers is not None:
        config["train"]["num_workers"] = int(args.num_workers)

    tcfg = config["train"]
    set_seed(int(tcfg["seed"]))
    device = resolve_device(str(tcfg["device"]))
    weights = _loss_weights(config)

    train_loader, val_loader = make_loaders(
        config, args.limit, seed=int(tcfg["seed"]), primitive_set=args.primitive_set)

    model = build_cascade(config, device)
    logger.info(
        "%s: stage-2 %d trainable params | refiner in=%d out=%d | iters=%d | scales=%s | primitive_set=%s",
        type(model).__name__, model.param_count, model.in_channels, model.out_channels,
        model.iters, model.v_patch_nums, args.primitive_set,
    )
    optimizer = torch.optim.Adam(model.refiner.parameters(), lr=float(tcfg["lr"]))

    tag = args.tag
    out_dir = PROJECT_ROOT / config["output_dir"]
    ckpt_dir = out_dir / "checkpoints"
    epochs = int(tcfg["epochs"])
    patience = int(tcfg.get("early_stop_patience", 0))
    min_delta = float(tcfg.get("min_delta", 0.0))
    best_dice = -1.0
    best_epoch = 0
    epochs_since_improve = 0
    history: dict[str, list[dict[str, float]]] = {"train": [], "val": []}
    logger.info("run tag=vae3d_%s | epochs=%d early_stop_patience=%d", tag, epochs, patience)

    for epoch in range(epochs):
        t0 = time.perf_counter()
        train_m = run_epoch(model, train_loader, device, weights, optimizer=optimizer)
        val_m = run_epoch(model, val_loader, device, weights)
        history["train"].append(train_m)
        history["val"].append(val_m)

        save_checkpoint(ckpt_dir / f"vae3d_{tag}_last.pt", model.refiner, optimizer, epoch, config,
                        {**val_m, "best_recon_dice": best_dice})
        if improved(val_m["recon_dice"], best_dice, min_delta):
            best_dice = val_m["recon_dice"]
            best_epoch = epoch
            epochs_since_improve = 0
            save_checkpoint(ckpt_dir / f"vae3d_{tag}_best.pt", model.refiner, optimizer, epoch, config,
                            {**val_m, "best_recon_dice": best_dice})
        else:
            epochs_since_improve += 1

        iter_str = " ".join(
            f"iter{i}={val_m[f'dice_iter{i}']:.4f}" for i in range(1, model.iters + 1))
        logger.info(
            "epoch %d/%d | %.0fs | train=%.4f val=%.4f | %s | stage1_base=%.4f gain=%+.4f | best=%.4f @ep%d",
            epoch + 1, epochs, time.perf_counter() - t0,
            train_m["total"], val_m["total"], iter_str,
            val_m["base_dice"], val_m["recon_dice"] - val_m["base_dice"], best_dice, best_epoch + 1,
        )
        if patience and epochs_since_improve >= patience:
            logger.info("early stop: final-iter Dice did not improve by > %.4g for %d epochs (best=%.4f)",
                        min_delta, patience, best_dice)
            break

    if history["train"]:
        plot_curves_vq(history, out_dir / f"vae3d_{tag}_training_curves.png",
                       title=f"Recurrent image-aware TP refiner ({tag})")
    logger.info("done (%s). best final-iter Dice=%.4f @ep%d", tag, best_dice, best_epoch + 1)


if __name__ == "__main__":
    main()
