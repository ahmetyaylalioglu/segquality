"""Stage-1 of the 2026-10-20 direction: train the VAR FP/FN/TP triple-head.

Same 2-channel ``[grayscale CT, broken mask]`` input as the Faz-3 FP/FN model, but the decoder
emits THREE channels ``[FP, FN, TP]`` (the advisor's completed error decomposition). Targets
come from :func:`data.fpfntp_targets.fpfntp_targets` built off the clean mask reconstructed from
the loader's ``[FP, FN]`` targets (the clean mask is never a network input). Loss =
``BCE(FP) + BCE(FN) + BCE(TP) + VQ``. The val metric is the Dice of the reconstructed clean mask
``TP_pred OR FN_pred`` vs the clean mask (checkpoint / early-stop).

Reuses ``train_imgcond_fpfn.make_loaders`` verbatim (so data / degradation / split are
identical to the Faz-3 line). No protected file is touched.

Usage
-----
    python scripts/train_var_fpfntp.py --config configs/vae3d_var_fpfntp.yaml --tag g_var_tp
    python scripts/train_var_fpfntp.py --config configs/vae3d_var_fpfntp.yaml --tag g_var_tp \
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
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data import load_config  # noqa: E402
from data.fpfn_targets import correct_mask  # noqa: E402
from data.fpfntp_targets import fpfntp_targets, reconstruct_from_fntp  # noqa: E402
from models.vae3d_var_fpfntp import VARVQVAE3D_ImgCond_FPFNTP  # noqa: E402
from train import improved, resolve_device, save_checkpoint, set_seed  # noqa: E402
from train_imgcond import _hard_dice  # noqa: E402
from train_imgcond_fpfn import make_loaders  # noqa: E402
from train_vae3d_vq import plot_curves_vq  # noqa: E402

logger = logging.getLogger("train_var_fpfntp")


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


def build_model(config: dict[str, Any]) -> VARVQVAE3D_ImgCond_FPFNTP:
    return VARVQVAE3D_ImgCond_FPFNTP(**_var_kwargs(config["vae3d_var"]))


def run_epoch(
    model: VARVQVAE3D_ImgCond_FPFNTP, loader, device: torch.device,
    *, optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """One pass predicting ``[FP, FN, TP]``; loss = per-map BCE + VQ.

    Val reports ``recon_dice`` = Dice of ``TP_pred OR FN_pred`` vs the clean mask.
    """
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = defaultdict(float)
    perp_sum = used_sum = dice_sum = 0.0
    n = 0
    torch.set_grad_enabled(training)
    for x, target in loader:
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        broken = x[:, 1:2]
        clean = correct_mask(broken, target[:, 0:1], target[:, 1:2])       # (B,1,...) {0,1}
        fp_t, fn_t, tp_t = fpfntp_targets(broken, clean)

        logits, vq_term, info = model.forward(x)
        fp_loss = F.binary_cross_entropy_with_logits(logits[:, 0:1], fp_t)
        fn_loss = F.binary_cross_entropy_with_logits(logits[:, 1:2], fn_t)
        tp_loss = F.binary_cross_entropy_with_logits(logits[:, 2:3], tp_t)
        recon = fp_loss + fn_loss + tp_loss
        total = recon + vq_term
        if training:
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()

        bs = x.size(0)
        n += bs
        totals["total"] += total.item() * bs
        totals["recon"] += recon.item() * bs
        totals["fp"] += fp_loss.item() * bs
        totals["fn"] += fn_loss.item() * bs
        totals["tp"] += tp_loss.item() * bs
        totals["vq"] += vq_term.item() * bs
        perp_sum += float(info["perplexity"].item()) * bs
        used_sum += float(info["codebook_used"].item()) * bs
        if not training:
            probs = torch.sigmoid(logits)
            recon_mask = reconstruct_from_fntp(probs[:, 1:2], probs[:, 2:3])
            dice_sum += _hard_dice(recon_mask, clean) * bs
    torch.set_grad_enabled(True)
    metrics = {k: totals[k] / n for k in totals}
    metrics["perplexity"] = perp_sum / n
    metrics["codebook_used"] = used_sum / n
    metrics.setdefault("bce", 0.0)
    metrics.setdefault("dice", 0.0)
    if not training:
        metrics["recon_dice"] = dice_sum / n
        metrics["blur"] = 0.0
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the VAR FP/FN/TP triple-head (stage-1, 2026-10-20 direction).")
    p.add_argument("--config", default="configs/vae3d_var_fpfntp.yaml")
    p.add_argument("--primitive-set", choices=("mixed", "partial"), default="mixed")
    p.add_argument("--tag", default="g_var_tp")
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

    train_loader, val_loader = make_loaders(
        config, args.limit, seed=int(tcfg["seed"]), primitive_set=args.primitive_set)

    model = build_model(config).to(device)
    num_embeddings = int(config["vae3d_var"]["num_embeddings"])
    logger.info(
        "%s: %d params | in=%d out=%d | codebook=%d | scales=%s | primitive_set=%s",
        type(model).__name__, model.param_count, model.in_channels, model.out_channels,
        num_embeddings, model.v_patch_nums, args.primitive_set,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(tcfg["lr"]))

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
        train_m = run_epoch(model, train_loader, device, optimizer=optimizer)
        val_m = run_epoch(model, val_loader, device)
        history["train"].append(train_m)
        history["val"].append(val_m)

        save_checkpoint(ckpt_dir / f"vae3d_{tag}_last.pt", model, optimizer, epoch, config,
                        {**val_m, "best_recon_dice": best_dice})
        if improved(val_m["recon_dice"], best_dice, min_delta):
            best_dice = val_m["recon_dice"]
            best_epoch = epoch
            epochs_since_improve = 0
            save_checkpoint(ckpt_dir / f"vae3d_{tag}_best.pt", model, optimizer, epoch, config,
                            {**val_m, "best_recon_dice": best_dice})
        else:
            epochs_since_improve += 1

        logger.info(
            "epoch %d/%d | %.0fs | train=%.4f val=%.4f | fp=%.4f fn=%.4f tp=%.4f vq=%.4f | "
            "recon_dice=%.4f | perplexity=%.1f/%d(%.0f%%) | best=%.4f @ep%d",
            epoch + 1, epochs, time.perf_counter() - t0,
            train_m["total"], val_m["total"], val_m["fp"], val_m["fn"], val_m["tp"], val_m["vq"],
            val_m["recon_dice"], val_m["perplexity"], num_embeddings,
            100.0 * val_m["perplexity"] / num_embeddings, best_dice, best_epoch + 1,
        )
        if patience and epochs_since_improve >= patience:
            logger.info("early stop: recon Dice did not improve by > %.4g for %d epochs (best=%.4f)",
                        min_delta, patience, best_dice)
            break

    if history["train"]:
        plot_curves_vq(history, out_dir / f"vae3d_{tag}_training_curves.png",
                       title=f"VAR FP/FN/TP triple-head ({tag})")
    logger.info("done (%s). best recon Dice=%.4f @ep%d", tag, best_dice, best_epoch + 1)


if __name__ == "__main__":
    main()
