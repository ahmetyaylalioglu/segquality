"""Train the 3D VAR-style tokenizer (multi-scale residual VQ) on clean RAOS volumes.

A thin sibling of ``train_vae3d_vq.py``: the ONLY difference is the model built
(:class:`models.vae3d_var.VARVQVAE3D` instead of ``ConvVQVAE3D``). The VAR model
returns the SAME ``(logits, vq_loss, info)`` contract with ``info`` holding
``perplexity``/``codebook_used``, so the whole VQ training/eval/diagnosis stack
(``run_epoch_vq``, ``run_meanreversion_diag``, ``log_diag``, ``plot_curves_vq``)
is imported and reused verbatim. This keeps the QC comparison honest: single-scale
VQ (vae3d_vq) vs multi-scale residual VQ (this), same everything else.

GAN-free (VAR Eq.(5) without the lambda_G term). Faz 2 will add the 3D PatchGAN.

Usage
-----
    python scripts/train_vae3d_var.py --config configs/vae3d_var.yaml
    python scripts/train_vae3d_var.py --config configs/vae3d_var.yaml --limit 20 --epochs 6  # smoke
    python scripts/train_vae3d_var.py --config configs/vae3d_var.yaml --resume outputs/checkpoints/vae3d_var_last.pt
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch  # noqa: E402

from data import load_config  # noqa: E402
from data.raos_volume import RAOSVolumes  # noqa: E402
from models.vae3d_var import VARVQVAE3D  # noqa: E402
# Reuse dimension-agnostic helpers from the 2D trainer (unchanged).
from train import improved, resolve_device, save_checkpoint, set_seed  # noqa: E402
# Reuse the 3D data loaders + orthogonal preview from the 3D VAE trainer (unchanged).
from train_vae3d import make_loaders, save_recon_preview_3d  # noqa: E402
# Reuse the VQ training/eval/diagnosis stack verbatim (duck-typed on the model).
from train_vae3d_vq import (  # noqa: E402
    log_diag,
    plot_curves_vq,
    run_epoch_vq,
    run_meanreversion_diag,
)

logger = logging.getLogger("train_vae3d_var")


def build_model(config: dict[str, Any]) -> VARVQVAE3D:
    v = config["vae3d_var"]
    return VARVQVAE3D(
        base_channels=int(v["base_channels"]),
        num_embeddings=int(v["num_embeddings"]),
        embedding_dim=int(v["embedding_dim"]),
        input_size=tuple(v["input_size"]),
        v_patch_nums=tuple(v["v_patch_nums"]),
        using_znorm=bool(v["using_znorm"]),
        beta=float(v["beta"]),
        quant_resi=float(v["quant_resi"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the 3D VAR-style tokenizer on clean volumes.")
    parser.add_argument("--config", default="configs/vae3d_var.yaml", help="path to config YAML")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument("--limit", type=int, default=None, help="cap volumes per split (smoke test)")
    parser.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    parser.add_argument("--num-embeddings", type=int, default=None,
                        help="override codebook size (vae3d_var.num_embeddings); None -> use config")
    parser.add_argument("--using-znorm", dest="using_znorm", action="store_true", default=None,
                        help="force cosine/unit-norm nearest-code lookup (SimVQ-like)")
    parser.add_argument("--tag", default="var",
                        help="checkpoint/output name tag; default 'var'")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="override train.num_workers; None -> use config. Set 0 on Windows to "
                             "avoid DataLoader spawn crashes when run as a subprocess")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logging.basicConfig(
        level=getattr(logging, config["logging"]["level"], logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.epochs is not None:
        config["train"]["epochs"] = int(args.epochs)
    if args.num_embeddings is not None:
        config["vae3d_var"]["num_embeddings"] = int(args.num_embeddings)
    if args.using_znorm is not None:
        config["vae3d_var"]["using_znorm"] = bool(args.using_znorm)
    if args.num_workers is not None:
        config["train"]["num_workers"] = int(args.num_workers)

    tcfg = config["train"]
    lcfg = config["loss"]
    set_seed(int(tcfg["seed"]))
    device = resolve_device(str(tcfg["device"]))

    train_loader, val_loader = make_loaders(config, args.limit)
    val_ds: RAOSVolumes = val_loader.dataset  # type: ignore[assignment]

    model = build_model(config).to(device)
    num_embeddings = int(config["vae3d_var"]["num_embeddings"])
    logger.info("VARVQVAE3D: %d params | codebook=%d x %d | scales=%s | grid=%s | znorm=%s",
                model.param_count, num_embeddings, int(config["vae3d_var"]["embedding_dim"]),
                model.v_patch_nums, model.feature_shape, config["vae3d_var"]["using_znorm"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(tcfg["lr"]))

    tag = args.tag
    out_dir = PROJECT_ROOT / config["output_dir"]
    ckpt_dir = out_dir / "checkpoints"

    start_epoch = 0
    best_dice = -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_dice = float(ckpt["metrics"].get("best_recon_dice", -1.0))
        logger.info("resumed from %s at epoch %d", args.resume, start_epoch)

    epochs = int(tcfg["epochs"])
    dice_weight = float(lcfg["dice_weight"])
    preview_every = int(tcfg["recon_preview_every"])
    diag_every = int(tcfg.get("diag_every", 5))
    diag_k = int(tcfg.get("diag_k", 20))
    diag_seed = int(tcfg.get("diag_seed", 42))
    patience = int(tcfg.get("early_stop_patience", 0))
    min_delta = float(tcfg.get("min_delta", 0.0))
    epochs_since_improve = 0
    best_epoch = start_epoch
    history: dict[str, list[dict[str, float]]] = {"train": [], "val": []}
    logger.info("run tag=vae3d_%s | epochs=%d diag_every=%d diag_k=%d early_stop_patience=%d",
                tag, epochs, diag_every, diag_k, patience)

    for epoch in range(start_epoch, epochs):
        t0 = time.perf_counter()
        train_m = run_epoch_vq(model, train_loader, device, dice_weight=dice_weight, optimizer=optimizer)
        val_m = run_epoch_vq(model, val_loader, device, dice_weight=dice_weight)
        history["train"].append(train_m)
        history["val"].append(val_m)

        save_checkpoint(
            ckpt_dir / f"vae3d_{tag}_last.pt", model, optimizer, epoch, config,
            {**val_m, "best_recon_dice": best_dice},
        )
        if improved(val_m["recon_dice"], best_dice, min_delta):
            best_dice = val_m["recon_dice"]
            best_epoch = epoch
            epochs_since_improve = 0
            save_checkpoint(
                ckpt_dir / f"vae3d_{tag}_best.pt", model, optimizer, epoch, config,
                {**val_m, "best_recon_dice": best_dice},
            )
        else:
            epochs_since_improve += 1

        gap = train_m["total"] - val_m["total"]
        logger.info(
            "epoch %d/%d | %.0fs | train=%.4f val=%.4f | recon=%.4f vq=%.4f | "
            "val_dice=%.4f blur=%.4f | perplexity=%.1f/%d(%.0f%%) | gap=%+.4f | best=%.4f @ep%d",
            epoch + 1, epochs, time.perf_counter() - t0,
            train_m["total"], val_m["total"], val_m["recon"], val_m["vq"],
            val_m["recon_dice"], val_m["blur"],
            val_m["perplexity"], num_embeddings, 100.0 * val_m["perplexity"] / num_embeddings,
            gap, best_dice, best_epoch + 1,
        )

        is_last = epoch + 1 == epochs
        if diag_every and ((epoch + 1) % diag_every == 0 or is_last):
            stats = run_meanreversion_diag(model, val_ds, device, diag_k, diag_seed)
            log_diag(stats, num_embeddings, epoch, epochs)

        if preview_every and (epoch + 1) % preview_every == 0:
            save_recon_preview_3d(model, val_ds, device, out_dir / f"vae3d_{tag}_recon_epoch_{epoch + 1}.png")

        if patience and epochs_since_improve >= patience:
            logger.info("early stop: val recon Dice did not improve by > %.4g for %d epochs "
                        "(best=%.4f at epoch %d)",
                        min_delta, patience, best_dice, epoch + 1 - epochs_since_improve)
            break

    if history["train"]:
        plot_curves_vq(history, out_dir / f"vae3d_{tag}_training_curves.png",
                       title=f"3D VAR tokenizer training curves ({tag})")
    logger.info("done (%s). best val recon Dice=%.4f", tag, best_dice)


if __name__ == "__main__":
    main()
