"""Faz 2: train the 3D VAR tokenizer WITH the VAR Eq.(5) adversarial term.

Same generator (:class:`models.vae3d_var.VARVQVAE3D`) and data pipeline as
``train_vae3d_var.py``; the only addition is a 3D PatchGAN discriminator
(:class:`models.patchgan3d.PatchGAN3D`) supplying the ``lambda_G * L_GAN`` term of
VAR Eq.(5) ("GAN hatasi"). Hinge objective (VQGAN's choice):

    D: 0.5 * (relu(1 - D(real)).mean() + relu(1 + D(fake.detach())).mean())
    G: recon(BCE + Dice) + vq + lambda_G * (-D(fake).mean())

``fake`` is the reconstruction probability ``sigmoid(logits)``; ``real`` is the
input mask. The adversarial term is warmed up: for the first ``gan_start_epoch``
epochs we train recon + VQ alone (so the discriminator does not overpower a random
generator), then enable GAN.

The GENERATOR checkpoint is written with the SAME ``model_state`` key as the other
3D trainers, so ``scripts/evaluate_correlation_3d_var.py`` scores it unchanged — the
QC comparison vs the GAN-free VAR and the VQ baseline is apples-to-apples. The
discriminator state is saved alongside for resume.

Usage
-----
    python scripts/train_vae3d_var_gan.py --config configs/vae3d_var_gan.yaml
    python scripts/train_vae3d_var_gan.py --config configs/vae3d_var_gan.yaml --limit 8 --epochs 4 --gan-start-epoch 1 --num-workers 0  # smoke
    python scripts/train_vae3d_var_gan.py --config configs/vae3d_var_gan.yaml --resume outputs/checkpoints/vae3d_vargan_last.pt
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
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from data import load_config  # noqa: E402
from data.raos_volume import RAOSVolumes  # noqa: E402
from losses.vae_losses import vq_loss  # noqa: E402
from models.patchgan3d import PatchGAN3D  # noqa: E402
from models.vae3d_var import VARVQVAE3D  # noqa: E402
from train import (  # noqa: E402
    hard_dice,
    improved,
    resolve_device,
    save_checkpoint,
    set_seed,
    sharpness_penalty,
)
from train_vae3d import make_loaders, save_recon_preview_3d  # noqa: E402
from train_vae3d_var import build_model  # noqa: E402
from train_vae3d_vq import log_diag, plot_curves_vq, run_meanreversion_diag  # noqa: E402

logger = logging.getLogger("train_vae3d_var_gan")


def run_epoch_gan(
    model: VARVQVAE3D,
    disc: PatchGAN3D,
    loader: DataLoader,
    device: torch.device,
    *,
    dice_weight: float,
    lambda_g: float,
    gan_active: bool,
    opt_g: torch.optim.Optimizer | None = None,
    opt_d: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """One pass. Training if ``opt_g`` is given, else eval. When ``gan_active`` the
    discriminator is trained (hinge) and its adversarial signal is added to G."""
    training = opt_g is not None
    model.train(training)
    disc.train(training and gan_active)
    totals: dict[str, float] = defaultdict(float)
    perp_sum = used_sum = dice_sum = blur_sum = gadv_sum = dloss_sum = 0.0
    n = 0
    torch.set_grad_enabled(training)
    for x in loader:
        x = x.to(device, non_blocking=True)
        logits, vq_term, info = model(x)
        probs = torch.sigmoid(logits)

        # -- discriminator step (only while GAN active + training) -------------
        d_loss_val = 0.0
        if training and gan_active:
            d_real = disc(x)
            d_fake = disc(probs.detach())
            d_loss = 0.5 * (F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean())
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()
            d_loss_val = float(d_loss.item())

        # -- generator step -----------------------------------------------------
        terms = vq_loss(logits, x, vq_term, dice_weight=dice_weight)
        g_adv = probs.new_zeros(())
        if gan_active:
            g_adv = -disc(probs).mean()
        total_g = terms["total"] + lambda_g * g_adv
        if training:
            opt_g.zero_grad(set_to_none=True)
            total_g.backward()
            opt_g.step()

        bs = x.size(0)
        n += bs
        for key, value in terms.items():
            totals[key] += value.item() * bs
        perp_sum += float(info["perplexity"].item()) * bs
        used_sum += float(info["codebook_used"].item()) * bs
        gadv_sum += float(g_adv.item()) * bs
        dloss_sum += d_loss_val * bs
        if not training:
            dice_sum += hard_dice(probs, x).item() * bs
            blur_sum += sharpness_penalty(probs).item() * bs
    torch.set_grad_enabled(True)
    metrics = {key: totals[key] / n for key in totals}
    metrics["perplexity"] = perp_sum / n
    metrics["codebook_used"] = used_sum / n
    metrics["g_adv"] = gadv_sum / n
    metrics["d_loss"] = dloss_sum / n
    if not training:
        metrics["recon_dice"] = dice_sum / n
        metrics["blur"] = blur_sum / n
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the 3D VAR tokenizer with the Eq.(5) GAN term.")
    p.add_argument("--config", default="configs/vae3d_var_gan.yaml")
    p.add_argument("--resume", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--num-embeddings", type=int, default=None)
    p.add_argument("--using-znorm", dest="using_znorm", action="store_true", default=None)
    p.add_argument("--lambda-g", type=float, default=None, help="override gan.lambda_g")
    p.add_argument("--gan-start-epoch", type=int, default=None, help="override gan.gan_start_epoch")
    p.add_argument("--tag", default="vargan")
    p.add_argument("--num-workers", type=int, default=None)
    return p.parse_args()


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
    if args.lambda_g is not None:
        config["gan"]["lambda_g"] = float(args.lambda_g)
    if args.gan_start_epoch is not None:
        config["gan"]["gan_start_epoch"] = int(args.gan_start_epoch)
    if args.num_workers is not None:
        config["train"]["num_workers"] = int(args.num_workers)

    tcfg = config["train"]
    lcfg = config["loss"]
    gcfg = config["gan"]
    set_seed(int(tcfg["seed"]))
    device = resolve_device(str(tcfg["device"]))

    train_loader, val_loader = make_loaders(config, args.limit)
    val_ds: RAOSVolumes = val_loader.dataset  # type: ignore[assignment]

    model = build_model(config).to(device)
    disc = PatchGAN3D(
        in_channels=model.in_channels,
        base_channels=int(gcfg["disc_base_channels"]),
        n_layers=int(gcfg["disc_n_layers"]),
    ).to(device)
    num_embeddings = int(config["vae3d_var"]["num_embeddings"])
    lambda_g = float(gcfg["lambda_g"])
    gan_start_epoch = int(gcfg["gan_start_epoch"])
    logger.info("VARVQVAE3D(G): %d params | PatchGAN3D(D): %d params | codebook=%d | scales=%s | "
                "lambda_g=%.3f gan_start_epoch=%d znorm=%s",
                model.param_count, disc.param_count, num_embeddings, model.v_patch_nums,
                lambda_g, gan_start_epoch, config["vae3d_var"]["using_znorm"])

    opt_g = torch.optim.Adam(model.parameters(), lr=float(tcfg["lr"]), betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(disc.parameters(), lr=float(gcfg["disc_lr"]), betas=(0.5, 0.9))

    tag = args.tag
    out_dir = PROJECT_ROOT / config["output_dir"]
    ckpt_dir = out_dir / "checkpoints"
    disc_path = ckpt_dir / f"vae3d_{tag}_disc_last.pt"

    start_epoch = 0
    best_dice = -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        opt_g.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_dice = float(ckpt["metrics"].get("best_recon_dice", -1.0))
        if disc_path.exists():
            dstate = torch.load(disc_path, map_location=device, weights_only=False)
            disc.load_state_dict(dstate["disc_state"])
            opt_d.load_state_dict(dstate["disc_opt"])
            logger.info("resumed discriminator from %s", disc_path)
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
    logger.info("run tag=vae3d_%s | epochs=%d diag_every=%d early_stop_patience=%d",
                tag, epochs, diag_every, patience)

    for epoch in range(start_epoch, epochs):
        t0 = time.perf_counter()
        gan_active = epoch >= gan_start_epoch
        train_m = run_epoch_gan(model, disc, train_loader, device, dice_weight=dice_weight,
                                lambda_g=lambda_g, gan_active=gan_active, opt_g=opt_g, opt_d=opt_d)
        val_m = run_epoch_gan(model, disc, val_loader, device, dice_weight=dice_weight,
                              lambda_g=lambda_g, gan_active=gan_active)
        history["train"].append(train_m)
        history["val"].append(val_m)

        save_checkpoint(
            ckpt_dir / f"vae3d_{tag}_last.pt", model, opt_g, epoch, config,
            {**val_m, "best_recon_dice": best_dice},
        )
        torch.save({"disc_state": disc.state_dict(), "disc_opt": opt_d.state_dict()}, disc_path)
        if improved(val_m["recon_dice"], best_dice, min_delta):
            best_dice = val_m["recon_dice"]
            best_epoch = epoch
            epochs_since_improve = 0
            save_checkpoint(
                ckpt_dir / f"vae3d_{tag}_best.pt", model, opt_g, epoch, config,
                {**val_m, "best_recon_dice": best_dice},
            )
        else:
            epochs_since_improve += 1

        logger.info(
            "epoch %d/%d | %.0fs | gan=%s | train=%.4f val=%.4f | recon=%.4f vq=%.4f | "
            "val_dice=%.4f blur=%.4f | g_adv=%+.3f d_loss=%.3f | perp=%.1f(%.0f%%) | best=%.4f @ep%d",
            epoch + 1, epochs, time.perf_counter() - t0, "on" if gan_active else "off",
            train_m["total"], val_m["total"], val_m["recon"], val_m["vq"],
            val_m["recon_dice"], val_m["blur"], train_m["g_adv"], train_m["d_loss"],
            val_m["perplexity"], 100.0 * val_m["perplexity"] / num_embeddings,
            best_dice, best_epoch + 1,
        )

        is_last = epoch + 1 == epochs
        if diag_every and ((epoch + 1) % diag_every == 0 or is_last):
            stats = run_meanreversion_diag(model, val_ds, device, diag_k, diag_seed)
            log_diag(stats, num_embeddings, epoch, epochs)

        if preview_every and (epoch + 1) % preview_every == 0:
            save_recon_preview_3d(model, val_ds, device, out_dir / f"vae3d_{tag}_recon_epoch_{epoch + 1}.png")

        if patience and epochs_since_improve >= patience:
            logger.info("early stop: val recon Dice did not improve by > %.4g for %d epochs",
                        min_delta, patience)
            break

    if history["train"]:
        plot_curves_vq(history, out_dir / f"vae3d_{tag}_training_curves.png",
                       title=f"3D VAR+GAN training curves ({tag})")
    logger.info("done (%s). best val recon Dice=%.4f", tag, best_dice)


if __name__ == "__main__":
    main()
