"""Train the shape-VAE on clean RAOS liver masks.

Only clean ground-truth masks are used (the degradation engine is never touched
here — it belongs to Phase 6 evaluation). The VAE learns "what a normal liver
mask looks like"; its reconstruction error on a degraded mask later becomes the
quality signal.

Usage
-----
    python train.py --config configs/vae.yaml            # full run
    python train.py --config configs/vae.yaml --limit 500  # quick smoke test
    python train.py --config configs/vae.yaml --resume outputs/checkpoints/vae_last.pt

The full run is meant to be launched by the user on the GPU; this script only
needs to survive a short ``--limit`` smoke test unattended (loss should drop, a
training-curve plot and at least one reconstruction preview should appear).
"""
from __future__ import annotations

import argparse
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from data import load_config
from data.raos_dataset import RAOSMaskSlices
from losses.vae_losses import vae_loss
from models.vae import ConvVAE

logger = logging.getLogger("train")

PROJECT_ROOT = Path(__file__).resolve().parent
N_PREVIEW = 6          # val samples shown in a reconstruction preview
_DICE_EPS = 1.0        # smoothing for the hard reconstruction Dice


# --------------------------------------------------------------------------
# setup helpers
# --------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed python / numpy / torch (incl. CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda":
        logger.info("device: cuda (%s)", torch.cuda.get_device_name(0))
    else:
        logger.info("device: cpu")
    return device


def collate_masks(batch: list[dict[str, Any]]) -> Tensor:
    """Stack the ``mask`` field of dataset samples into a (B, 1, H, W) float32 tensor."""
    masks = np.stack([np.asarray(s["mask"], dtype=np.float32) for s in batch])
    return torch.from_numpy(masks).unsqueeze(1)


def _limit_dataset(ds: RAOSMaskSlices, n: int | None) -> RAOSMaskSlices:
    """Truncate a dataset to its first ``n`` slices (for quick smoke tests)."""
    if n is not None and n < len(ds.samples):
        ds.samples = ds.samples[:n]
    return ds


def make_loaders(
    config: dict[str, Any], limit: int | None
) -> tuple[DataLoader, DataLoader]:
    tcfg = config["train"]
    # Volume LRU cache size. With random shuffling every slice access can force a
    # full-volume NIfTI read; a cache large enough to hold the working set of
    # patient volumes turns epochs 2+ into pure compute. Tuned per run in config.
    cache_size = int(tcfg.get("cache_size", 4))
    train_ds = _limit_dataset(
        RAOSMaskSlices.from_config(config, split="train", group="train", cache_size=cache_size),
        limit,
    )
    val_ds = _limit_dataset(
        RAOSMaskSlices.from_config(config, split="val", group="train", cache_size=cache_size),
        limit,
    )
    common = dict(
        batch_size=int(tcfg["batch_size"]),
        num_workers=int(tcfg["num_workers"]),
        collate_fn=collate_masks,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    logger.info("train slices: %d | val slices: %d", len(train_ds), len(val_ds))
    return train_loader, val_loader


def build_model(config: dict[str, Any]) -> ConvVAE:
    vcfg = config["vae"]
    return ConvVAE(
        latent_dim=int(vcfg["latent_dim"]),
        base_channels=int(vcfg["base_channels"]),
        input_size=tuple(vcfg["input_size"]),
    )


# --------------------------------------------------------------------------
# training / evaluation
# --------------------------------------------------------------------------
def beta_at(epoch: int, target: float, warmup_epochs: int) -> float:
    """Linear KL warmup: ramp beta from 0 to ``target`` over ``warmup_epochs``."""
    if warmup_epochs and warmup_epochs > 0:
        return target * min(1.0, (epoch + 1) / warmup_epochs)
    return target


def hard_dice(probs: Tensor, target: Tensor) -> Tensor:
    """Mean per-sample Dice of the thresholded reconstruction vs the target."""
    pred = (probs > 0.5).float()
    dims = tuple(range(1, pred.dim()))
    inter = (pred * target).sum(dim=dims)
    denom = pred.sum(dim=dims) + target.sum(dim=dims)
    return ((2.0 * inter + _DICE_EPS) / (denom + _DICE_EPS)).mean()


def sharpness_penalty(probs: Tensor) -> Tensor:
    """Mean ``min(p, 1-p)`` over all pixels: 0 = crisp (probs at 0/1), 0.5 = fully
    blurred (probs at 0.5). A blurry VAE collapses every mask toward the mean and
    would destroy the Phase-6 good-vs-bad signal, so this is tracked explicitly and
    should drop clearly by the end of a real run."""
    return torch.minimum(probs, 1.0 - probs).mean()


def improved(current: float, best: float, min_delta: float) -> bool:
    """True iff ``current`` beats ``best`` by more than ``min_delta`` (so that
    microscopic wiggles don't count as improvement for early stopping)."""
    return current > best + min_delta


def run_epoch(
    model: ConvVAE,
    loader: DataLoader,
    device: torch.device,
    *,
    beta: float,
    dice_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """One pass over ``loader``. Training if ``optimizer`` is given, else eval.

    Returns batch-size-weighted mean of every loss term; in eval it also reports
    ``recon_dice`` (hard Dice of reconstruct(x) > 0.5 vs x) — the Phase-6 signal.
    """
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = defaultdict(float)
    dice_sum = 0.0
    blur_sum = 0.0
    n = 0
    torch.set_grad_enabled(training)
    for x in loader:
        x = x.to(device, non_blocking=True)
        logits, mu, logvar = model(x)
        terms = vae_loss(logits, x, mu, logvar, beta=beta, dice_weight=dice_weight)
        if training:
            optimizer.zero_grad(set_to_none=True)
            terms["total"].backward()
            optimizer.step()
        bs = x.size(0)
        n += bs
        for key, value in terms.items():
            totals[key] += value.item() * bs
        if not training:
            probs = torch.sigmoid(logits)
            dice_sum += hard_dice(probs, x).item() * bs
            blur_sum += sharpness_penalty(probs).item() * bs
    torch.set_grad_enabled(True)
    metrics = {key: totals[key] / n for key in totals}
    if not training:
        metrics["recon_dice"] = dice_sum / n
        metrics["blur"] = blur_sum / n
    return metrics


# --------------------------------------------------------------------------
# checkpoints & artifacts
# --------------------------------------------------------------------------
def save_checkpoint(
    path: Path,
    model: ConvVAE,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def plot_curves(history: dict[str, list[dict[str, float]]], out_path: Path, title: str) -> None:
    """Plot train/val loss terms plus val reconstruction Dice and sharpness."""
    epochs = range(1, len(history["train"]) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panels = [("total", "total loss"), ("recon", "recon loss"), ("kl", "KL")]
    for ax, (key, panel_title) in zip(axes.flat, panels):
        ax.plot(epochs, [m[key] for m in history["train"]], label="train", marker="o", ms=3)
        ax.plot(epochs, [m[key] for m in history["val"]], label="val", marker="o", ms=3)
        ax.set_title(panel_title)
        ax.set_xlabel("epoch")
        ax.legend()

    ax_dice = axes.flat[3]
    ax_dice.plot(epochs, [m["recon_dice"] for m in history["val"]], color="C2", marker="o", ms=3)
    ax_dice.set_title("val reconstruction Dice")
    ax_dice.set_xlabel("epoch")
    ax_dice.set_ylim(0, 1)

    ax_blur = axes.flat[4]
    ax_blur.plot(epochs, [m["blur"] for m in history["val"]], color="C3", marker="o", ms=3)
    ax_blur.set_title("val blur  mean(min(p,1-p))\n(lower = sharper)")
    ax_blur.set_xlabel("epoch")
    ax_blur.set_ylim(0, 0.5)

    axes.flat[5].axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def save_recon_preview(
    model: ConvVAE, val_ds: RAOSMaskSlices, device: torch.device, out_path: Path
) -> None:
    """Save a [input | reconstruction | difference] grid for a few val masks."""
    n = min(N_PREVIEW, len(val_ds))
    masks = np.stack([np.asarray(val_ds[i]["mask"], dtype=np.float32) for i in range(n)])
    x = torch.from_numpy(masks).unsqueeze(1).to(device)
    model.eval()
    with torch.no_grad():
        recon = model.reconstruct(x).cpu().numpy()[:, 0]
    inp = masks
    diff = np.abs(inp - recon)
    fig, axes = plt.subplots(n, 3, figsize=(6, 2 * n))
    axes = np.atleast_2d(axes)
    col_titles = ["input", "reconstruction", "|difference|"]
    for row in range(n):
        for col, img in enumerate((inp[row], recon[row], diff[row])):
            ax = axes[row, col]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(col_titles[col])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the shape-VAE on clean masks.")
    parser.add_argument("--config", default="configs/vae.yaml", help="path to config YAML")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument("--limit", type=int, default=None, help="cap slices per split (smoke test)")
    parser.add_argument("--latent-dim", type=int, default=None,
                        help="override vae.latent_dim (e.g. 32 vs 64 ablation)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logging.basicConfig(
        level=getattr(logging, config["logging"]["level"], logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.latent_dim is not None:
        config["vae"]["latent_dim"] = int(args.latent_dim)

    tcfg = config["train"]
    lcfg = config["loss"]
    set_seed(int(tcfg["seed"]))
    device = resolve_device(str(tcfg["device"]))

    train_loader, val_loader = make_loaders(config, args.limit)
    val_ds: RAOSMaskSlices = val_loader.dataset  # type: ignore[assignment]

    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(tcfg["lr"]))

    # Per-run tag so the two-latent ablation writes to distinct files.
    tag = f"latent{int(config['vae']['latent_dim'])}"
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
    beta_target = float(lcfg["beta"])
    warmup = int(lcfg["beta_warmup_epochs"])
    preview_every = int(tcfg["recon_preview_every"])
    patience = int(tcfg.get("early_stop_patience", 0))
    min_delta = float(tcfg.get("min_delta", 0.0))
    epochs_since_improve = 0
    history: dict[str, list[dict[str, float]]] = {"train": [], "val": []}
    logger.info("run tag=%s | epochs=%d early_stop_patience=%d min_delta=%.4g",
                tag, epochs, patience, min_delta)

    for epoch in range(start_epoch, epochs):
        beta = beta_at(epoch, beta_target, warmup)
        train_m = run_epoch(
            model, train_loader, device,
            beta=beta, dice_weight=dice_weight, optimizer=optimizer,
        )
        val_m = run_epoch(
            model, val_loader, device, beta=beta, dice_weight=dice_weight,
        )
        history["train"].append(train_m)
        history["val"].append(val_m)
        logger.info(
            "epoch %d/%d beta=%.3f | train total=%.4f recon=%.4f kl=%.4f | "
            "val total=%.4f recon=%.4f kl=%.4f recon_dice=%.4f blur=%.4f",
            epoch + 1, epochs, beta,
            train_m["total"], train_m["recon"], train_m["kl"],
            val_m["total"], val_m["recon"], val_m["kl"],
            val_m["recon_dice"], val_m["blur"],
        )

        save_checkpoint(
            ckpt_dir / f"vae_last_{tag}.pt", model, optimizer, epoch, config,
            {**val_m, "best_recon_dice": best_dice},
        )
        if improved(val_m["recon_dice"], best_dice, min_delta):
            best_dice = val_m["recon_dice"]
            epochs_since_improve = 0
            save_checkpoint(
                ckpt_dir / f"vae_best_{tag}.pt", model, optimizer, epoch, config,
                {**val_m, "best_recon_dice": best_dice},
            )
            logger.info("new best val recon Dice=%.4f -> vae_best_%s.pt", best_dice, tag)
        else:
            epochs_since_improve += 1

        if preview_every and (epoch + 1) % preview_every == 0:
            save_recon_preview(
                model, val_ds, device, out_dir / f"vae_recon_epoch_{epoch + 1}_{tag}.png"
            )

        if patience and epochs_since_improve >= patience:
            logger.info(
                "early stop: val recon Dice did not improve by > %.4g for %d epochs "
                "(best=%.4f at epoch %d)",
                min_delta, patience, best_dice, epoch + 1 - epochs_since_improve,
            )
            break

    if history["train"]:
        plot_curves(history, out_dir / f"vae_training_curves_{tag}.png",
                    title=f"shape-VAE training curves ({tag})")
    logger.info("done (%s). best val recon Dice=%.4f", tag, best_dice)


if __name__ == "__main__":
    main()
