"""Train a latent DDPM shape prior over the frozen VAR tokenizer (diffusion
workstream, Part 2, Path A).

Pipeline
--------
1. Load the FROZEN VAR tokenizer from ``diffusion3d.var_checkpoint`` (encode +
   decode; never updated here).
2. For each clean liver mask, take the VAR bottleneck's *quantized* latent
   ``f_hat = var.quantize(mask)[0]`` (shape ``(B, embed_dim, s, s, s)``) -- the
   exact tensor the VAR decoder was trained on.
3. One pass over the training set fixes per-channel latent normalisation stats on
   the DDPM (so the epsilon target is unit-scale).
4. Train the latent DDPM with the standard epsilon-prediction MSE; early-stop on
   the (deterministically seeded) validation loss.

The frozen VAR encode/decode and the 222/48/47 data pipeline are reused verbatim
from the existing 3D stack -- nothing here edits a protected file. The product is
a generative prior used AnoDDPM-style in ``scripts/evaluate_localization.py`` to
localize injected errors; that script, not this one, reports the headline
voxel-AUROC/AP against the ``recon_diff/VAR`` 0.846 target.

Usage
-----
    python scripts/train_diffusion3d.py --config configs/diffusion3d.yaml
    # smoke (assistant only):
    python scripts/train_diffusion3d.py --config configs/diffusion3d.yaml \
        --epochs 2 --limit 24 --num-workers 0
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from data import load_config  # noqa: E402
from models.diffusion3d import LatentDDPM3D, build_from_config  # noqa: E402
from train import resolve_device, save_checkpoint, set_seed  # noqa: E402
from train_vae3d import make_loaders, N_PREVIEW  # noqa: E402

import evaluate_correlation_3d_var as evv  # noqa: E402

logger = logging.getLogger("train_diffusion3d")


@torch.no_grad()
def encode_latent(var, masks: torch.Tensor) -> torch.Tensor:
    """Frozen-VAR quantized bottleneck latent ``f_hat`` for a mask batch
    ``(B, 1, D, H, W)`` -> ``(B, C, s, s, s)``."""
    f_hat, _, _ = var.quantize(masks)
    return f_hat


@torch.no_grad()
def compute_latent_stats(var, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean/std of the VAR latent over the whole training set."""
    s = s2 = None
    count = 0
    for masks in loader:
        masks = masks.to(device)
        z = encode_latent(var, masks)                      # (B, C, s, s, s)
        c = z.shape[1]
        flat = z.permute(1, 0, 2, 3, 4).reshape(c, -1)     # (C, B*s^3)
        if s is None:
            s = flat.sum(dim=1)
            s2 = (flat * flat).sum(dim=1)
        else:
            s += flat.sum(dim=1)
            s2 += (flat * flat).sum(dim=1)
        count += flat.shape[1]
    mean = s / count
    var_ = (s2 / count) - mean * mean
    std = var_.clamp_min(1e-12).sqrt()
    return mean, std


def run_epoch(ddpm: LatentDDPM3D, var, loader, device, optimizer=None,
              val_seed: int | None = None) -> float:
    """Average epsilon-prediction MSE over a loader. When ``optimizer`` is None the
    model is scored in eval mode; ``val_seed`` makes the validation noise/timesteps
    identical across epochs so the numbers are comparable for early stopping."""
    train = optimizer is not None
    ddpm.train(train)
    if val_seed is not None:
        torch.manual_seed(val_seed)
    total = 0.0
    n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for masks in loader:
            masks = masks.to(device)
            z = encode_latent(var, masks)                  # frozen VAR, no grad
            loss = ddpm.training_loss(z)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            bs = masks.shape[0]
            total += float(loss.detach()) * bs
            n += bs
    return total / max(n, 1)


@torch.no_grad()
def save_sample_preview(ddpm: LatentDDPM3D, var, device, out_path: Path) -> None:
    """Decode a few unconditional DDPM samples to orthogonal centre slices -- a
    sanity check that the prior generates plausible liver shapes."""
    n = N_PREVIEW
    z = ddpm.sample(n, device)
    vol = torch.sigmoid(var.decode(z))[:, 0].cpu().numpy()

    def _ortho(v: np.ndarray) -> list[np.ndarray]:
        d, h, w = v.shape
        return [v[d // 2], v[:, h // 2], v[:, :, w // 2]]

    fig, axes = plt.subplots(n, 3, figsize=(6, 2 * n))
    axes = np.atleast_2d(axes)
    for row in range(n):
        for col, img in enumerate(_ortho(vol[row])):
            ax = axes[row, col]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
    for col, title in enumerate(["axial", "coronal", "sagittal"]):
        axes[0, col].set_title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_curves(history: dict[str, list[float]], out_path: Path, title: str) -> None:
    epochs = range(1, len(history["train"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, history["train"], label="train", marker="o", ms=3)
    ax.plot(epochs, history["val"], label="val", marker="o", ms=3)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("eps-prediction MSE")
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a latent DDPM shape prior over the frozen VAR tokenizer.")
    p.add_argument("--config", default="configs/diffusion3d.yaml")
    p.add_argument("--var", default=None, help="override diffusion3d.var_checkpoint")
    p.add_argument("--resume", default=None)
    p.add_argument("--limit", type=int, default=None, help="cap volumes per split (smoke test)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None,
                   help="override train.num_workers; set 0 on Windows for subprocess runs")
    p.add_argument("--tag", default="diffusion")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logging.basicConfig(
        level=getattr(logging, config.get("logging", {}).get("level", "INFO"), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.epochs is not None:
        config["train"]["epochs"] = int(args.epochs)
    if args.num_workers is not None:
        config["train"]["num_workers"] = int(args.num_workers)
    if args.var is not None:
        config["diffusion3d"]["var_checkpoint"] = args.var

    tcfg = config["train"]
    set_seed(int(tcfg["seed"]))
    device = resolve_device(str(tcfg["device"]))

    # Frozen VAR tokenizer (encode/decode only).
    var_ckpt = Path(config["diffusion3d"]["var_checkpoint"])
    if not var_ckpt.is_absolute():
        var_ckpt = PROJECT_ROOT / var_ckpt
    var = evv.load_model(var_ckpt, device)
    for param in var.parameters():
        param.requires_grad_(False)
    var.eval()
    latent_channels = var.embedding_dim
    latent_side = var.feature_shape[0]
    logger.info("frozen VAR latent: %d channels x %d^3", latent_channels, latent_side)

    train_loader, val_loader = make_loaders(config, args.limit)

    ddpm = build_from_config(config, latent_channels, latent_side).to(device)
    logger.info("LatentDDPM3D: %d params | T=%d | locate_t=%d",
                sum(p.numel() for p in ddpm.parameters()),
                ddpm.timesteps, int(config["diffusion3d"]["locate_t"]))

    # Fix per-channel latent normalisation from the training set.
    mean, std = compute_latent_stats(var, train_loader, device)
    ddpm.set_latent_stats(mean, std)
    logger.info("latent stats set: mean|%.3f std|%.3f (per-channel)",
                float(mean.mean()), float(std.mean()))

    optimizer = torch.optim.Adam(ddpm.parameters(), lr=float(tcfg["lr"]))

    tag = args.tag
    out_dir = PROJECT_ROOT / config["output_dir"]
    ckpt_dir = out_dir / "checkpoints"

    start_epoch = 0
    best_val = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        ddpm.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val = float(ckpt["metrics"].get("best_val_loss", float("inf")))
        logger.info("resumed from %s at epoch %d", args.resume, start_epoch)

    epochs = int(tcfg["epochs"])
    val_seed = int(tcfg["seed"]) + 1000
    patience = int(tcfg.get("early_stop_patience", 0))
    min_delta = float(tcfg.get("min_delta", 0.0))
    preview_every = int(tcfg.get("preview_every", 0))
    epochs_since_improve = 0
    best_epoch = start_epoch
    history: dict[str, list[float]] = {"train": [], "val": []}

    for epoch in range(start_epoch, epochs):
        t0 = time.perf_counter()
        train_loss = run_epoch(ddpm, var, train_loader, device, optimizer=optimizer)
        val_loss = run_epoch(ddpm, var, val_loader, device, val_seed=val_seed)
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        save_checkpoint(
            ckpt_dir / f"vae3d_{tag}_last.pt", ddpm, optimizer, epoch, config,
            {"train_loss": train_loss, "val_loss": val_loss, "best_val_loss": best_val},
        )
        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_epoch = epoch
            epochs_since_improve = 0
            save_checkpoint(
                ckpt_dir / f"vae3d_{tag}_best.pt", ddpm, optimizer, epoch, config,
                {"train_loss": train_loss, "val_loss": val_loss, "best_val_loss": best_val},
            )
        else:
            epochs_since_improve += 1

        logger.info("epoch %d/%d | %.0fs | train=%.5f val=%.5f | best=%.5f @ep%d",
                    epoch + 1, epochs, time.perf_counter() - t0,
                    train_loss, val_loss, best_val, best_epoch + 1)

        if preview_every and (epoch + 1) % preview_every == 0:
            save_sample_preview(ddpm, var, device,
                                out_dir / f"vae3d_{tag}_samples_epoch_{epoch + 1}.png")

        if patience and epochs_since_improve >= patience:
            logger.info("early stop: val loss did not improve by > %.4g for %d epochs "
                        "(best=%.5f at epoch %d)", min_delta, patience, best_val, best_epoch + 1)
            break

    if history["train"]:
        plot_curves(history, out_dir / f"vae3d_{tag}_training_curves.png",
                    title=f"latent DDPM training curves ({tag})")
    logger.info("done (%s). best val loss=%.5f @ep%d", tag, best_val, best_epoch + 1)


if __name__ == "__main__":
    main()
