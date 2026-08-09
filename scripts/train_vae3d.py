"""Train the 3D shape-VAE on clean RAOS liver volumes (Phase 7).

The volumetric sibling of ``train.py``. One sample is a whole liver *volume*
(not a slice), so there are only ~222 training examples — the core 3D scarcity.
Everything dimension-agnostic is **imported from the 2D ``train.py``** rather than
re-implemented (set_seed, resolve_device, beta_at, run_epoch, improved,
save_checkpoint, plot_curves); only the genuinely 3D pieces live here (volume
collate, 3D model build, an orthogonal-slice reconstruction preview, and an
augmentation hook placeholder).

Colab-friendly: a checkpoint is written every epoch to ``vae3d_last_*.pt`` and
``--resume`` picks up exactly where a disconnect left off.

Usage
-----
    python scripts/train_vae3d.py --config configs/vae3d.yaml
    python scripts/train_vae3d.py --config configs/vae3d.yaml --limit 20     # smoke
    python scripts/train_vae3d.py --config configs/vae3d.yaml --resume outputs/checkpoints/vae3d_last_latent64.pt

The full 64^3 run is launched by the user on Colab; this script only needs to
survive a short ``--limit`` smoke test unattended (loss should drop, a curve plot
and at least one orthogonal-slice reconstruction preview should appear).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import load_config  # noqa: E402
from data.raos_volume import RAOSVolumes  # noqa: E402
from models.vae3d import ConvVAE3D  # noqa: E402
# Reuse every dimension-agnostic helper from the 2D trainer — do not duplicate.
from train import (  # noqa: E402
    beta_at,
    improved,
    plot_curves,
    resolve_device,
    run_epoch,
    save_checkpoint,
    set_seed,
)

logger = logging.getLogger("train_vae3d")

N_PREVIEW = 4  # val volumes shown in an orthogonal-slice reconstruction preview


# --------------------------------------------------------------------------
# augmentation hook (placeholder — intentionally a no-op for now)
# --------------------------------------------------------------------------
def augment_volume(mask: np.ndarray) -> np.ndarray:
    """Hook for future volumetric augmentation (flips, small rotations, elastic).

    Deliberately a no-op in Phase 7: with only ~222 volumes, augmentation is the
    obvious next lever against overfitting, so the wiring point is reserved here
    now. To enable later, transform ``mask`` (a (D,H,W) bool/uint8 array) and
    return it; ``_VolumeTransform`` already routes every training sample through
    this function, so no other call site needs to change.
    """
    return mask


class _VolumeTransform:
    """Per-sample transform that routes the mask through :func:`augment_volume`.

    Applied to the *training* dataset only (val/preview stay deterministic). While
    ``augment_volume`` is a no-op this changes nothing; it exists so enabling
    augmentation is a one-function edit.
    """

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        sample["mask"] = augment_volume(sample["mask"])
        return sample


# --------------------------------------------------------------------------
# 3D-specific data / model
# --------------------------------------------------------------------------
def collate_volumes(batch: list[dict[str, Any]]) -> Tensor:
    """Stack the ``mask`` field into a (B, 1, D, H, W) float32 tensor."""
    masks = np.stack([np.asarray(s["mask"], dtype=np.float32) for s in batch])
    return torch.from_numpy(masks).unsqueeze(1)


def _limit_dataset(ds: RAOSVolumes, n: int | None) -> RAOSVolumes:
    """Truncate a dataset to its first ``n`` volumes (for quick smoke tests)."""
    if n is not None and n < len(ds.samples):
        ds.samples = ds.samples[:n]
    return ds


def make_loaders(
    config: dict[str, Any], limit: int | None
) -> tuple[DataLoader, DataLoader]:
    tcfg = config["train"]
    cache_size = int(tcfg.get("cache_size", 8))
    train_ds = _limit_dataset(
        RAOSVolumes.from_config(
            config, split="train", group="train",
            cache_size=cache_size, transform=_VolumeTransform(),
        ),
        limit,
    )
    val_ds = _limit_dataset(
        RAOSVolumes.from_config(config, split="val", group="train", cache_size=cache_size),
        limit,
    )
    common = dict(
        batch_size=int(tcfg["batch_size"]),
        num_workers=int(tcfg["num_workers"]),
        collate_fn=collate_volumes,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    logger.info("train volumes: %d | val volumes: %d", len(train_ds), len(val_ds))
    return train_loader, val_loader


def build_model(config: dict[str, Any]) -> ConvVAE3D:
    vcfg = config["vae3d"]
    return ConvVAE3D(
        latent_dim=int(vcfg["latent_dim"]),
        base_channels=int(vcfg["base_channels"]),
        input_size=tuple(vcfg["input_size"]),
    )


# --------------------------------------------------------------------------
# 3D reconstruction preview: three orthogonal centre slices
# --------------------------------------------------------------------------
def save_recon_preview_3d(
    model: ConvVAE3D, val_ds: RAOSVolumes, device: torch.device, out_path: Path
) -> None:
    """Save orthogonal centre-slice [input|recon] grids for a few val volumes.

    A 3D volume can't be shown directly, so each row shows the axial/coronal/
    sagittal centre slice of the input then of the reconstruction (6 panels/row).
    """
    n = min(N_PREVIEW, len(val_ds))
    masks = np.stack([np.asarray(val_ds[i]["mask"], dtype=np.float32) for i in range(n)])
    x = torch.from_numpy(masks).unsqueeze(1).to(device)
    model.eval()
    with torch.no_grad():
        recon = model.reconstruct(x).cpu().numpy()[:, 0]

    def _ortho(vol: np.ndarray) -> list[np.ndarray]:
        d, h, w = vol.shape
        return [vol[d // 2, :, :], vol[:, h // 2, :], vol[:, :, w // 2]]

    col_titles = ["in-axial", "in-coronal", "in-sagittal",
                  "rec-axial", "rec-coronal", "rec-sagittal"]
    fig, axes = plt.subplots(n, 6, figsize=(12, 2 * n))
    axes = np.atleast_2d(axes)
    for row in range(n):
        panels = _ortho(masks[row]) + _ortho(recon[row])
        for col, img in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(col_titles[col], fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the 3D shape-VAE on clean volumes.")
    parser.add_argument("--config", default="configs/vae3d.yaml", help="path to config YAML")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument("--limit", type=int, default=None, help="cap volumes per split (smoke test)")
    parser.add_argument("--latent-dim", type=int, default=None, help="override vae3d.latent_dim")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logging.basicConfig(
        level=getattr(logging, config["logging"]["level"], logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.latent_dim is not None:
        config["vae3d"]["latent_dim"] = int(args.latent_dim)

    tcfg = config["train"]
    lcfg = config["loss"]
    set_seed(int(tcfg["seed"]))
    device = resolve_device(str(tcfg["device"]))

    train_loader, val_loader = make_loaders(config, args.limit)
    val_ds: RAOSVolumes = val_loader.dataset  # type: ignore[assignment]

    model = build_model(config).to(device)
    logger.info("ConvVAE3D: %d params | ~%.1f MB/forward (b=%d)",
                model.param_count, model.estimate_forward_mb(int(tcfg["batch_size"])),
                int(tcfg["batch_size"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(tcfg["lr"]))

    tag = f"latent{int(config['vae3d']['latent_dim'])}"  # file suffix, e.g. latent64
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
    logger.info("run tag=vae3d_%s | epochs=%d early_stop_patience=%d min_delta=%.4g",
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
        # Overfit watch: with only ~222 volumes, train-vs-val divergence is the key
        # risk, so the train/val total gap is logged every epoch.
        gap = train_m["total"] - val_m["total"]
        logger.info(
            "epoch %d/%d beta=%.3f | train total=%.4f recon=%.4f kl=%.4f | "
            "val total=%.4f recon=%.4f kl=%.4f recon_dice=%.4f blur=%.4f | gap=%.4f",
            epoch + 1, epochs, beta,
            train_m["total"], train_m["recon"], train_m["kl"],
            val_m["total"], val_m["recon"], val_m["kl"],
            val_m["recon_dice"], val_m["blur"], gap,
        )

        save_checkpoint(
            ckpt_dir / f"vae3d_last_{tag}.pt", model, optimizer, epoch, config,
            {**val_m, "best_recon_dice": best_dice},
        )
        if improved(val_m["recon_dice"], best_dice, min_delta):
            best_dice = val_m["recon_dice"]
            epochs_since_improve = 0
            save_checkpoint(
                ckpt_dir / f"vae3d_best_{tag}.pt", model, optimizer, epoch, config,
                {**val_m, "best_recon_dice": best_dice},
            )
            logger.info("new best val recon Dice=%.4f -> vae3d_best_%s.pt", best_dice, tag)
        else:
            epochs_since_improve += 1

        if preview_every and (epoch + 1) % preview_every == 0:
            save_recon_preview_3d(
                model, val_ds, device, out_dir / f"vae3d_recon_epoch_{epoch + 1}_{tag}.png"
            )

        if patience and epochs_since_improve >= patience:
            logger.info(
                "early stop: val recon Dice did not improve by > %.4g for %d epochs "
                "(best=%.4f at epoch %d)",
                min_delta, patience, best_dice, epoch + 1 - epochs_since_improve,
            )
            break

    if history["train"]:
        plot_curves(history, out_dir / f"vae3d_training_curves_{tag}.png",
                    title=f"3D shape-VAE training curves ({tag})")
    logger.info("done (%s). best val recon Dice=%.4f", tag, best_dice)


if __name__ == "__main__":
    main()
