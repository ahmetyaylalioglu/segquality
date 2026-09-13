"""Train the 3D VQ-VAE on clean RAOS liver volumes (Phase 9).

A near-copy of ``train_vae3d.py`` with three deliberate differences:

1. Model is :class:`models.vae3d_vq.ConvVQVAE3D` (spatial discrete codebook) and the
   objective is :func:`losses.vae_losses.vq_loss` (reconstruction + VQ, **no KL**),
   so this script has its own ``run_epoch_vq`` — the 2D ``run_epoch`` assumes a
   ``(logits, mu, logvar)`` VAE and is left untouched.
2. **Embedded mean-reversion + codebook diagnosis.** Every ``diag_every`` epochs
   (and at the end) it logs the cross-reconstruction ``ratio`` and input-/recon-
   pairwise Dice computed by the SAME :func:`diagnose_reconstruction.meanreversion_stats`
   the standalone script uses (same K, seed and Dice), so the VQ-VAE number is
   directly comparable to the standard VAE's 1.005. It also logs codebook perplexity
   as ``P/num_embeddings = %usage`` — the codebook-collapse early warning.
3. Everything else — the RAOSVolumes data pipeline, the 222/48/47 split, precomputed
   volumes, checkpoint/resume/early-stopping, and the orthogonal-slice preview — is
   reused verbatim from ``train_vae3d.py`` / ``train.py``.

Usage
-----
    python scripts/train_vae3d_vq.py --config configs/vae3d_vq.yaml
    python scripts/train_vae3d_vq.py --config configs/vae3d_vq.yaml --limit 20 --epochs 6  # smoke
    python scripts/train_vae3d_vq.py --config configs/vae3d_vq.yaml --resume outputs/checkpoints/vae3d_vq_last.pt
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# scripts/ on the path so the diagnosis single-source can be imported directly.
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from data import load_config  # noqa: E402
from data.raos_volume import RAOSVolumes  # noqa: E402
from losses.vae_losses import vq_loss  # noqa: E402
from models.vae3d_vq import ConvVQVAE3D  # noqa: E402
# Reuse dimension-agnostic helpers from the 2D trainer (unchanged).
from train import (  # noqa: E402
    hard_dice,
    improved,
    resolve_device,
    save_checkpoint,
    set_seed,
    sharpness_penalty,
)
# Reuse the 3D data loaders + orthogonal preview from the 3D VAE trainer (unchanged).
from train_vae3d import make_loaders, save_recon_preview_3d  # noqa: E402
# Single source of truth for the mean-reversion numbers (identical to the standalone).
from diagnose_reconstruction import meanreversion_stats, select_indices  # noqa: E402

logger = logging.getLogger("train_vae3d_vq")


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
def build_model(config: dict[str, Any]) -> ConvVQVAE3D:
    v = config["vae3d_vq"]
    return ConvVQVAE3D(
        base_channels=int(v["base_channels"]),
        num_embeddings=int(v["num_embeddings"]),
        embedding_dim=int(v["embedding_dim"]),
        commitment_cost=float(v["commitment_cost"]),
        input_size=tuple(v["input_size"]),
        use_skip=bool(v["use_skip"]),
    )


# --------------------------------------------------------------------------
# training / evaluation (VQ objective: recon + vq, no KL)
# --------------------------------------------------------------------------
def run_epoch_vq(
    model: ConvVQVAE3D,
    loader: DataLoader,
    device: torch.device,
    *,
    dice_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """One pass over ``loader``. Training if ``optimizer`` is given, else eval.

    Returns batch-size-weighted means of every loss term plus codebook usage
    (``perplexity``, ``codebook_used``); eval additionally reports ``recon_dice``
    and ``blur`` (the Phase-6 signal, same as the VAE trainer).
    """
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = defaultdict(float)
    perp_sum = 0.0
    used_sum = 0.0
    dice_sum = 0.0
    blur_sum = 0.0
    n = 0
    torch.set_grad_enabled(training)
    for x in loader:
        x = x.to(device, non_blocking=True)
        logits, vq_term, info = model(x)
        terms = vq_loss(logits, x, vq_term, dice_weight=dice_weight)
        if training:
            optimizer.zero_grad(set_to_none=True)
            terms["total"].backward()
            optimizer.step()
        bs = x.size(0)
        n += bs
        for key, value in terms.items():
            totals[key] += value.item() * bs
        perp_sum += float(info["perplexity"].item()) * bs
        used_sum += float(info["codebook_used"].item()) * bs
        if not training:
            probs = torch.sigmoid(logits)
            dice_sum += hard_dice(probs, x).item() * bs
            blur_sum += sharpness_penalty(probs).item() * bs
    torch.set_grad_enabled(True)
    metrics = {key: totals[key] / n for key in totals}
    metrics["perplexity"] = perp_sum / n
    metrics["codebook_used"] = used_sum / n
    if not training:
        metrics["recon_dice"] = dice_sum / n
        metrics["blur"] = blur_sum / n
    return metrics


# --------------------------------------------------------------------------
# embedded mean-reversion + codebook diagnosis (comparable to the standalone)
# --------------------------------------------------------------------------
def run_meanreversion_diag(
    model: ConvVQVAE3D, val_ds: RAOSVolumes, device: torch.device, k: int, seed: int
) -> dict[str, Any]:
    """Compute cross-recon ratio, input-/recon-pairwise Dice and codebook usage on a
    fixed val subset. Uses :func:`select_indices` + :func:`meanreversion_stats` so the
    numbers are apples-to-apples with ``scripts/diagnose_reconstruction.py``."""
    n = len(val_ds)
    idx = select_indices(n, min(k, n), seed)
    masks = np.stack([np.asarray(val_ds[i]["mask"], dtype=np.float32) for i in idx])
    x = torch.from_numpy(masks).unsqueeze(1).to(device)
    model.eval()
    with torch.no_grad():
        logits, _, info = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[:, 0]
    stats = meanreversion_stats(probs > 0.5, masks > 0.5)
    stats["k"] = len(idx)
    stats["diag_perplexity"] = float(info["perplexity"].item())
    stats["diag_codebook_used"] = int(info["codebook_used"].item())
    return stats


def log_diag(stats: dict[str, Any], num_embeddings: int, epoch: int, epochs: int) -> None:
    perp = stats["diag_perplexity"]
    used = stats["diag_codebook_used"]
    logger.info(
        "  [diag ep%d/%d, K=%d] cross-recon ratio=%.4f (diag %.4f vs off %.4f) | "
        "recon-pairwise=%.4f vs input-pairwise=%.4f | perplexity=%.1f/%d=%.1f%% "
        "codes_used=%d/%d",
        epoch + 1, epochs, stats["k"], stats["ratio"], stats["diag_mean"],
        stats["offdiag_mean"], stats["recon_pairwise_dice"], stats["input_pairwise_dice"],
        perp, num_embeddings, 100.0 * perp / num_embeddings, used, num_embeddings,
    )
    # One-line interpretation against the standard-VAE baseline (ratio 1.005).
    verdict = "input-specific (VQ helping)" if stats["ratio"] > 1.15 else "still mean-reverting"
    diversity = ("diversity recovered" if stats["recon_pairwise_dice"]
                 <= stats["input_pairwise_dice"] + 0.05 else "recons still over-similar")
    logger.info("    -> ratio %s; %s vs standard-VAE (ratio 1.005, recon-pw 0.992)",
                verdict, diversity)


# --------------------------------------------------------------------------
# curves
# --------------------------------------------------------------------------
def plot_curves_vq(history: dict[str, list[dict[str, float]]], out_path: Path, title: str) -> None:
    """Train/val total + recon + vq, plus val recon Dice, blur and codebook perplexity."""
    epochs = range(1, len(history["train"]) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (key, panel) in zip(axes.flat, [("total", "total"), ("recon", "recon"), ("vq", "VQ")]):
        ax.plot(epochs, [m[key] for m in history["train"]], label="train", marker="o", ms=3)
        ax.plot(epochs, [m[key] for m in history["val"]], label="val", marker="o", ms=3)
        ax.set_title(panel + " loss"); ax.set_xlabel("epoch"); ax.legend()

    ax_dice = axes.flat[3]
    ax_dice.plot(epochs, [m["recon_dice"] for m in history["val"]], color="C2", marker="o", ms=3)
    ax_dice.set_title("val reconstruction Dice"); ax_dice.set_xlabel("epoch"); ax_dice.set_ylim(0, 1)

    ax_blur = axes.flat[4]
    ax_blur.plot(epochs, [m["blur"] for m in history["val"]], color="C3", marker="o", ms=3)
    ax_blur.set_title("val blur (lower=sharper)"); ax_blur.set_xlabel("epoch"); ax_blur.set_ylim(0, 0.5)

    ax_perp = axes.flat[5]
    ax_perp.plot(epochs, [m["perplexity"] for m in history["val"]], color="C4", marker="o", ms=3)
    ax_perp.set_title("val codebook perplexity\n(higher=more codes used)"); ax_perp.set_xlabel("epoch")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the 3D VQ-VAE on clean volumes.")
    parser.add_argument("--config", default="configs/vae3d_vq.yaml", help="path to config YAML")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument("--limit", type=int, default=None, help="cap volumes per split (smoke test)")
    parser.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    parser.add_argument("--num-embeddings", type=int, default=None,
                        help="override codebook size (vae3d_vq.num_embeddings); None -> use config")
    parser.add_argument("--tag", default="vqvae",
                        help="checkpoint/output name tag; default 'vqvae' keeps existing filenames")
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
    # Backward-compatible codebook-size override (None -> config value untouched).
    # Enables the codebook ablation without editing configs/vae3d_vq.yaml.
    if args.num_embeddings is not None:
        config["vae3d_vq"]["num_embeddings"] = int(args.num_embeddings)
    if args.num_workers is not None:
        config["train"]["num_workers"] = int(args.num_workers)

    tcfg = config["train"]
    lcfg = config["loss"]
    set_seed(int(tcfg["seed"]))
    device = resolve_device(str(tcfg["device"]))

    train_loader, val_loader = make_loaders(config, args.limit)
    val_ds: RAOSVolumes = val_loader.dataset  # type: ignore[assignment]

    model = build_model(config).to(device)
    num_embeddings = int(config["vae3d_vq"]["num_embeddings"])
    logger.info("ConvVQVAE3D: %d params | codebook=%d x %d | use_skip=%s | grid=%s",
                model.param_count, num_embeddings, int(config["vae3d_vq"]["embedding_dim"]),
                config["vae3d_vq"]["use_skip"], model.feature_shape)
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
                       title=f"3D VQ-VAE training curves ({tag})")
    logger.info("done (%s). best val recon Dice=%.4f", tag, best_dice)


if __name__ == "__main__":
    main()
