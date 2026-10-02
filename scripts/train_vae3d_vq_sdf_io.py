"""Train the SDF-input / SDF-output 3D VQ-VAE (advisor request), reusing the
untouched trainer's data / checkpoint / curve / diagnosis machinery.

Difference from WP-3's ``train_vae3d_vq_sdf.py``
------------------------------------------------
WP-3 fed the encoder a **binary** mask and regressed the SDF target
(``forward(binary)`` vs ``target_sdf``). Here the encoder INPUT is *also* the
normalised truncated SDF, so this is an **SDF autoencoder** with a discrete
bottleneck: the same field ``sdf = normalize(truncate(signed_distance(x), δ))`` is
both fed in and used as the target.

Loss  = ``SmoothL1(pred_sdf, sdf_target)`` + ``vq_term``. Both ``sdf_input`` and
``sdf_target`` are the identical normalised SDF, computed on the fly from the binary
batch (``quality.sdf_paper.batch_input_sdf``; ~EDT/epoch, no new npz, zero touches to
the data pipeline). Val recon Dice uses the binary bridge ``pred_sdf < 0 -> inside``.

Everything else — ``make_loaders``, ``save_checkpoint``, ``plot_curves_vq``, early
stop, the mean-reversion diagnosis (re-derived with the bridge) — is imported
unchanged, so this run shares the identical split / loop / logging with WP-3.

Usage
-----
    # single-epoch smoke
    python scripts/train_vae3d_vq_sdf_io.py --tag sdfio_smoke --epochs 1 --limit 24 --num-workers 0
    # full run (user runs)
    python scripts/train_vae3d_vq_sdf_io.py --tag sdfio --num-workers 0
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data import load_config  # noqa: E402
from data.raos_volume import RAOSVolumes  # noqa: E402
from models.vae3d_vq_sdf_io import ConvVQVAE3D_sdf_io  # noqa: E402
from quality.sdf_paper import DELTA, batch_input_sdf, normalized_to_binary  # noqa: E402
from train import improved, resolve_device, save_checkpoint, set_seed  # noqa: E402
from train_vae3d import make_loaders, save_recon_preview_3d  # noqa: E402
from train_vae3d_vq import log_diag, plot_curves_vq  # noqa: E402
from diagnose_reconstruction import meanreversion_stats, select_indices  # noqa: E402

logger = logging.getLogger("train_vae3d_vq_sdf_io")


def build_sdf_io_model(config: dict[str, Any], delta: float) -> ConvVQVAE3D_sdf_io:
    v = config["vae3d_vq"]
    return ConvVQVAE3D_sdf_io(
        base_channels=int(v["base_channels"]),
        num_embeddings=int(v["num_embeddings"]),
        embedding_dim=int(v["embedding_dim"]),
        commitment_cost=float(v["commitment_cost"]),
        input_size=tuple(v["input_size"]),
        use_skip=bool(v["use_skip"]),
        delta=float(delta),
    )


def _hard_dice_from_binary(pred_bin: torch.Tensor, target: torch.Tensor) -> float:
    dims = tuple(range(1, pred_bin.dim()))
    inter = (pred_bin * target).sum(dim=dims)
    denom = pred_bin.sum(dim=dims) + target.sum(dim=dims)
    return ((2.0 * inter + 1.0) / (denom + 1.0)).mean().item()


def run_epoch_sdf_io(
    model: ConvVQVAE3D_sdf_io,
    loader: DataLoader,
    device: torch.device,
    *,
    delta: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """One pass. Input AND target are the normalised SDF of the binary batch.

    Loss = SmoothL1(pred_sdf, sdf_target) + vq. Eval adds the binary-bridge Dice
    (``pred_sdf < 0``) against the original binary mask ``x``.
    """
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = defaultdict(float)
    perp_sum = used_sum = dice_sum = 0.0
    n = 0
    torch.set_grad_enabled(training)
    for x in loader:
        x = x.to(device, non_blocking=True)
        sdf = batch_input_sdf(x, delta, device=device)   # input == target (SDF autoencoder)
        pred_sdf, vq_term, info = model.forward(sdf)
        recon = F.smooth_l1_loss(pred_sdf, sdf)
        total = recon + vq_term
        if training:
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
        bs = x.size(0)
        n += bs
        totals["total"] += total.item() * bs
        totals["recon"] += recon.item() * bs
        totals["vq"] += vq_term.item() * bs
        perp_sum += float(info["perplexity"].item()) * bs
        used_sum += float(info["codebook_used"].item()) * bs
        if not training:
            pred_bin = (pred_sdf < 0).float()   # binary bridge vs the original mask
            dice_sum += _hard_dice_from_binary(pred_bin, x) * bs
    torch.set_grad_enabled(True)
    metrics = {k: totals[k] / n for k in totals}
    metrics["perplexity"] = perp_sum / n
    metrics["codebook_used"] = used_sum / n
    metrics.setdefault("bce", 0.0)   # keep plot_curves_vq happy (reads bce/dice/blur)
    metrics.setdefault("dice", 0.0)
    if not training:
        metrics["recon_dice"] = dice_sum / n
        metrics["blur"] = 0.0
    return metrics


def run_sdf_io_diag(model: ConvVQVAE3D_sdf_io, val_ds: RAOSVolumes, device: torch.device,
                    k: int, seed: int) -> dict[str, Any]:
    """Mean-reversion + codebook diagnosis using the SDF binary bridge (predict_sdf<0),
    reusing the shared ``meanreversion_stats`` so the ratio stays comparable."""
    n = len(val_ds)
    idx = select_indices(n, min(k, n), seed)
    masks = np.stack([np.asarray(val_ds[i]["mask"], dtype=np.float32) for i in idx])
    x = torch.from_numpy(masks).unsqueeze(1).to(device)
    model.eval()
    with torch.no_grad():
        pred_sdf = model.predict_sdf(x)          # converts binary -> SDF internally
        recon_bin = normalized_to_binary(pred_sdf).cpu().numpy()[:, 0]
        _, _, info = model.forward(model.to_sdf_input(x))
    stats = meanreversion_stats(recon_bin > 0.5, masks > 0.5)
    stats["k"] = len(idx)
    stats["diag_perplexity"] = float(info["perplexity"].item())
    stats["diag_codebook_used"] = int(info["codebook_used"].item())
    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the SDF-input/SDF-output 3D VQ-VAE.")
    p.add_argument("--config", default="configs/vae3d_vq_sdf_io.yaml")
    p.add_argument("--tag", default="vqvae_sdf_io", help="checkpoint/output name tag")
    p.add_argument("--delta", type=float, default=None, help="TSDF half-width δ (else config)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="cap volumes per split (smoke)")
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
    if args.epochs is not None:
        config["train"]["epochs"] = int(args.epochs)
    if args.num_workers is not None:
        config["train"]["num_workers"] = int(args.num_workers)

    io_cfg = config.get("vae3d_vq_sdf_io", {})
    delta = float(args.delta) if args.delta is not None else float(io_cfg.get("delta", DELTA))
    config["vae3d_vq_sdf_io"] = {"delta": delta, "normalize": True, "target": "sdf", "input": "sdf"}

    tcfg = config["train"]
    set_seed(int(tcfg["seed"]))
    device = resolve_device(str(tcfg["device"]))

    train_loader, val_loader = make_loaders(config, args.limit)
    val_ds: RAOSVolumes = val_loader.dataset  # type: ignore[assignment]

    model = build_sdf_io_model(config, delta).to(device)
    num_embeddings = int(config["vae3d_vq"]["num_embeddings"])
    logger.info("ConvVQVAE3D_sdf_io: %d params | codebook=%d x %d | SDF in+out δ=±%.1f | grid=%s",
                model.param_count, num_embeddings, int(config["vae3d_vq"]["embedding_dim"]),
                delta, model.feature_shape)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(tcfg["lr"]))

    tag = args.tag
    out_dir = PROJECT_ROOT / config["output_dir"]
    ckpt_dir = out_dir / "checkpoints"
    epochs = int(tcfg["epochs"])
    preview_every = int(tcfg["recon_preview_every"])
    diag_every = int(tcfg.get("diag_every", 5))
    diag_k = int(tcfg.get("diag_k", 20))
    diag_seed = int(tcfg.get("diag_seed", 42))
    patience = int(tcfg.get("early_stop_patience", 0))
    min_delta = float(tcfg.get("min_delta", 0.0))
    best_dice = -1.0
    best_epoch = 0
    epochs_since_improve = 0
    history: dict[str, list[dict[str, float]]] = {"train": [], "val": []}
    logger.info("run tag=vae3d_%s | epochs=%d diag_every=%d early_stop_patience=%d",
                tag, epochs, diag_every, patience)

    for epoch in range(epochs):
        t0 = time.perf_counter()
        train_m = run_epoch_sdf_io(model, train_loader, device, delta=delta, optimizer=optimizer)
        val_m = run_epoch_sdf_io(model, val_loader, device, delta=delta)
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
            "epoch %d/%d | %.0fs | train=%.4f val=%.4f | recon(SmoothL1)=%.4f vq=%.4f | "
            "val_dice(bridge)=%.4f | perplexity=%.1f/%d(%.0f%%) | best=%.4f @ep%d",
            epoch + 1, epochs, time.perf_counter() - t0,
            train_m["total"], val_m["total"], val_m["recon"], val_m["vq"],
            val_m["recon_dice"], val_m["perplexity"], num_embeddings,
            100.0 * val_m["perplexity"] / num_embeddings, best_dice, best_epoch + 1,
        )

        is_last = epoch + 1 == epochs
        if diag_every and ((epoch + 1) % diag_every == 0 or is_last):
            stats = run_sdf_io_diag(model, val_ds, device, diag_k, diag_seed)
            log_diag(stats, num_embeddings, epoch, epochs)
        if preview_every and (epoch + 1) % preview_every == 0:
            save_recon_preview_3d(model, val_ds, device, out_dir / f"vae3d_{tag}_recon_epoch_{epoch + 1}.png")
        if patience and epochs_since_improve >= patience:
            logger.info("early stop: val bridge Dice did not improve by > %.4g for %d epochs (best=%.4f)",
                        min_delta, patience, best_dice)
            break

    if history["train"]:
        plot_curves_vq(history, out_dir / f"vae3d_{tag}_training_curves.png",
                       title=f"3D SDF-in/out VQ-VAE training curves ({tag})")
    logger.info("done (%s). best val bridge Dice=%.4f @ep%d", tag, best_dice, best_epoch + 1)


if __name__ == "__main__":
    main()
