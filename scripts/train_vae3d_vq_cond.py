"""Train the geometric-feature-CONDITIONED 3D VQ-VAE (WP-2), reusing the untouched
``train_vae3d_vq.py`` machinery so the three WP-2 runs are byte-identical apart from
the conditioning.

Why a separate trainer (nothing existing is edited)
---------------------------------------------------
``train_vae3d_vq.build_model`` hard-wires the plain :class:`ConvVQVAE3D`. This script
builds :class:`ConvVQVAE3D_cond` instead and adds the two things conditioning needs —
train-split feature standardisation and embedding the conditioning spec in the
checkpoint — then delegates EVERY per-epoch step (``run_epoch_vq``, ``save_checkpoint``,
``plot_curves_vq``, the mean-reversion/codebook ``diag``, ``improved`` early stop) to
the imported, unchanged trainer. So the loop, loss, split, and diagnostics are
identical across the three runs.

The three WP-2 runs are all this one script, differing only in flags:
  (a) topology-only : --cond-dim 16 --feature-groups topology
  (b) full set      : --cond-dim 16 --feature-groups topology,shape
  (c) baseline      : --cond-dim 0            (no-op -> plain VQ-VAE, SAME trainer)

Because ``cond_dim=0`` is an exact no-op, run (c) is a fair baseline measured on the
identical trainer/exam — so any Δrho is attributable to conditioning, not to a
script difference (the old WP-1 rho=0.86 came from a different script).

Conditioning spec travels in the checkpoint
--------------------------------------------
``config["vae3d_vq_cond"] = {cond_dim, feature_groups, feature_names, cond_hidden}`` is
added before ``save_checkpoint`` (which persists ``config`` verbatim), and the
standardisation buffers ``feat_mean``/``feat_std`` live in ``state_dict`` — so the
cond-aware evaluator can rebuild and restore the exact model with no side channel.

Usage
-----
    # single-epoch chain smoke (checkpoint + eval round-trip)
    python scripts/train_vae3d_vq_cond.py --cond-dim 16 --feature-groups topology,shape \
        --tag wp2_smoke --epochs 1 --limit 24 --num-workers 0
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data import load_config  # noqa: E402
from data.raos_volume import RAOSVolumes  # noqa: E402
from models.vae3d_vq_cond import ConvVQVAE3D_cond  # noqa: E402
from quality.geom_features import compute_features_batch, resolve_feature_names  # noqa: E402
from train import improved, resolve_device, save_checkpoint, set_seed  # noqa: E402
from train_vae3d import make_loaders, save_recon_preview_3d  # noqa: E402
# Reuse the exact VQ loop, curves and codebook/mean-reversion diagnosis, unchanged.
from train_vae3d_vq import (  # noqa: E402
    log_diag,
    plot_curves_vq,
    run_epoch_vq,
    run_meanreversion_diag,
)

logger = logging.getLogger("train_vae3d_vq_cond")


def build_cond_model(
    config: dict[str, Any], cond_dim: int, feature_names: list[str]
) -> ConvVQVAE3D_cond:
    v = config["vae3d_vq"]
    return ConvVQVAE3D_cond(
        base_channels=int(v["base_channels"]),
        num_embeddings=int(v["num_embeddings"]),
        embedding_dim=int(v["embedding_dim"]),
        commitment_cost=float(v["commitment_cost"]),
        input_size=tuple(v["input_size"]),
        use_skip=bool(v["use_skip"]),
        cond_dim=cond_dim,
        feature_names=feature_names,
    )


def compute_train_feature_stats(loader, feature_names: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean/std of each feature over the whole train loader (CPU)."""
    chunks = [compute_features_batch(x, feature_names, device=torch.device("cpu")) for x in loader]
    feats = torch.cat(chunks, dim=0)
    return feats.mean(0), feats.std(0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the geometric-feature-conditioned 3D VQ-VAE.")
    p.add_argument("--config", default="configs/vae3d_vq.yaml")
    p.add_argument("--tag", default="vqvae_cond",
                   help="checkpoint/output name tag; keeps runs from clobbering each other")
    p.add_argument("--cond-dim", type=int, default=16,
                   help="conditioning embedding width; 0 -> no-op baseline (plain VQ-VAE)")
    p.add_argument("--feature-groups", default="topology,shape",
                   help="comma-separated feature GROUPS (topology,shape); ignored when cond-dim=0")
    p.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    p.add_argument("--limit", type=int, default=None, help="cap volumes per split (smoke)")
    p.add_argument("--num-workers", type=int, default=None,
                   help="override train.num_workers; set 0 on Windows subprocess runs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    config = load_config(args.config)
    logging.basicConfig(
        level=getattr(logging, config["logging"]["level"], logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.epochs is not None:
        config["train"]["epochs"] = int(args.epochs)
    if args.num_workers is not None:
        config["train"]["num_workers"] = int(args.num_workers)

    tcfg = config["train"]
    lcfg = config["loss"]
    set_seed(int(tcfg["seed"]))
    device = resolve_device(str(tcfg["device"]))

    # Resolve conditioning spec. cond_dim=0 => empty feature set => exact no-op.
    cond_dim = int(args.cond_dim)
    groups = [g for g in args.feature_groups.split(",") if g.strip()] if cond_dim > 0 else []
    feature_names = resolve_feature_names(groups) if cond_dim > 0 else []
    # Embed the spec so the checkpoint alone fully specifies the model.
    config["vae3d_vq_cond"] = {
        "cond_dim": cond_dim,
        "feature_groups": groups,
        "feature_names": feature_names,
        "cond_hidden": 32,
    }

    train_loader, val_loader = make_loaders(config, args.limit)
    val_ds: RAOSVolumes = val_loader.dataset  # type: ignore[assignment]

    model = build_cond_model(config, cond_dim, feature_names).to(device)
    num_embeddings = int(config["vae3d_vq"]["num_embeddings"])
    logger.info("ConvVQVAE3D_cond: %d params | codebook=%d x %d | cond_dim=%d groups=%s (%d feats) | grid=%s",
                model.param_count, num_embeddings, int(config["vae3d_vq"]["embedding_dim"]),
                cond_dim, groups, len(feature_names), model.feature_shape)

    # Standardise features from the train split (skip for the no-op baseline).
    if cond_dim > 0:
        mean, std = compute_train_feature_stats(train_loader, feature_names)
        model.set_feature_stats(mean, std)
        logger.info("feature stats (train split) | names=%s | mean=%s | std=%s",
                    feature_names, [round(float(m), 3) for m in mean],
                    [round(float(s), 3) for s in std])

    optimizer = torch.optim.Adam(model.parameters(), lr=float(tcfg["lr"]))
    tag = args.tag
    out_dir = PROJECT_ROOT / config["output_dir"]
    ckpt_dir = out_dir / "checkpoints"

    epochs = int(tcfg["epochs"])
    dice_weight = float(lcfg["dice_weight"])
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
    logger.info("run tag=vae3d_%s | epochs=%d diag_every=%d diag_k=%d early_stop_patience=%d",
                tag, epochs, diag_every, diag_k, patience)

    for epoch in range(epochs):
        t0 = time.perf_counter()
        train_m = run_epoch_vq(model, train_loader, device, dice_weight=dice_weight, optimizer=optimizer)
        val_m = run_epoch_vq(model, val_loader, device, dice_weight=dice_weight)
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
            logger.info("early stop: val recon Dice did not improve by > %.4g for %d epochs (best=%.4f)",
                        min_delta, patience, best_dice)
            break

    if history["train"]:
        plot_curves_vq(history, out_dir / f"vae3d_{tag}_training_curves.png",
                       title=f"3D VQ-VAE conditioned training curves ({tag})")
    logger.info("done (%s). best val recon Dice=%.4f @ep%d", tag, best_dice, best_epoch + 1)


if __name__ == "__main__":
    main()
