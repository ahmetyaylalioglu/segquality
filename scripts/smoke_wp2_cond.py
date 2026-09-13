"""WP-2 smoke: train the CONDITIONED 3D VQ-VAE for 1-2 epochs and sanity-check the
three things the plan asks for, without touching any existing script.

Reuses verbatim (imports, never edits):
* ``train_vae3d_vq.run_epoch_vq``  — the exact VQ train/eval loop. It only needs a
  model whose ``forward(x) -> (logits, vq_loss, info)``; ``ConvVQVAE3D_cond`` honours
  that contract, so the loop drives the conditioned model unchanged.
* ``train_vae3d.make_loaders``     — the same 222/48 clean-volume loaders.
* ``evaluate_correlation_3d`` + ``evaluate_correlation_3d_vq`` — the SAME degrader /
  eval-set / VQ score functions the canonical evaluator uses, so rho is end-to-end.

Reports
-------
1. **features flow** — cond vector reaches the decoder: forward runs, and after one
   backward ``cond_mlp`` receives non-zero gradient (proves the feature path is live,
   not dead-wired).
2. **usage/perplexity** — val perplexity vs the WP-1 K=512 baseline (~11). Close =>
   feature complements the latent; a big drop => conditioning is starting to eat the
   codebook (the new face of collapse).
3. **rho end-to-end** — Spearman rho of recon_dice vs true Dice on the SAME exam the
   canonical evaluator runs. The value is meaningless at 1-2 epochs; we only prove the
   pipeline computes it for the conditioned model.

Non-destructive: writes nothing to ``outputs/`` (no checkpoints, csv or png).

Usage
-----
    python scripts/smoke_wp2_cond.py --epochs 2 --train-limit 24 --n-per-mask 2
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data import load_config  # noqa: E402
from models.vae3d_vq_cond import ConvVQVAE3D_cond  # noqa: E402
from quality.geom_features import compute_features_batch, resolve_feature_names  # noqa: E402
from train import resolve_device, set_seed  # noqa: E402
from train_vae3d import make_loaders  # noqa: E402
from train_vae3d_vq import run_epoch_vq  # noqa: E402
import evaluate_correlation_3d as ev3d  # noqa: E402
import evaluate_correlation_3d_vq as ev3dvq  # noqa: E402

logger = logging.getLogger("smoke_wp2_cond")


def build_cond_model(config: dict[str, Any], cond_dim: int, feature_names: list[str]) -> ConvVQVAE3D_cond:
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
    """Mean/std of each feature over the train loader (CPU), for the model's
    standardisation buffers — mirrors what the full trainer will do."""
    chunks = [compute_features_batch(x, feature_names, device=torch.device("cpu")) for x in loader]
    feats = torch.cat(chunks, dim=0)
    return feats.mean(0), feats.std(0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WP-2 conditioning smoke.")
    p.add_argument("--config", default="configs/vae3d_vq.yaml")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--train-limit", type=int, default=24, help="cap volumes/split for a fast smoke")
    p.add_argument("--cond-dim", type=int, default=16)
    p.add_argument("--features", default="topology,shape",
                   help="comma-separated feature GROUPS (topology,shape)")
    p.add_argument("--split", default="test")
    p.add_argument("--n-per-mask", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = load_config(args.config)
    config["train"]["epochs"] = int(args.epochs)
    config["train"]["num_workers"] = 0  # Windows subprocess/DLL-safe
    device = resolve_device(str(config["train"]["device"]))
    set_seed(int(config["train"]["seed"]))

    groups = [g for g in args.features.split(",") if g.strip()]
    feature_names = resolve_feature_names(groups)
    num_embeddings = int(config["vae3d_vq"]["num_embeddings"])
    logger.info("WP-2 smoke | cond_dim=%d | feature groups=%s -> %d features %s | K=%d",
                args.cond_dim, groups, len(feature_names), feature_names, num_embeddings)

    train_loader, val_loader = make_loaders(config, args.train_limit)

    model = build_cond_model(config, args.cond_dim, feature_names).to(device)
    logger.info("ConvVQVAE3D_cond: %d params | first dec-stage in_channels=%d",
                model.param_count, model.dec_stages[0][0].in_channels)

    # Standardise features from the train split (what the full run will do).
    if args.cond_dim > 0:
        mean, std = compute_train_feature_stats(train_loader, feature_names)
        model.set_feature_stats(mean, std)
        logger.info("feature stats set | mean=%s | std=%s",
                    [round(float(m), 3) for m in mean], [round(float(s), 3) for s in std])

    dice_weight = float(config["loss"]["dice_weight"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["train"]["lr"]))

    # ---- CHECK 1: features flow (cond MLP receives gradient) -------------
    x0 = next(iter(train_loader)).to(device)
    model.train()
    logits, vq_term, _ = model(x0)
    from losses.vae_losses import vq_loss as _vq_loss
    _vq_loss(logits, x0, vq_term, dice_weight=dice_weight)["total"].backward()
    if args.cond_dim > 0:
        g = model.cond_mlp[0].weight.grad
        cond_flows = g is not None and torch.isfinite(g).all().item() and float(g.abs().sum()) > 0.0
        logger.info("[CHECK 1] cond path live: forward OK, cond_mlp grad sum=%.3e -> %s",
                    float(g.abs().sum()) if g is not None else float("nan"),
                    "FLOWING" if cond_flows else "DEAD")
    else:
        cond_flows = None
        logger.info("[CHECK 1] cond_dim=0 baseline: no cond path (forward OK) -> N/A")
    optimizer.zero_grad(set_to_none=True)

    # ---- train 1-2 epochs ------------------------------------------------
    val_m: dict[str, float] = {}
    for epoch in range(args.epochs):
        train_m = run_epoch_vq(model, train_loader, device, dice_weight=dice_weight, optimizer=optimizer)
        val_m = run_epoch_vq(model, val_loader, device, dice_weight=dice_weight)
        logger.info("epoch %d/%d | train=%.4f val=%.4f | val_dice=%.4f | perplexity=%.2f/%d (%.1f%%)",
                    epoch + 1, args.epochs, train_m["total"], val_m["total"],
                    val_m["recon_dice"], val_m["perplexity"], num_embeddings,
                    100.0 * val_m["perplexity"] / num_embeddings)

    # ---- CHECK 2: usage/perplexity vs WP-1 K=512 baseline (~11) ----------
    perp = val_m["perplexity"]
    logger.info("[CHECK 2] val perplexity=%.2f (WP-1 K=512 baseline ~11). "
                "close => feature complements latent; big drop => conditioning eating codebook.",
                perp)

    # ---- CHECK 3: rho end-to-end (same exam as canonical evaluator) ------
    set_seed(args.seed)
    clean = ev3d.load_clean_volumes(config, args.split, None, args.seed)
    degrader = ev3d.calibrate_degrader(config, clean, args.seed)
    masks, true_dice, _ = ev3d.build_eval_set(clean, degrader, args.n_per_mask, args.seed)
    scores = ev3dvq.score_masks_vq(model, masks, device)
    summary = ev3dvq.summarise_vq(scores, true_dice)
    rho = float(summary["recon_dice"]["global"]["spearman"])
    logger.info("[CHECK 3] rho(recon_dice vs true Dice)=%.4f on n=%d (%s split). "
                "value meaningless at %d epochs; pipeline works = pass.",
                rho, len(true_dice), args.split, args.epochs)

    logger.info("SMOKE SUMMARY | cond_flows=%s | perplexity=%.2f | rho=%.4f | params=%d",
                cond_flows, perp, rho, model.param_count)


if __name__ == "__main__":
    main()
