"""WP-3 smoke: train the SDF-target VQ-VAE 1-2 epochs and check the four things the
plan asks for, touching no existing script.

Reuses verbatim: ``train_vae3d.make_loaders`` (same clean-volume loaders),
``train_vae3d_vq_sdf.{build_sdf_model,run_epoch_sdf}`` (the SDF loop), and
``evaluate_correlation_3d`` + ``evaluate_correlation_3d_vq`` (same degrader / eval-set
/ VQ scores) so rho is end-to-end. Writes nothing to ``outputs/``.

Reports
-------
1. SDF flows       — predict_sdf is raw continuous (has negatives), SmoothL1 drops,
                     encoder gradient is live.
2. bridge EXACT    — on a REAL training batch, (clamp'd target SDF < 0) == binary,
                     i.e. the sign-preserving TSDF bridge is lossless in practice.
3. perplexity      — val codebook perplexity vs the WP-1/2 K=512 range (early-epoch
                     ~5 is fine; only a collapse to ~1 would be alarming).
4. rho end-to-end  — SDF model -> bridge -> binary -> recon_dice -> rho vs true Dice.

Usage
-----
    python scripts/smoke_wp3_sdf.py --epochs 2 --train-limit 24 --n-per-mask 2
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data import load_config  # noqa: E402
from quality.sdf import TRUNCATION, batch_target_sdf, sdf_to_binary  # noqa: E402
from train import resolve_device, set_seed  # noqa: E402
from train_vae3d import make_loaders  # noqa: E402
from train_vae3d_vq_sdf import build_sdf_model, run_epoch_sdf  # noqa: E402
import evaluate_correlation_3d as ev3d  # noqa: E402
import evaluate_correlation_3d_vq as ev3dvq  # noqa: E402

logger = logging.getLogger("smoke_wp3_sdf")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WP-3 SDF-VQ-VAE smoke.")
    p.add_argument("--config", default="configs/vae3d_vq.yaml")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--train-limit", type=int, default=24)
    p.add_argument("--truncation", type=float, default=TRUNCATION)
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
    config["train"]["num_workers"] = 0
    device = resolve_device(str(config["train"]["device"]))
    set_seed(int(config["train"]["seed"]))
    num_embeddings = int(config["vae3d_vq"]["num_embeddings"])
    trunc = float(args.truncation)
    logger.info("WP-3 smoke | K=%d | TSDF=±%.1f", num_embeddings, trunc)

    train_loader, val_loader = make_loaders(config, args.train_limit)
    model = build_sdf_model(config).to(device)
    logger.info("ConvVQVAE3D_sdf: %d params", model.param_count)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["train"]["lr"]))
    dw_note = "SmoothL1(pred_sdf, TSDF) + vq (no BCE/Dice/KL)"

    # ---- CHECK 1 (part a) + CHECK 2 on a REAL batch ----------------------
    x0 = next(iter(train_loader)).to(device)
    target0 = batch_target_sdf(x0, trunc, device=device)
    # CHECK 2: clamp'd target SDF bridges back to the exact binary input
    bridge = sdf_to_binary(target0)
    bridge_exact = torch.equal(bridge, x0)
    logger.info("[CHECK 2] clamp'd-SDF bridge on real batch: (TSDF<0)==input EXACT -> %s "
                "(target range [%.2f, %.2f])", bridge_exact, float(target0.min()), float(target0.max()))

    model.train()
    pred0 = model.predict_sdf(x0)
    has_neg = float(pred0.min()) < 0.0
    pred_sdf, vq_term, _ = model.forward(x0)
    full = F.smooth_l1_loss(pred_sdf, target0) + vq_term
    full.backward()
    g = model.enc_stages[0][0].weight.grad
    grad_ok = g is not None and torch.isfinite(g).all().item() and float(g.abs().sum()) > 0.0
    logger.info("[CHECK 1] SDF raw output has negatives=%s | SmoothL1=%.4f | encoder grad live=%s (%s)",
                has_neg, float(F.smooth_l1_loss(pred_sdf, target0)), grad_ok, dw_note)
    optimizer.zero_grad(set_to_none=True)

    # ---- train 1-2 epochs (loss should drop) -----------------------------
    val_m: dict[str, float] = {}
    first_recon = None
    for epoch in range(args.epochs):
        train_m = run_epoch_sdf(model, train_loader, device, truncation=trunc, optimizer=optimizer)
        val_m = run_epoch_sdf(model, val_loader, device, truncation=trunc)
        if first_recon is None:
            first_recon = train_m["recon"]
        logger.info("epoch %d/%d | train=%.4f val=%.4f | recon(SmoothL1)=%.4f vq=%.4f | "
                    "val_dice(bridge)=%.4f | perplexity=%.2f/%d (%.1f%%)",
                    epoch + 1, args.epochs, train_m["total"], val_m["total"],
                    train_m["recon"], train_m["vq"], val_m["recon_dice"],
                    val_m["perplexity"], num_embeddings, 100.0 * val_m["perplexity"] / num_embeddings)

    # ---- CHECK 3: perplexity vs WP-1/2 K=512 range -----------------------
    perp = val_m["perplexity"]
    logger.info("[CHECK 3] val perplexity=%.2f (WP-1/2 K=512 converged ~11; early-epoch ~5 fine; "
                "collapse to ~1 would be alarming)", perp)

    # ---- CHECK 4: rho end-to-end (same exam as canonical evaluator) ------
    set_seed(args.seed)
    clean = ev3d.load_clean_volumes(config, args.split, None, args.seed)
    degrader = ev3d.calibrate_degrader(config, clean, args.seed)
    masks, true_dice, _ = ev3d.build_eval_set(clean, degrader, args.n_per_mask, args.seed)
    scores = ev3dvq.score_masks_vq(model, masks, device)  # uses model.reconstruct = bridge
    summary = ev3dvq.summarise_vq(scores, true_dice)
    rho = float(summary["recon_dice"]["global"]["spearman"])
    logger.info("[CHECK 4] rho(recon_dice via bridge vs true Dice)=%.4f on n=%d (%s). "
                "meaningless at %d epochs; chain works = pass.", rho, len(true_dice), args.split, args.epochs)

    logger.info("SMOKE SUMMARY | bridge_exact=%s | sdf_has_neg=%s | grad_ok=%s | perplexity=%.2f | rho=%.4f",
                bridge_exact, has_neg, grad_ok, perp, rho)


if __name__ == "__main__":
    main()
