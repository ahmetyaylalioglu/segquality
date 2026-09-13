"""İŞ 1 smoke: prove the advisor's ``vq_mode='commitment_only'`` variant FREEZES the
codebook, and measure rho against the standard baseline — touching no existing script.

Reuses verbatim: ``train_vae3d.make_loaders`` (same clean-volume loaders),
``train_vae3d_vq.{build_model,run_epoch_vq}`` (the exact VQ loop), and
``evaluate_correlation_3d`` + ``evaluate_correlation_3d_vq`` (same degrader / eval-set
/ VQ scores) so rho is end-to-end. Writes nothing to ``outputs/``.

The freezing claim is proven DIRECTLY, NOT via perplexity (perplexity depends on which
codes are *selected*, so it moves with the encoder even when the codebook is frozen):

  PROOF A — gradient: after one backward, ``quantizer.embedding.weight.grad is None``
            for commitment_only (no gradient path), but is a live tensor for standard.
  PROOF B — bit-identity: snapshot ``embedding.weight`` at init, train N epochs, then
            ``torch.equal(snapshot, weight)`` is True for commitment_only (byte-frozen),
            False for standard (codebook_loss moved it).

Both modes then get an end-to-end rho (recon_dice vs true Dice) so the real question —
does a frozen-random codebook + encoder alignment preserve the QC signal? — is answered.

Usage
-----
    python scripts/smoke_commitment_only.py --epochs 2 --train-limit 24 --n-per-mask 2
"""
from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data import load_config  # noqa: E402
from losses.vae_losses import vq_loss  # noqa: E402
from train import resolve_device, set_seed  # noqa: E402
from train_vae3d import make_loaders  # noqa: E402
from train_vae3d_vq import build_model, run_epoch_vq  # noqa: E402
import evaluate_correlation_3d as ev3d  # noqa: E402
import evaluate_correlation_3d_vq as ev3dvq  # noqa: E402

logger = logging.getLogger("smoke_commitment_only")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="İŞ 1 commitment_only freeze + rho smoke.")
    p.add_argument("--config", default="configs/vae3d_vq.yaml")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--train-limit", type=int, default=24)
    p.add_argument("--split", default="test")
    p.add_argument("--n-per-mask", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_mode(config: dict, mode: str, commitment_cost: float, device: torch.device):
    """Build a fresh model in the requested vq_mode with a fixed seed (so both modes
    start from the SAME random codebook init — a fair frozen-vs-trained comparison)."""
    cfg = copy.deepcopy(config)
    cfg["vae3d_vq"]["vq_mode"] = mode
    cfg["vae3d_vq"]["commitment_cost"] = commitment_cost
    set_seed(int(cfg["train"]["seed"]))
    return build_model(cfg).to(device)


def one_backward_grad_is_none(model, x, device) -> bool:
    """Run a single forward+backward and report whether the codebook got a gradient."""
    model.train()
    model.zero_grad(set_to_none=True)
    logits, vq_term, _ = model(x)
    terms = vq_loss(logits, x, vq_term, dice_weight=float(load_cfg_dice()))
    terms["total"].backward()
    grad = model.quantizer.embedding.weight.grad
    is_none = grad is None or float(grad.abs().sum()) == 0.0
    model.zero_grad(set_to_none=True)
    return is_none


def load_cfg_dice() -> float:
    return 1.0  # dice_weight used everywhere in this repo's VQ runs


def rho_end_to_end(model, config, split, n_per_mask, seed, device) -> float:
    set_seed(seed)
    clean = ev3d.load_clean_volumes(config, split, None, seed)
    degrader = ev3d.calibrate_degrader(config, clean, seed)
    masks, true_dice, _ = ev3d.build_eval_set(clean, degrader, n_per_mask, seed)
    scores = ev3dvq.score_masks_vq(model, masks, device)
    summary = ev3dvq.summarise_vq(scores, true_dice)
    return float(summary["recon_dice"]["global"]["spearman"]), len(true_dice)


def run_mode(name, mode, commitment_cost, config, args, device):
    logger.info("=== MODE: %s (vq_mode=%s, commitment_cost=%.2f) ===", name, mode, commitment_cost)
    model = build_mode(config, mode, commitment_cost, device)
    train_loader, val_loader = make_loaders(config, args.train_limit)

    # Snapshot the codebook BEFORE any training (PROOF B baseline).
    init_codebook = model.quantizer.embedding.weight.detach().clone()

    # PROOF A — gradient path on a real batch.
    x0 = next(iter(train_loader)).to(device)
    grad_none = one_backward_grad_is_none(model, x0, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["train"]["lr"]))
    val_m: dict[str, float] = {}
    for epoch in range(args.epochs):
        train_m = run_epoch_vq(model, train_loader, device, optimizer=optimizer, dice_weight=load_cfg_dice())
        val_m = run_epoch_vq(model, val_loader, device, dice_weight=load_cfg_dice())
        logger.info("  epoch %d/%d | train=%.4f val=%.4f | recon_dice=%.4f | perplexity=%.2f",
                    epoch + 1, args.epochs, train_m["total"], val_m["total"],
                    val_m.get("recon_dice", float("nan")), val_m["perplexity"])

    # PROOF B — bit-identity after training.
    final_codebook = model.quantizer.embedding.weight.detach()
    bit_identical = torch.equal(init_codebook, final_codebook)
    max_delta = float((final_codebook - init_codebook).abs().max())

    rho, n = rho_end_to_end(model, config, args.split, args.n_per_mask, args.seed, device)
    logger.info("  [PROOF A] codebook grad is None/zero after backward = %s", grad_none)
    logger.info("  [PROOF B] codebook bit-identical after %d epochs = %s (max|Δ|=%.3e)",
                args.epochs, bit_identical, max_delta)
    logger.info("  [rho]     recon_dice vs true Dice = %.4f on n=%d (%s)", rho, n, args.split)
    return {"grad_none": grad_none, "bit_identical": bit_identical, "max_delta": max_delta,
            "rho": rho, "perplexity": val_m["perplexity"]}


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
    K = int(config["vae3d_vq"]["num_embeddings"])
    logger.info("İŞ 1 smoke | K=%d | epochs=%d | comparing standard vs commitment_only", K, args.epochs)

    std = run_mode("standard baseline", "standard", 0.25, config, args, device)
    com = run_mode("commitment_only (advisor)", "commitment_only", 1.0, config, args, device)

    logger.info("──────────────────────────── SUMMARY ────────────────────────────")
    logger.info("standard        : grad_none=%s bit_identical=%s max|Δ|=%.3e rho=%.4f perp=%.2f",
                std["grad_none"], std["bit_identical"], std["max_delta"], std["rho"], std["perplexity"])
    logger.info("commitment_only : grad_none=%s bit_identical=%s max|Δ|=%.3e rho=%.4f perp=%.2f",
                com["grad_none"], com["bit_identical"], com["max_delta"], com["rho"], com["perplexity"])
    freeze_proven = com["grad_none"] and com["bit_identical"]
    std_trains = (not std["grad_none"]) and (not std["bit_identical"])
    logger.info("FREEZE PROVEN (commitment_only frozen AND standard trains) = %s",
                freeze_proven and std_trains)
    logger.info("(rho is meaningless at %d epochs; the full run answers preserve/drop/rise.)", args.epochs)


if __name__ == "__main__":
    main()
