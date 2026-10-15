"""Faz 4: train the cascade stage-2 refiner on top of the frozen VAR FP/FN stage-1.

Advisor's 2-VAE drawing: stage-1 (frozen Faz-3 ``vae3d_g_var_best.pt``) turns ``[image,
broken]`` into ``[FP, FN]`` and a ``corrected`` mask; stage-2 (trained here) takes
``[corrected, FP, FN]`` (3ch, NO image) and predicts the single REFERENCE mask. Only the
stage-2 refiner's parameters are optimized; stage-1 stays frozen.

The data / degradation / split are reused verbatim from ``train_imgcond_fpfn.make_loaders``:
each batch yields ``x = [image, broken]`` and ``target = [FP_tgt, FN_tgt]``, from which the
clean reference mask is reconstructed as ``correct_mask(broken, FP_tgt, FN_tgt)`` — the
stage-2 supervision target. Loss = ``recon(BCE+Dice)(refined, clean) + VQ(stage-2)``.

The logged ``corrected_dice`` is the stage-2 refined-mask Dice vs clean (checkpoint metric);
``base_dice`` is the stage-1 corrected-mask Dice vs clean — the accuracy the 2nd VAE must beat
(advisor's question). The checkpoint stores ONLY the stage-2 refiner under ``model_state`` so
the cascade eval rebuilds it with the same frozen stage-1.

Usage
-----
    python scripts/train_cascade.py --config configs/vae3d_cascade.yaml --tag h_cascade
    python scripts/train_cascade.py --config configs/vae3d_cascade.yaml --tag h_cascade --epochs 2 --limit 24 --num-workers 0  # smoke
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data import load_config  # noqa: E402
from data.fpfn_targets import correct_mask  # noqa: E402
from losses.vae_losses import reconstruction_loss  # noqa: E402
from models.vae3d_cascade import CascadeFPFN, VARVQVAE3D_Refiner  # noqa: E402
from models.vae3d_var_fpfn import VARVQVAE3D_ImgCond_FPFN  # noqa: E402
from train import improved, resolve_device, save_checkpoint, set_seed  # noqa: E402
from train_imgcond import _hard_dice  # noqa: E402
from train_imgcond_fpfn import make_loaders  # noqa: E402
from train_vae3d_vq import plot_curves_vq  # noqa: E402

logger = logging.getLogger("train_cascade")


def _var_kwargs(v: dict[str, Any]) -> dict[str, Any]:
    return dict(
        base_channels=int(v["base_channels"]),
        num_embeddings=int(v["num_embeddings"]),
        embedding_dim=int(v["embedding_dim"]),
        input_size=tuple(v["input_size"]),
        v_patch_nums=tuple(v["v_patch_nums"]),
        using_znorm=bool(v["using_znorm"]),
        beta=float(v["beta"]),
        quant_resi=float(v["quant_resi"]),
    )


def build_cascade(config: dict[str, Any], device: torch.device) -> CascadeFPFN:
    """Load the frozen stage-1 FP/FN model and build a fresh stage-2 refiner."""
    v = config["vae3d_var"]
    ccfg = config["cascade"]
    stage1 = VARVQVAE3D_ImgCond_FPFN(**_var_kwargs(v))
    ckpt_path = ccfg["stage1_checkpoint"]
    if not Path(ckpt_path).is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    stage1.load_state_dict(ckpt["model_state"])
    logger.info("stage-1 (frozen) loaded from %s (epoch=%s)", Path(ckpt_path).name, ckpt.get("epoch"))
    refiner = VARVQVAE3D_Refiner(in_channels=int(ccfg.get("refiner_in_channels", 3)), **_var_kwargs(v))
    return CascadeFPFN(stage1, refiner).to(device)


def run_epoch_cascade(
    model: CascadeFPFN, loader, device: torch.device,
    *, dice_weight: float, optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """One pass: frozen stage-1 -> [corrected, FP, FN] -> stage-2 refiner -> reference mask.

    Target = clean mask = ``correct_mask(broken, FP_tgt, FN_tgt)``. Loss = recon(BCE+Dice) on
    the refined logits + stage-2 VQ. Val also reports the refined-mask Dice (``recon_dice``) and
    the stage-1 corrected-mask Dice (``base_dice``) vs clean so the accuracy gain is visible.
    """
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = defaultdict(float)
    perp_sum = used_sum = dice_sum = base_sum = 0.0
    n = 0
    torch.set_grad_enabled(training)
    for x, target in loader:
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        broken = x[:, 1:2]
        clean = correct_mask(broken, target[:, 0:1], target[:, 1:2])   # (B,1,...) {0,1}

        corrected, fp, fn = model.stage1_products(x)                   # frozen, no grad
        refiner_in = model.refiner_input(corrected, fp, fn)
        logits, vq_term, info = model.forward(refiner_in)
        recon, bce, dice = reconstruction_loss(logits, clean, dice_weight)
        total = recon + vq_term
        if training:
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()

        bs = x.size(0)
        n += bs
        totals["total"] += total.item() * bs
        totals["recon"] += recon.item() * bs
        totals["bce"] += bce.item() * bs
        totals["dice"] += dice.item() * bs
        totals["vq"] += vq_term.item() * bs
        perp_sum += float(info["perplexity"].item()) * bs
        used_sum += float(info["codebook_used"].item()) * bs
        if not training:
            refined = (torch.sigmoid(logits) > 0.5).float()
            dice_sum += _hard_dice(refined, clean) * bs
            base_sum += _hard_dice(corrected, clean) * bs
    torch.set_grad_enabled(True)
    metrics = {k: totals[k] / n for k in totals}
    metrics["perplexity"] = perp_sum / n
    metrics["codebook_used"] = used_sum / n
    if not training:
        metrics["recon_dice"] = dice_sum / n
        metrics["base_dice"] = base_sum / n
        metrics["blur"] = 0.0
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the cascade stage-2 refiner on the frozen VAR FP/FN (dir H, Faz 4).")
    p.add_argument("--config", default="configs/vae3d_cascade.yaml")
    p.add_argument("--primitive-set", choices=("mixed", "partial"), default="mixed")
    p.add_argument("--tag", default="h_cascade")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
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
    config.setdefault("data", {})["data_source"] = config.get("data", {}).get("data_source", "precomputed")
    if args.epochs is not None:
        config["train"]["epochs"] = int(args.epochs)
    if args.num_workers is not None:
        config["train"]["num_workers"] = int(args.num_workers)

    tcfg = config["train"]
    set_seed(int(tcfg["seed"]))
    device = resolve_device(str(tcfg["device"]))
    dice_weight = float(config.get("loss", {}).get("dice_weight", 1.0))

    train_loader, val_loader = make_loaders(
        config, args.limit, seed=int(tcfg["seed"]), primitive_set=args.primitive_set)

    model = build_cascade(config, device)
    num_embeddings = int(config["vae3d_var"]["num_embeddings"])
    logger.info(
        "%s: stage-2 %d trainable params | refiner in=%d out=%d | codebook=%d | scales=%s | primitive_set=%s",
        type(model).__name__, model.param_count, model.in_channels, model.out_channels,
        num_embeddings, model.v_patch_nums, args.primitive_set,
    )
    optimizer = torch.optim.Adam(model.refiner.parameters(), lr=float(tcfg["lr"]))

    tag = args.tag
    out_dir = PROJECT_ROOT / config["output_dir"]
    ckpt_dir = out_dir / "checkpoints"
    epochs = int(tcfg["epochs"])
    patience = int(tcfg.get("early_stop_patience", 0))
    min_delta = float(tcfg.get("min_delta", 0.0))
    best_dice = -1.0
    best_epoch = 0
    epochs_since_improve = 0
    history: dict[str, list[dict[str, float]]] = {"train": [], "val": []}
    logger.info("run tag=vae3d_%s | epochs=%d early_stop_patience=%d", tag, epochs, patience)

    for epoch in range(epochs):
        t0 = time.perf_counter()
        train_m = run_epoch_cascade(model, train_loader, device, dice_weight=dice_weight, optimizer=optimizer)
        val_m = run_epoch_cascade(model, val_loader, device, dice_weight=dice_weight)
        history["train"].append(train_m)
        history["val"].append(val_m)

        save_checkpoint(ckpt_dir / f"vae3d_{tag}_last.pt", model.refiner, optimizer, epoch, config,
                        {**val_m, "best_recon_dice": best_dice})
        if improved(val_m["recon_dice"], best_dice, min_delta):
            best_dice = val_m["recon_dice"]
            best_epoch = epoch
            epochs_since_improve = 0
            save_checkpoint(ckpt_dir / f"vae3d_{tag}_best.pt", model.refiner, optimizer, epoch, config,
                            {**val_m, "best_recon_dice": best_dice})
        else:
            epochs_since_improve += 1

        logger.info(
            "epoch %d/%d | %.0fs | train=%.4f val=%.4f | bce=%.4f dice=%.4f vq=%.4f | "
            "refined_dice=%.4f (stage1_base=%.4f, gain=%+.4f) | perplexity=%.1f/%d(%.0f%%) | best=%.4f @ep%d",
            epoch + 1, epochs, time.perf_counter() - t0,
            train_m["total"], val_m["total"], val_m["bce"], val_m["dice"], val_m["vq"],
            val_m["recon_dice"], val_m["base_dice"], val_m["recon_dice"] - val_m["base_dice"],
            val_m["perplexity"], num_embeddings,
            100.0 * val_m["perplexity"] / num_embeddings, best_dice, best_epoch + 1,
        )
        if patience and epochs_since_improve >= patience:
            logger.info("early stop: refined Dice did not improve by > %.4g for %d epochs (best=%.4f)",
                        min_delta, patience, best_dice)
            break

    if history["train"]:
        plot_curves_vq(history, out_dir / f"vae3d_{tag}_training_curves.png",
                       title=f"Cascade stage-2 refiner on VAR FP/FN ({tag})")
    logger.info("done (%s). best refined Dice=%.4f @ep%d", tag, best_dice, best_epoch + 1)


if __name__ == "__main__":
    main()
