"""Faz 3: train the FP/FN dual-head on the VAR multi-scale backbone (binary head).

A thin sibling of ``train_imgcond_fpfn.py``: the ONLY change is the model built —
:class:`models.vae3d_var_fpfn.VARVQVAE3D_ImgCond_FPFN` (VAR multi-scale residual
quantizer) instead of the plain-VQ ``ConvVQVAE3D_ImgCond_FPFN``. Input, targets, loss
(``L(FP) + L(FN) + VQ``), the corrected-mask Dice metric, and the data/degradation
pipeline are all reused verbatim via ``make_loaders`` and ``run_epoch_fpfn``.

Binary head only (the established best FP/FN head). The checkpoint is written with the
same ``model_state`` key so the FP/FN eval can score it.

Usage
-----
    python scripts/train_var_fpfn.py --config configs/vae3d_var_fpfn.yaml --tag g_var
    python scripts/train_var_fpfn.py --config configs/vae3d_var_fpfn.yaml --tag g_var --epochs 2 --limit 24 --num-workers 0  # smoke
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
from models.vae3d_var_fpfn import VARVQVAE3D_ImgCond_FPFN  # noqa: E402
from train import improved, resolve_device, save_checkpoint, set_seed  # noqa: E402
from train_imgcond_fpfn import make_loaders, run_epoch_fpfn  # noqa: E402
from train_vae3d_vq import plot_curves_vq  # noqa: E402

logger = logging.getLogger("train_var_fpfn")


def build_var_fpfn_model(config: dict[str, Any]) -> VARVQVAE3D_ImgCond_FPFN:
    v = config["vae3d_var"]
    return VARVQVAE3D_ImgCond_FPFN(
        base_channels=int(v["base_channels"]),
        num_embeddings=int(v["num_embeddings"]),
        embedding_dim=int(v["embedding_dim"]),
        input_size=tuple(v["input_size"]),
        v_patch_nums=tuple(v["v_patch_nums"]),
        using_znorm=bool(v["using_znorm"]),
        beta=float(v["beta"]),
        quant_resi=float(v["quant_resi"]),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the FP/FN dual-head on the VAR backbone (dir G, Faz 3).")
    p.add_argument("--config", default="configs/vae3d_var_fpfn.yaml")
    p.add_argument("--primitive-set", choices=("mixed", "partial"), default="mixed")
    p.add_argument("--tag", default="g_var")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--using-znorm", dest="using_znorm", action="store_true", default=None)
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
    if args.using_znorm is not None:
        config["vae3d_var"]["using_znorm"] = bool(args.using_znorm)

    tcfg = config["train"]
    set_seed(int(tcfg["seed"]))
    device = resolve_device(str(tcfg["device"]))

    train_loader, val_loader = make_loaders(
        config, args.limit, seed=int(tcfg["seed"]), primitive_set=args.primitive_set)

    model = build_var_fpfn_model(config).to(device)
    num_embeddings = int(config["vae3d_var"]["num_embeddings"])
    logger.info(
        "%s: %d params | VAR backbone | in=%d out=%d | codebook=%d | scales=%s | primitive_set=%s",
        type(model).__name__, model.param_count, model.in_channels, model.out_channels,
        num_embeddings, model.v_patch_nums, args.primitive_set,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(tcfg["lr"]))

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
        train_m = run_epoch_fpfn(model, train_loader, device, head="binary", delta=0.0, optimizer=optimizer)
        val_m = run_epoch_fpfn(model, val_loader, device, head="binary", delta=0.0)
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
            "epoch %d/%d | %.0fs | train=%.4f val=%.4f | fp=%.4f fn=%.4f vq=%.4f | "
            "corrected_dice=%.4f | perplexity=%.1f/%d(%.0f%%) | best=%.4f @ep%d",
            epoch + 1, epochs, time.perf_counter() - t0,
            train_m["total"], val_m["total"], val_m["fp"], val_m["fn"], val_m["vq"],
            val_m["recon_dice"], val_m["perplexity"], num_embeddings,
            100.0 * val_m["perplexity"] / num_embeddings, best_dice, best_epoch + 1,
        )
        if patience and epochs_since_improve >= patience:
            logger.info("early stop: corrected Dice did not improve by > %.4g for %d epochs (best=%.4f)",
                        min_delta, patience, best_dice)
            break

    if history["train"]:
        plot_curves_vq(history, out_dir / f"vae3d_{tag}_training_curves.png",
                       title=f"FP/FN dual-head on VAR backbone ({tag})")
    logger.info("done (%s). best corrected Dice=%.4f @ep%d", tag, best_dice, best_epoch + 1)


if __name__ == "__main__":
    main()
