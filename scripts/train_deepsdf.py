"""Train the DeepSDF auto-decoder (continuous implicit shape prior).

This is a NEW, self-contained trainer; it edits nothing in the VQ line. It reuses the
untouched data (``RAOSVolumes.from_precomputed``), SDF helpers (``quality/sdf.py``)
and the shared checkpoint/seed/device utilities, so the shape prior it learns sits in
the same 64^3 / same-sign-convention frame as WP-3.

Auto-decoder recipe (Park et al., 2019)
---------------------------------------
* Each training shape owns a latent row in ``model.latents`` (an ``nn.Embedding``).
* A step samples ``points_per_shape`` ``(xyz, sdf)`` pairs for a minibatch of shapes
  (near-surface weighted) and regresses ``SmoothL1(clamp(pred, +-t), clamp(tgt, +-t))``
  plus a ``latent_reg * ||z||^2`` code regulariser. Decoder and latents get separate
  LRs (the latents move faster, as in the paper).
* Validation here is a *train-reconstruction* sanity metric only: decode a fixed
  subset of train shapes on the full grid, bridge ``sdf < 0 -> inside`` and Dice
  against the mask. Held-out latent inference (the real QC correlation) is Stage 2.

Usage
-----
    # single-epoch smoke
    python scripts/train_deepsdf.py --tag deepsdf_smoke --epochs 1 --limit 16
    # full run (user runs)
    python scripts/train_deepsdf.py --tag deepsdf
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data import load_config  # noqa: E402
from data.raos_volume import RAOSVolumes  # noqa: E402
from data.sdf_points import SDFPointSampler  # noqa: E402
from models.deepsdf import DeepSDF  # noqa: E402
from quality.sdf import TRUNCATION  # noqa: E402
from train import improved, resolve_device, save_checkpoint, set_seed  # noqa: E402

logger = logging.getLogger("train_deepsdf")


def load_masks(config: dict[str, Any], split: str, limit: int | None) -> list[np.ndarray]:
    ds = RAOSVolumes.from_precomputed(config, split)
    n = len(ds) if limit is None else min(limit, len(ds))
    return [np.asarray(ds[i]["mask"], dtype=np.float32) for i in range(n)]


def build_model(config: dict[str, Any], num_shapes: int) -> DeepSDF:
    d = config["deepsdf"]
    return DeepSDF(
        num_shapes=num_shapes,
        latent_dim=int(d["latent_dim"]),
        hidden_dim=int(d["hidden_dim"]),
        num_layers=int(d["num_layers"]),
        skip_layers=tuple(int(s) for s in d["skip_layers"]),
        dropout=float(d["dropout"]),
        latent_init_std=float(d["latent_init_std"]),
        weight_norm=bool(d["weight_norm"]),
    )


@torch.no_grad()
def decode_binary(model: DeepSDF, sampler: SDFPointSampler, shape_idx: int,
                  device: torch.device, chunk: int = 65536) -> np.ndarray:
    """Decode shape ``shape_idx`` on the full grid -> binary via ``sdf < 0``."""
    model.eval()
    coords = sampler.coords.to(device)
    z = model.latents(torch.tensor([shape_idx], device=device))  # (1, L)
    preds = []
    for start in range(0, coords.shape[0], chunk):
        c = coords[start:start + chunk]
        zc = z.expand(c.shape[0], -1)
        preds.append(model.decode(zc, c))
    sdf = torch.cat(preds).view(sampler.shape)
    return (sdf < 0).cpu().numpy()


def _dice(pred_bin: np.ndarray, target_bin: np.ndarray) -> float:
    inter = np.logical_and(pred_bin, target_bin).sum()
    denom = pred_bin.sum() + target_bin.sum()
    return float((2.0 * inter + 1.0) / (denom + 1.0))


def recon_dice(model: DeepSDF, sampler: SDFPointSampler, masks: list[np.ndarray],
               idx: list[int], device: torch.device) -> float:
    scores = [_dice(decode_binary(model, sampler, s, device), masks[s] > 0.5) for s in idx]
    return float(np.mean(scores))


def plot_curves(history: list[dict[str, float]], out_path: Path, title: str) -> None:
    """Loss/recon/reg and bridge-Dice curves. Dice is sparse (recon_every), so its
    panel plots only the epochs where it was measured."""
    epochs = list(range(1, len(history) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [h["loss"] for h in history], marker="o", ms=3)
    axes[0].set_title("total loss"); axes[0].set_xlabel("epoch")
    axes[1].plot(epochs, [h["recon"] for h in history], color="C1", marker="o", ms=3, label="recon (SmoothL1)")
    axes[1].plot(epochs, [h["reg"] for h in history], color="C3", marker="o", ms=3, label="reg (||z||^2)")
    axes[1].set_title("recon / latent-reg"); axes[1].set_xlabel("epoch"); axes[1].legend()
    de = [(e, h["dice"]) for e, h in zip(epochs, history) if not np.isnan(h["dice"])]
    if de:
        axes[2].plot([e for e, _ in de], [d for _, d in de], color="C2", marker="o", ms=4)
    axes[2].set_title("train reconstruction Dice (bridge)"); axes[2].set_xlabel("epoch"); axes[2].set_ylim(0, 1)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the DeepSDF auto-decoder.")
    p.add_argument("--config", default="configs/deepsdf.yaml")
    p.add_argument("--tag", default="deepsdf", help="checkpoint/output name tag")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="cap train shapes (smoke)")
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

    tcfg = config["train"]
    dcfg = config["deepsdf"]
    set_seed(int(tcfg["seed"]))
    device = resolve_device(str(tcfg["device"]))

    masks = load_masks(config, "train", args.limit)
    num_shapes = len(masks)
    truncation = float(dcfg.get("truncation", TRUNCATION))
    sampler = SDFPointSampler(masks, truncation=truncation,
                              surface_band=float(dcfg["surface_band"]))

    model = build_model(config, num_shapes).to(device)
    logger.info("DeepSDF: %d params | shapes=%d | latent=%d | MLP=%dx%d skip=%s | TSDF=+-%.1f",
                model.param_count, num_shapes, model.latent_dim, model.hidden_dim,
                model.num_layers, sorted(model.skip_layers), truncation)

    decoder_params = [p for n, p in model.named_parameters() if not n.startswith("latents")]
    optimizer = torch.optim.Adam([
        {"params": decoder_params, "lr": float(tcfg["lr"])},
        {"params": model.latents.parameters(), "lr": float(tcfg["latent_lr"])},
    ])

    tag = args.tag
    out_dir = PROJECT_ROOT / config["output_dir"]
    ckpt_dir = out_dir / "checkpoints"
    epochs = int(tcfg["epochs"])
    ppShape = int(tcfg["points_per_shape"])
    near_frac = float(tcfg["near_surface_frac"])
    shapes_per_batch = int(tcfg["shapes_per_batch"])
    latent_reg = float(tcfg["latent_reg"])
    recon_every = int(tcfg.get("recon_every", 10))
    recon_k = int(tcfg.get("recon_k", 12))
    recon_seed = int(tcfg.get("recon_seed", 42))
    patience = int(tcfg.get("early_stop_patience", 0))
    min_delta = float(tcfg.get("min_delta", 0.0))

    rng = np.random.default_rng(recon_seed)
    recon_idx = sorted(rng.choice(num_shapes, size=min(recon_k, num_shapes), replace=False).tolist())

    best_dice = -1.0
    best_epoch = 0
    epochs_since_improve = 0
    history: list[dict[str, float]] = []
    logger.info("run tag=deepsdf_%s | epochs=%d shapes/batch=%d points/shape=%d latent_reg=%g",
                tag, epochs, shapes_per_batch, ppShape, latent_reg)

    for epoch in range(epochs):
        t0 = time.perf_counter()
        model.train()
        order = np.random.permutation(num_shapes)
        loss_sum = recon_sum = reg_sum = 0.0
        n_steps = 0
        for start in range(0, num_shapes, shapes_per_batch):
            batch = order[start:start + shapes_per_batch].tolist()
            idx, coords, tgt = sampler.sample(batch, ppShape, near_frac)
            idx = idx.to(device)
            coords = coords.to(device)
            tgt = tgt.to(device)
            pred = model(idx, coords)
            recon = F.smooth_l1_loss(pred.clamp(-truncation, truncation),
                                     tgt.clamp(-truncation, truncation))
            reg = model.latent_l2(torch.tensor(batch, device=device))
            loss = recon + latent_reg * reg
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            recon_sum += recon.item()
            reg_sum += reg.item()
            n_steps += 1

        train_loss = loss_sum / n_steps
        train_recon = recon_sum / n_steps
        train_reg = reg_sum / n_steps

        is_last = epoch + 1 == epochs
        dice = float("nan")
        if recon_every and ((epoch + 1) % recon_every == 0 or is_last):
            dice = recon_dice(model, sampler, masks, recon_idx, device)
        history.append({"loss": train_loss, "recon": train_recon, "reg": train_reg, "dice": dice})

        save_checkpoint(ckpt_dir / f"deepsdf_{tag}_last.pt", model, optimizer, epoch, config,
                        {"loss": train_loss, "recon_dice": dice, "best_recon_dice": best_dice})
        improved_now = not np.isnan(dice) and improved(dice, best_dice, min_delta)
        if improved_now:
            best_dice = dice
            best_epoch = epoch
            epochs_since_improve = 0
            save_checkpoint(ckpt_dir / f"deepsdf_{tag}_best.pt", model, optimizer, epoch, config,
                            {"loss": train_loss, "recon_dice": dice, "best_recon_dice": best_dice})
        elif not np.isnan(dice):
            epochs_since_improve += 1

        logger.info(
            "epoch %d/%d | %.0fs | loss=%.4f recon(SmoothL1)=%.4f reg=%.4f | "
            "train_dice(bridge)=%s | best=%.4f @ep%d",
            epoch + 1, epochs, time.perf_counter() - t0, train_loss, train_recon, train_reg,
            "%.4f" % dice if not np.isnan(dice) else "  -  ", best_dice, best_epoch + 1,
        )
        if patience and epochs_since_improve >= patience:
            logger.info("early stop: train bridge Dice did not improve by > %.4g for %d epochs (best=%.4f)",
                        min_delta, patience, best_dice)
            break

    # Persist the full run so results can be read back without scraping the console.
    summary = {
        "tag": tag,
        "num_shapes": num_shapes,
        "epochs_run": len(history),
        "best_recon_dice": best_dice,
        "best_epoch": best_epoch + 1,
        "final_loss": history[-1]["loss"] if history else float("nan"),
        "history": history,
    }
    hist_path = out_dir / f"deepsdf_{tag}_history.json"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    with hist_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    if history:
        plot_curves(history, out_dir / f"deepsdf_{tag}_training_curves.png",
                    title=f"DeepSDF training curves ({tag})")
    logger.info("wrote %s and training-curve PNG", hist_path.name)
    logger.info("done (%s). best train bridge Dice=%.4f @ep%d", tag, best_dice, best_epoch + 1)


if __name__ == "__main__":
    main()
