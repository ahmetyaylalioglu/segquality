"""Train the FP/FN dual-head image-conditioned 3D VQ-VAE (advisor direction G).

Input is the SAME 2-channel ``[grayscale CT, broken mask]`` as the E2-E7 line; the output is
TWO binary masks — a false-positive mask (voxels to REMOVE from the input) and a
false-negative mask (voxels to ADD). The FP/FN targets are computed from the clean mask
*relative to the broken input* (:func:`data.fpfn_targets.fpfn_targets`), so the clean mask is
NEVER a network input — it only supervises the loss. Because the targets are keyed to the
input mask, the model cannot ignore the mask channel (the completion-regime confound).

Four models via ``--head {binary,sdfio}`` x ``--quantizer {vq,simvq}``
(:mod:`models.vae3d_imgcond_fpfn`). The binary head predicts FP/FN mask logits (BCE); the
SDF-io head consumes ``[image, SDF(broken)]`` and predicts the normalised SDFs of the FP/FN
regions (SmoothL1), restored to binary by the ``sdf < 0`` bridge. The input mask is always
degraded (FP/FN are only meaningful against a broken input); the degradation set mirrors
:mod:`scripts.train_imgcond` (``mixed`` default / ``partial``). Loss is ``L(FP) + L(FN) +
VQ``. The val metric is the Dice of the CORRECTED mask ``(broken AND NOT FP_pred) OR
FN_pred`` against the (reconstructed) clean mask — higher is better, used for
checkpointing/early-stop.

Usage
-----
    # single-epoch smoke (binary, standard VQ)
    python scripts/train_imgcond_fpfn.py --head binary --quantizer vq --tag g_vq \
        --epochs 1 --limit 24 --num-workers 0
    # full runs (user runs): SimVQ binary, and the two SDF-io variants
    python scripts/train_imgcond_fpfn.py --head binary --quantizer simvq --tag g_simvq --num-workers 0
    python scripts/train_imgcond_fpfn.py --head sdfio  --quantizer vq    --tag g_sdfio_vq --num-workers 0
    python scripts/train_imgcond_fpfn.py --head sdfio  --quantizer simvq --tag g_sdfio_simvq --num-workers 0
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
from data.degradations import SyntheticDegrader  # noqa: E402
from data.fpfn_targets import correct_mask, fpfn_targets  # noqa: E402
from data.raos_image_volume import RAOSImageVolumes  # noqa: E402
from models.vae3d_imgcond_fpfn import (  # noqa: E402
    ConvVQVAE3D_ImgCond_FPFN,
    ConvVQVAE3D_ImgCond_FPFN_sdf_io,
    ConvVQVAE3D_SimVQ_ImgCond_FPFN,
    ConvVQVAE3D_SimVQ_ImgCond_FPFN_sdf_io,
)
from quality.sdf_paper import DELTA, batch_input_sdf, normalized_to_binary  # noqa: E402
from train import improved, resolve_device, save_checkpoint, set_seed  # noqa: E402
from train_imgcond import (  # noqa: E402
    TRAIN_PRIMITIVE_SETS,
    _degrade_partial_single,
    _hard_dice,
)
from train_vae3d import _limit_dataset  # noqa: E402
from train_vae3d_vq import plot_curves_vq  # noqa: E402

logger = logging.getLogger("train_imgcond_fpfn")

# (head, quantizer) -> model class. All consume 2-channel [image, broken mask] and emit a
# 2-channel [FP, FN] prediction (mask logits for binary, normalised SDFs for sdfio).
FPFN_MODELS = {
    ("binary", "vq"): ConvVQVAE3D_ImgCond_FPFN,
    ("binary", "simvq"): ConvVQVAE3D_SimVQ_ImgCond_FPFN,
    ("sdfio", "vq"): ConvVQVAE3D_ImgCond_FPFN_sdf_io,
    ("sdfio", "simvq"): ConvVQVAE3D_SimVQ_ImgCond_FPFN_sdf_io,
}


class FPFNCollate:
    """Collate ``RAOSImageVolumes`` into ``(x=[image, broken mask], target=[FP, FN])``.

    ``x`` is ``(B, 2, D, H, W)`` with channel 0 the CT image and channel 1 the BROKEN input
    mask; ``target`` is ``(B, 2, D, H, W)`` with channel 0 the FP target and channel 1 the FN
    target, both derived from the clean mask *relative to the broken one*. The clean mask is
    used only here, to build targets — never fed to the network. A single seeded RNG makes
    the stream reproducible. ``partial=True`` uses the single-primitive partial degradation
    path, matching :class:`scripts.train_imgcond.ImgCondCollate`.
    """

    def __init__(self, degrader: SyntheticDegrader, seed: int, partial: bool = False) -> None:
        self.degrader = degrader
        self.rng = np.random.default_rng(seed)
        self.partial = partial

    def __call__(self, batch: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
        images = np.stack([np.asarray(s["image"], dtype=np.float32) for s in batch])
        clean = np.stack([np.asarray(s["mask"], dtype=np.float32) for s in batch])
        broken = np.empty_like(clean)
        for i in range(len(batch)):
            if self.partial:
                degraded = _degrade_partial_single(self.degrader, clean[i] > 0.5, self.rng)
            else:
                degraded, _ = self.degrader.make_pair(clean[i] > 0.5, self.rng, target_dice=None)
            broken[i] = (np.asarray(degraded) > 0).astype(np.float32)
        fp, fn = fpfn_targets(broken, clean)                         # (B,D,H,W) each, {0,1}
        x = torch.from_numpy(np.stack([images, broken], axis=1))      # (B,2,D,H,W)
        target = torch.from_numpy(np.stack([fp, fn], axis=1))         # (B,2,D,H,W)
        return x, target


def make_loaders(
    config: dict[str, Any], limit: int | None, *, seed: int, primitive_set: str = "mixed",
) -> tuple[DataLoader, DataLoader]:
    """Build train/val loaders with the FP/FN collate (input always degraded)."""
    tcfg = config["train"]
    cache_size = int(tcfg.get("cache_size", 8))
    train_ds = _limit_dataset(
        RAOSImageVolumes.from_config(config, split="train", group="train", cache_size=cache_size),
        limit,
    )
    val_ds = _limit_dataset(
        RAOSImageVolumes.from_config(config, split="val", group="train", cache_size=cache_size),
        limit,
    )
    primitives = TRAIN_PRIMITIVE_SETS[primitive_set]
    train_degrader = SyntheticDegrader.from_config(config, primitives=primitives)
    val_degrader = SyntheticDegrader.from_config(config, primitives=primitives)
    if primitive_set == "partial":
        n_cal = min(16, len(train_ds))
        cal_masks = [np.asarray(train_ds[i]["mask"], dtype=np.float32) > 0.5 for i in range(n_cal)]
        train_degrader.calibrate(cal_masks)
        val_degrader.calibrate(cal_masks)
        logger.info("calibrated partial degrader on %d masks: %s", n_cal,
                    {k: (round(v[0], 2), round(v[1], 2)) for k, v in train_degrader.type_ranges.items()})
    common = dict(
        batch_size=int(tcfg["batch_size"]),
        num_workers=int(tcfg["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    is_partial = primitive_set == "partial"
    train_loader = DataLoader(
        train_ds, shuffle=True, drop_last=False,
        collate_fn=FPFNCollate(train_degrader, seed, partial=is_partial), **common,
    )
    val_loader = DataLoader(
        val_ds, shuffle=False, drop_last=False,
        collate_fn=FPFNCollate(val_degrader, seed + 1, partial=is_partial), **common,
    )
    logger.info("train volumes: %d | val volumes: %d | primitive_set=%s",
                len(train_ds), len(val_ds), primitive_set)
    return train_loader, val_loader


def build_fpfn_model(config: dict[str, Any], head: str, quantizer: str, delta: float):
    """Instantiate the (head, quantizer) FP/FN model from the ``vae3d_vq`` block (out=2)."""
    v = config["vae3d_vq"]
    kwargs: dict[str, Any] = dict(
        base_channels=int(v["base_channels"]),
        num_embeddings=int(v["num_embeddings"]),
        embedding_dim=int(v["embedding_dim"]),
        commitment_cost=float(v["commitment_cost"]),
        input_size=tuple(v["input_size"]),
        use_skip=bool(v.get("use_skip", False)),
    )
    if head == "sdfio":
        kwargs["delta"] = float(delta)
    return FPFN_MODELS[(head, quantizer)](**kwargs)


def run_epoch_fpfn(
    model, loader: DataLoader, device: torch.device,
    *, head: str, delta: float, optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """One pass over ``[image, broken]`` inputs predicting the ``[FP, FN]`` channels.

    Loss = ``L(FP) + L(FN) + VQ`` where L is BCE-with-logits on the mask (binary head) or
    SmoothL1 on the normalised SDF of each region (sdfio head). Val additionally scores the
    hard Dice of the corrected mask ``(broken AND NOT FP_pred) OR FN_pred`` against the clean
    mask (reconstructed from broken + the target FP/FN), the completion metric.
    """
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = defaultdict(float)
    perp_sum = used_sum = dice_sum = 0.0
    n = 0
    torch.set_grad_enabled(training)
    for x, target in loader:
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if head == "sdfio":
            sdf_in = model.to_sdf_input(x)                        # [image, SDF(broken)]
            logits, vq_term, info = model.forward(sdf_in)
            fp_tgt = batch_input_sdf(target[:, 0:1], delta, device=device)
            fn_tgt = batch_input_sdf(target[:, 1:2], delta, device=device)
            fp_loss = F.smooth_l1_loss(logits[:, 0:1], fp_tgt)
            fn_loss = F.smooth_l1_loss(logits[:, 1:2], fn_tgt)
        else:
            logits, vq_term, info = model.forward(x)
            fp_loss = F.binary_cross_entropy_with_logits(logits[:, 0:1], target[:, 0:1])
            fn_loss = F.binary_cross_entropy_with_logits(logits[:, 1:2], target[:, 1:2])
        recon = fp_loss + fn_loss
        total = recon + vq_term
        if training:
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
        bs = x.size(0)
        n += bs
        totals["total"] += total.item() * bs
        totals["recon"] += recon.item() * bs
        totals["fp"] += fp_loss.item() * bs
        totals["fn"] += fn_loss.item() * bs
        totals["vq"] += vq_term.item() * bs
        perp_sum += float(info["perplexity"].item()) * bs
        used_sum += float(info["codebook_used"].item()) * bs
        if not training:
            broken = x[:, 1:2]
            if head == "sdfio":
                fp_pred = normalized_to_binary(logits[:, 0:1])
                fn_pred = normalized_to_binary(logits[:, 1:2])
            else:
                probs = torch.sigmoid(logits)
                fp_pred, fn_pred = probs[:, 0:1], probs[:, 1:2]
            corrected = correct_mask(broken, fp_pred, fn_pred)
            clean = correct_mask(broken, target[:, 0:1], target[:, 1:2])
            dice_sum += _hard_dice(corrected, clean) * bs
    torch.set_grad_enabled(True)
    metrics = {k: totals[k] / n for k in totals}
    metrics["perplexity"] = perp_sum / n
    metrics["codebook_used"] = used_sum / n
    metrics.setdefault("bce", 0.0)     # keep plot_curves_vq happy (reads bce/dice/blur)
    metrics.setdefault("dice", 0.0)
    if not training:
        metrics["recon_dice"] = dice_sum / n
        metrics["blur"] = 0.0
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the FP/FN dual-head image-conditioned 3D VQ-VAE (dir G).")
    p.add_argument("--config", default="configs/vae3d_imgcond_fpfn.yaml",
                   help="64³ config providing the vae3d_vq model block + precomputed data")
    p.add_argument("--head", choices=("binary", "sdfio"), default="binary",
                   help="FP/FN head: binary (BCE on masks) or sdfio (SmoothL1 on region SDFs)")
    p.add_argument("--quantizer", choices=("vq", "simvq"), default="vq")
    p.add_argument("--primitive-set", choices=("mixed", "partial"), default="mixed",
                   help="training degradation set: 'mixed' (default) or 'partial' (pure completion)")
    p.add_argument("--tag", default=None, help="checkpoint/output name tag (default: auto)")
    p.add_argument("--delta", type=float, default=None, help="TSDF half-width δ for sdfio (else config)")
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
    config.setdefault("data", {})["data_source"] = config.get("data", {}).get("data_source", "precomputed")
    if args.epochs is not None:
        config["train"]["epochs"] = int(args.epochs)
    if args.num_workers is not None:
        config["train"]["num_workers"] = int(args.num_workers)

    delta = float(args.delta) if args.delta is not None else float(
        config.get("vae3d_vq_sdf_io", {}).get("delta", DELTA))

    tcfg = config["train"]
    set_seed(int(tcfg["seed"]))
    device = resolve_device(str(tcfg["device"]))

    train_loader, val_loader = make_loaders(
        config, args.limit, seed=int(tcfg["seed"]), primitive_set=args.primitive_set)

    model = build_fpfn_model(config, args.head, args.quantizer, delta).to(device)
    num_embeddings = int(config["vae3d_vq"]["num_embeddings"])
    logger.info(
        "%s: %d params | head=%s quantizer=%s | in=%d out=%d | codebook=%d | primitive_set=%s | delta=%.1f",
        type(model).__name__, model.param_count, args.head, args.quantizer,
        model.in_channels, model.out_channels, num_embeddings, args.primitive_set, delta,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(tcfg["lr"]))

    tag = args.tag or f"imgcond_fpfn_{args.head}_{args.quantizer}"
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
        train_m = run_epoch_fpfn(model, train_loader, device, head=args.head, delta=delta, optimizer=optimizer)
        val_m = run_epoch_fpfn(model, val_loader, device, head=args.head, delta=delta)
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
                       title=f"FP/FN dual-head 3D VQ-VAE curves ({tag})")
    logger.info("done (%s). best corrected Dice=%.4f @ep%d", tag, best_dice, best_epoch + 1)


if __name__ == "__main__":
    main()
