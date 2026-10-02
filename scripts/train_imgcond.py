"""Train the image-conditioned 3D VQ-VAE variants (image-channel workstream, E2-E7).

The 2-channel ``[grayscale CT, mask]`` sibling of the mask-only trainers. It feeds the
encoder both the CT image and a mask and predicts the **single clean mask** (binary head)
or its normalised SDF (SDF-io head); the image is a CONDITION, never regressed. Loss is
computed only on the 1-channel mask/SDF output.

Four models are selected by ``--head {binary,sdfio}`` x ``--quantizer {vq,simvq}``
(:mod:`models.vae3d_imgcond` / :mod:`models.vae3d_imgcond_sdf_io`), covering:

* E2/E3  binary head, SimVQ  (clean / degraded input)
* E4/E6  SDF-io head, VQ     (clean / degraded input)
* E5/E7  SDF-io head, SimVQ  (clean / degraded input)

``--degrade-input`` is the completion test (K4): the INPUT mask channel is broken with a
random :class:`SyntheticDegrader` degradation while the TARGET stays the clean mask, so the
model is forced to recover the clean shape using the CT image (it cannot just copy the mask
channel). Without the flag the input mask is clean (baseline autoencoder — the model can
ignore the image). The CT image is identical in both cases.

Data comes from :class:`data.raos_image_volume.RAOSImageVolumes` (``data.data_source:
precomputed`` -> ``outputs/image_volumes/{split}.npz``, both channels served from memory).
Everything dimension-agnostic — checkpointing, early stop, VQ curve plot, diagnosis
counters — is imported from the mask-only trainers, so this run shares their loop/logging.

Usage
-----
    # single-epoch smoke (E3: binary SimVQ, degraded input)
    python scripts/train_imgcond.py --head binary --quantizer simvq --degrade-input \
        --tag e3 --epochs 1 --limit 24 --num-workers 0
    # full run (user runs), e.g. E5: SDF-io SimVQ, clean input
    python scripts/train_imgcond.py --head sdfio --quantizer simvq --tag e5 --num-workers 0
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
from data.degradations_3d import PRIMITIVES_3D  # noqa: E402
from data.degradations_partial_3d import PARTIAL_PRIMITIVES_3D  # noqa: E402
from data.raos_image_volume import RAOSImageVolumes  # noqa: E402
from models.vae3d_imgcond import ConvVQVAE3D_ImgCond, ConvVQVAE3D_SimVQ_ImgCond  # noqa: E402
from models.vae3d_imgcond_sdf_io import (  # noqa: E402
    ConvVQVAE3D_ImgCond_sdf_io,
    ConvVQVAE3D_SimVQ_ImgCond_sdf_io,
)
from quality.sdf_paper import DELTA, batch_input_sdf  # noqa: E402
from train import improved, resolve_device, save_checkpoint, set_seed  # noqa: E402
from train_vae3d import _limit_dataset  # noqa: E402
from train_vae3d_vq import plot_curves_vq  # noqa: E402

logger = logging.getLogger("train_imgcond")

# Training-time input degradation uses the MILD standard set PLUS the SEVERE partial set
# (heavy_erode / center_only / partial): the mild breaks teach edge/blob completion, the
# severe ones (centre-only, fragment, Dice~0.5 erosion — the advisor's completion cases)
# force the model to READ THE CT IMAGE instead of leaning on the shape prior. Eval keeps the
# standard set only, so QC-correlation numbers stay comparable to the mask-only experiments.
TRAIN_PRIMITIVES_3D = {**PRIMITIVES_3D, **PARTIAL_PRIMITIVES_3D}

# Named training-time degradation sets. "mixed" (default) is the original behaviour; "partial"
# is the advisor's pure completion regime (severe PARTIAL_PRIMITIVES_3D only: heavy_erode /
# center_only / partial), which forces the model to rely on the CT image.
TRAIN_PRIMITIVE_SETS = {
    "mixed": TRAIN_PRIMITIVES_3D,
    "partial": PARTIAL_PRIMITIVES_3D,
}

# (head, quantizer) -> model class. binary heads predict the mask logits; sdfio heads
# predict the clean mask's normalised SDF. Both consume the 2-channel [image, mask] input.
IMGCOND_MODELS = {
    ("binary", "vq"): ConvVQVAE3D_ImgCond,
    ("binary", "simvq"): ConvVQVAE3D_SimVQ_ImgCond,
    ("sdfio", "vq"): ConvVQVAE3D_ImgCond_sdf_io,
    ("sdfio", "simvq"): ConvVQVAE3D_SimVQ_ImgCond_sdf_io,
}


# --------------------------------------------------------------------------
# 2-channel data
# --------------------------------------------------------------------------
def _degrade_partial_single(
    degrader: SyntheticDegrader, clean_bool: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Break ``clean_bool`` with a SINGLE calibrated partial primitive.

    The partial set (heavy_erode / center_only / partial) omits the primitive names that
    ``SyntheticDegrader``'s chain (``_CHAIN_TYPES``) and degenerate ``_guard`` reference by
    literal name, so the chaining path in ``make_pair`` would raise ``KeyError``. For the
    pure-completion regime a single severe break per sample is both sufficient and faithful
    to the intent, so we reuse only the safe, name-scoped helpers: ``_choose_type`` (returns
    a calibrated partial name) and ``_search_severity`` (binary-searches severity of that one
    name to hit a sampled target Dice). A short retry loop skips degenerate (empty/full)
    outputs; if every attempt degenerates the clean mask is returned unchanged.
    """
    low, high = degrader.params.target_dice_range
    n_vox = clean_bool.size
    for _ in range(4):
        target = float(rng.uniform(low, high))
        name = degrader._choose_type(rng, target)
        seed = int(rng.integers(0, 2**31 - 1))
        _, cand, _ = degrader._search_severity(name, clean_bool, clean_bool, seed, target)
        fg = int(np.asarray(cand).sum())
        if 0 < fg < n_vox:
            return np.asarray(cand)
    return clean_bool


class ImgCondCollate:
    """Collate ``RAOSImageVolumes`` samples into ``(x=[image, input_mask], target_mask)``.

    ``x`` is ``(B, 2, D, H, W)`` with channel 0 the CT image and channel 1 the (possibly
    degraded) INPUT mask; ``target`` is ``(B, 1, D, H, W)`` the CLEAN mask. When
    ``degrader`` is set, each sample's input mask is broken with a random degradation
    (target stays clean) — the completion setup. A single seeded RNG makes the whole
    stream reproducible (val gets its own fixed seed). ``partial=True`` uses the
    single-primitive partial path (:func:`_degrade_partial_single`) instead of the chaining
    ``make_pair``, since the partial set lacks the chain/guard primitive names.
    """

    def __init__(self, degrader: SyntheticDegrader | None, seed: int, partial: bool = False) -> None:
        self.degrader = degrader
        self.rng = np.random.default_rng(seed)
        self.partial = partial

    def __call__(self, batch: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
        images = np.stack([np.asarray(s["image"], dtype=np.float32) for s in batch])
        clean = np.stack([np.asarray(s["mask"], dtype=np.float32) for s in batch])
        if self.degrader is not None:
            inp = np.empty_like(clean)
            for i in range(len(batch)):
                if self.partial:
                    degraded = _degrade_partial_single(self.degrader, clean[i] > 0.5, self.rng)
                else:
                    degraded, _ = self.degrader.make_pair(clean[i] > 0.5, self.rng, target_dice=None)
                inp[i] = (np.asarray(degraded) > 0).astype(np.float32)
        else:
            inp = clean
        x = torch.from_numpy(np.stack([images, inp], axis=1))       # (B, 2, D, H, W)
        target = torch.from_numpy(clean).unsqueeze(1)               # (B, 1, D, H, W)
        return x, target


def make_loaders(
    config: dict[str, Any], limit: int | None, *, degrade_input: bool, seed: int,
    primitive_set: str = "mixed",
) -> tuple[DataLoader, DataLoader]:
    """Build train/val ``RAOSImageVolumes`` loaders with the 2-channel collate.

    When ``degrade_input`` is set, both loaders break the INPUT mask (train and val get
    independent seeded degraders so val stays reproducible); the CT image and the clean
    target are untouched. ``primitive_set`` selects the training-time degradation set
    (``"mixed"`` default, ``"partial"`` for the advisor's pure completion regime).
    """
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
    train_degrader = SyntheticDegrader.from_config(config, primitives=primitives) if degrade_input else None
    val_degrader = SyntheticDegrader.from_config(config, primitives=primitives) if degrade_input else None
    # The "partial" set omits the names hard-coded in the uncalibrated _choose_type fallback
    # (erode/dilate/remove_blob/translate), so an uncalibrated partial degrader would raise
    # KeyError. Calibrating on a sample of train masks populates type_ranges over the partial
    # primitives only, so type selection stays within the available set. (Mixed keeps the
    # original uncalibrated behaviour untouched.)
    if degrade_input and primitive_set == "partial":
        n_cal = min(16, len(train_ds))
        cal_masks = [np.asarray(train_ds[i]["mask"], dtype=np.float32) > 0.5 for i in range(n_cal)]
        train_degrader.calibrate(cal_masks)
        val_degrader.calibrate(cal_masks)
        logger.info("calibrated partial degrader on %d masks: %s",
                    n_cal, {k: (round(v[0], 2), round(v[1], 2)) for k, v in train_degrader.type_ranges.items()})
    common = dict(
        batch_size=int(tcfg["batch_size"]),
        num_workers=int(tcfg["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    is_partial = degrade_input and primitive_set == "partial"
    train_loader = DataLoader(
        train_ds, shuffle=True, drop_last=False,
        collate_fn=ImgCondCollate(train_degrader, seed, partial=is_partial), **common,
    )
    val_loader = DataLoader(
        val_ds, shuffle=False, drop_last=False,
        collate_fn=ImgCondCollate(val_degrader, seed + 1, partial=is_partial), **common,
    )
    logger.info("train volumes: %d | val volumes: %d | degrade_input=%s | primitive_set=%s",
                len(train_ds), len(val_ds), degrade_input, primitive_set)
    return train_loader, val_loader


def build_imgcond_model(config: dict[str, Any], head: str, quantizer: str, delta: float):
    """Instantiate the (head, quantizer) image-conditioned model from the ``vae3d_vq`` block."""
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
    return IMGCOND_MODELS[(head, quantizer)](**kwargs)


# --------------------------------------------------------------------------
# training loop
# --------------------------------------------------------------------------
def _hard_dice(pred_bin: torch.Tensor, target: torch.Tensor) -> float:
    dims = tuple(range(1, pred_bin.dim()))
    inter = (pred_bin * target).sum(dim=dims)
    denom = pred_bin.sum(dim=dims) + target.sum(dim=dims)
    return ((2.0 * inter + 1.0) / (denom + 1.0)).mean().item()


def run_epoch_imgcond(
    model,
    loader: DataLoader,
    device: torch.device,
    *,
    head: str,
    delta: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """One pass over ``[image, mask]`` inputs predicting the clean mask (or its SDF).

    Loss is on the 1-channel output only: BCE-with-logits vs the clean mask (binary head)
    or SmoothL1 vs the clean mask's SDF (sdfio head), plus the VQ term. Val adds the hard
    Dice of the recovered binary mask against the clean target (the completion metric).
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
            sdf_in = model.to_sdf_input(x)                      # [image, SDF(input mask)]
            pred, vq_term, info = model.forward(sdf_in)
            tgt_sdf = batch_input_sdf(target, delta, device=device)   # SDF of the CLEAN mask
            recon = F.smooth_l1_loss(pred, tgt_sdf)
        else:
            pred, vq_term, info = model.forward(x)
            recon = F.binary_cross_entropy_with_logits(pred, target)
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
            pred_bin = (pred < 0).float() if head == "sdfio" else (torch.sigmoid(pred) > 0.5).float()
            dice_sum += _hard_dice(pred_bin, target) * bs
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


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the image-conditioned 3D VQ-VAE (E2-E7).")
    p.add_argument("--config", default="configs/vae3d_vq_sdf_io.yaml",
                   help="64³ config providing the vae3d_vq model block + precomputed data")
    p.add_argument("--head", choices=("binary", "sdfio"), required=True,
                   help="mask head: binary (BCE) or sdfio (SmoothL1 on SDF)")
    p.add_argument("--quantizer", choices=("vq", "simvq"), required=True)
    p.add_argument("--primitive-set", choices=("mixed", "partial"), default="mixed",
                   help="training degradation set: 'mixed' (default) or 'partial' (advisor's pure completion)")
    p.add_argument("--degrade-input", action="store_true",
                   help="break the INPUT mask (target stays clean) — the completion test")
    p.add_argument("--tag", default=None, help="checkpoint/output name tag (default: auto)")
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
        config, args.limit, degrade_input=args.degrade_input, seed=int(tcfg["seed"]),
        primitive_set=args.primitive_set)
    val_ds = val_loader.dataset

    model = build_imgcond_model(config, args.head, args.quantizer, delta).to(device)
    num_embeddings = int(config["vae3d_vq"]["num_embeddings"])
    logger.info(
        "%s: %d params | head=%s quantizer=%s | in=%d out=%d | codebook=%d | degrade_input=%s | delta=%.1f",
        type(model).__name__, model.param_count, args.head, args.quantizer,
        model.in_channels, model.out_channels, num_embeddings, args.degrade_input, delta,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(tcfg["lr"]))

    suffix = "deg" if args.degrade_input else "clean"
    tag = args.tag or f"imgcond_{args.head}_{args.quantizer}_{suffix}"
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
        train_m = run_epoch_imgcond(model, train_loader, device, head=args.head, delta=delta, optimizer=optimizer)
        val_m = run_epoch_imgcond(model, val_loader, device, head=args.head, delta=delta)
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
            "epoch %d/%d | %.0fs | train=%.4f val=%.4f | recon=%.4f vq=%.4f | "
            "val_dice=%.4f | perplexity=%.1f/%d(%.0f%%) | best=%.4f @ep%d",
            epoch + 1, epochs, time.perf_counter() - t0,
            train_m["total"], val_m["total"], val_m["recon"], val_m["vq"],
            val_m["recon_dice"], val_m["perplexity"], num_embeddings,
            100.0 * val_m["perplexity"] / num_embeddings, best_dice, best_epoch + 1,
        )
        if patience and epochs_since_improve >= patience:
            logger.info("early stop: val Dice did not improve by > %.4g for %d epochs (best=%.4f)",
                        min_delta, patience, best_dice)
            break

    if history["train"]:
        plot_curves_vq(history, out_dir / f"vae3d_{tag}_training_curves.png",
                       title=f"image-conditioned 3D VQ-VAE curves ({tag})")
    logger.info("done (%s). best val Dice=%.4f @ep%d", tag, best_dice, best_epoch + 1)


if __name__ == "__main__":
    main()
