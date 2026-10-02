"""Does an image-conditioned 3D VQ-VAE's reconstruction error track true Dice? (E2-E7)

The 2-channel ``[grayscale CT, mask]`` counterpart of the mask-only correlators. It answers
the workstream's core question: with the CT image available, does the completion model's
reconstruction error correlate with a mask's true Dice better than the shape-only ceiling
(recon_dice rho ~0.83)?

The protocol is byte-identical to :mod:`evaluate_correlation_3d` — same test split, same
``PRIMITIVES_3D`` degradations, same calibration, same eval-set construction and console
table — with ONE structural change forced by the extra input channel: every mask (clean or
degraded) is scored **alongside its own patient's fixed clean CT image** (the completion
setup — image fixed, mask degrades). So the eval set carries a parallel ``images`` array,
and each 1-channel score compares the reconstruction against the mask channel ``x[:, 1:2]``.

Model + score selection mirrors the trainer (``--head {binary,sdfio}`` x
``--quantizer {vq,simvq}``). ``quant_error`` reads the effective codebook ``C @ W`` for
SimVQ and ``embedding.weight`` for standard VQ; for the SDF-io head the encoder input is
``[image, SDF(mask)]`` (via ``to_sdf_input``) and ``sdf_l1`` is added.

Non-clobbering: writes ``correlation_results_imgcond_{tag}.json`` and ``*_imgcond_{tag}.png``.

Usage
-----
    python scripts/evaluate_correlation_imgcond.py --head binary --quantizer simvq \
        --config configs/vae3d_vq_sdf_io.yaml \
        --checkpoint outputs/checkpoints/vae3d_imgcond_binary_simvq_deg_best.pt \
        --tag e3 --split test --n-per-mask 5 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from data import load_config  # noqa: E402
from data.degradations_3d import PRIMITIVES_3D  # noqa: E402
from data.raos_image_volume import RAOSImageVolumes  # noqa: E402
from quality.sdf_paper import DELTA  # noqa: E402
from quality.score import banded_correlation, correlation  # noqa: E402
from train import resolve_device, set_seed  # noqa: E402
from train_imgcond import IMGCOND_MODELS  # noqa: E402

import evaluate_correlation_3d as ev3d  # noqa: E402

logger = logging.getLogger("evaluate_correlation_imgcond")

SCORE_BATCH = ev3d.SCORE_BATCH


# --------------------------------------------------------------------------
# image-conditioned scores (compare the 1-channel recon against the mask channel)
# --------------------------------------------------------------------------
def _spatial_dims(x: torch.Tensor) -> tuple[int, ...]:
    return tuple(range(2, x.dim()))


@torch.no_grad()
def recon_dice_imgcond(model, x: torch.Tensor) -> torch.Tensor:
    """Hard Dice between the binarised reconstruction and the INPUT mask channel."""
    pred = (model.reconstruct(x) > 0.5).float()
    mask = x[:, 1:2]
    dims = _spatial_dims(x)
    inter = (pred * mask).sum(dim=dims)
    denom = pred.sum(dim=dims) + mask.sum(dim=dims)
    return ((2.0 * inter + 1.0) / (denom + 1.0)).squeeze(1)


@torch.no_grad()
def recon_l1_imgcond(model, x: torch.Tensor) -> torch.Tensor:
    """Mean ``|reconstruction - input mask|`` over space. Higher = worse."""
    pred = model.reconstruct(x)
    mask = x[:, 1:2]
    return (pred - mask).abs().mean(dim=_spatial_dims(x)).squeeze(1)


@torch.no_grad()
def recon_bce_imgcond(model, x: torch.Tensor) -> torch.Tensor:
    """Per-mask BCE between the forward logits and the input mask channel (binary head)."""
    logits, _, _ = model.forward(x)
    mask = x[:, 1:2]
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, mask, reduction="none")
    return bce.mean(dim=_spatial_dims(x)).squeeze(1)


@torch.no_grad()
def topology_imgcond(model, x: torch.Tensor) -> torch.Tensor:
    """|#components(recon) - #components(input mask)| via ``scipy.ndimage.label``."""
    from scipy.ndimage import label as cc_label

    pred = (model.reconstruct(x) > 0.5).cpu().numpy()
    inp = (x[:, 1:2] > 0.5).cpu().numpy()
    diffs = np.empty(pred.shape[0], dtype=np.float32)
    for i in range(pred.shape[0]):
        _, n_pred = cc_label(pred[i, 0])
        _, n_inp = cc_label(inp[i, 0])
        diffs[i] = abs(n_pred - n_inp)
    return torch.from_numpy(diffs).to(x.device)


@torch.no_grad()
def sdf_l1_imgcond(model, x: torch.Tensor) -> torch.Tensor:
    """SDF-space reconstruction error ``|predict_sdf(x) - SDF(input mask)|`` (sdfio head)."""
    pred_sdf = model.predict_sdf(x)
    tgt_sdf = model.to_sdf_input(x)[:, 1:2]     # channel 1 = SDF(input mask); channel 0 = image
    return (pred_sdf - tgt_sdf).abs().mean(dim=_spatial_dims(x)).squeeze(1)


def _quant_error_imgcond(head: str, quantizer: str):
    """Build the quant_error scorer: SDF-io encodes ``to_sdf_input(x)``; SimVQ uses ``C@W``."""

    @torch.no_grad()
    def fn(model, x: torch.Tensor) -> torch.Tensor:
        enc_in = model.to_sdf_input(x) if head == "sdfio" else x
        z_e, _ = model._encode_features(enc_in)
        b = z_e.shape[0]
        z = z_e.permute(0, 2, 3, 4, 1).contiguous()
        flat = z.view(-1, model.embedding_dim)
        w = model.quantizer.effective_codebook() if quantizer == "simvq" else model.quantizer.embedding.weight
        dist = flat.pow(2).sum(1, keepdim=True) - 2.0 * flat @ w.t() + w.pow(2).sum(1)
        return dist.min(dim=1).values.view(b, -1).mean(dim=1)

    return fn


def build_score_registry(head: str, quantizer: str) -> tuple[dict, dict]:
    """Return ``(score_fns, directions)`` for the (head, quantizer) combination."""
    quant = _quant_error_imgcond(head, quantizer)
    if head == "sdfio":
        fns = {
            "recon_dice": recon_dice_imgcond,
            "recon_l1": recon_l1_imgcond,
            "topology": topology_imgcond,
            "quant_error": quant,
            "sdf_l1": sdf_l1_imgcond,
        }
        dirs = {"recon_dice": +1, "recon_l1": -1, "topology": -1, "quant_error": -1, "sdf_l1": -1}
    else:
        fns = {
            "recon_dice": recon_dice_imgcond,
            "recon_bce": recon_bce_imgcond,
            "recon_l1": recon_l1_imgcond,
            "topology": topology_imgcond,
            "quant_error": quant,
        }
        dirs = {"recon_dice": +1, "recon_bce": -1, "recon_l1": -1, "topology": -1, "quant_error": -1}
    return fns, dirs


# --------------------------------------------------------------------------
# image-aware data + eval set
# --------------------------------------------------------------------------
def load_clean_image_volumes(
    config: dict[str, Any], split: str, limit: int | None, seed: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Every clean test liver as ``(mask bool, image float32)`` pairs, capped to ``limit``."""
    ds = RAOSImageVolumes.from_config(config, split=split)
    n = len(ds)
    indices = list(range(n))
    if limit and limit < n:
        rng = np.random.default_rng(seed)
        indices = sorted(int(i) for i in rng.choice(n, size=limit, replace=False))
    logger.info("clean image+mask volumes: %d of %d test livers", len(indices), n)
    masks = [np.asarray(ds[i]["mask"], dtype=bool) for i in indices]
    images = [np.asarray(ds[i]["image"], dtype=np.float32) for i in indices]
    return masks, images


def build_eval_set_imgcond(
    clean_masks: list[np.ndarray],
    clean_images: list[np.ndarray],
    degrader,
    n_per_mask: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Clean volumes + ``n_per_mask`` degradations each, carrying the FIXED patient image.

    Returns ``(masks (M,D,H,W) bool, images (M,D,H,W) float32, true_dice (M,), is_clean (M,))``.
    The image is the patient's clean CT for every row (clean and degraded alike) — the mask
    degrades, the image stays fixed, which is exactly the completion setup.
    """
    rng = np.random.default_rng(seed)
    masks: list[np.ndarray] = []
    images: list[np.ndarray] = []
    dice_vals: list[float] = []
    is_clean: list[bool] = []
    for m, img in zip(clean_masks, clean_images):
        masks.append(m.astype(bool)); images.append(img); dice_vals.append(1.0); is_clean.append(True)
        for _ in range(n_per_mask):
            degraded, true_d = degrader.make_pair(m, rng, target_dice=None)
            masks.append(np.asarray(degraded) > 0); images.append(img)
            dice_vals.append(float(true_d)); is_clean.append(False)
    logger.info("eval set: %d volumes (%d clean + %d degraded), image fixed per patient",
                len(masks), sum(is_clean), len(masks) - sum(is_clean))
    return np.stack(masks), np.stack(images), np.asarray(dice_vals), np.asarray(is_clean)


def score_masks_imgcond(
    model, masks: np.ndarray, images: np.ndarray, score_fns: dict, device: torch.device
) -> dict[str, np.ndarray]:
    """Run every score over ``x = [image, mask]`` (B,2,D,H,W) in batches."""
    model.eval()
    accum: dict[str, list[np.ndarray]] = {name: [] for name in score_fns}
    for start in range(0, len(masks), SCORE_BATCH):
        m = masks[start:start + SCORE_BATCH].astype(np.float32)
        img = images[start:start + SCORE_BATCH].astype(np.float32)
        x = torch.from_numpy(np.stack([img, m], axis=1)).to(device)   # ch0=image, ch1=mask
        for name, fn in score_fns.items():
            accum[name].append(fn(model, x).cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in accum.items()}


def summarise_imgcond(scores: dict[str, np.ndarray], true_dice: np.ndarray, directions: dict) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, values in scores.items():
        result[name] = {
            "direction": directions[name],
            "global": correlation(values, true_dice),
            "bands": banded_correlation(values, true_dice),
        }
    return result


def load_imgcond_model(checkpoint: Path, head: str, quantizer: str, device: torch.device):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    v = ckpt["config"]["vae3d_vq"]
    delta = float(ckpt["config"].get("vae3d_vq_sdf_io", {}).get("delta", DELTA))
    kwargs: dict[str, Any] = dict(
        base_channels=int(v["base_channels"]),
        num_embeddings=int(v["num_embeddings"]),
        embedding_dim=int(v["embedding_dim"]),
        commitment_cost=float(v["commitment_cost"]),
        input_size=tuple(v["input_size"]),
        use_skip=bool(v.get("use_skip", False)),
    )
    if head == "sdfio":
        kwargs["delta"] = delta
    model = IMGCOND_MODELS[(head, quantizer)](**kwargs)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    logger.info("loaded %s (%s head=%s quantizer=%s in=%d out=%d epoch=%s)",
                checkpoint.name, type(model).__name__, head, quantizer,
                model.in_channels, model.out_channels, ckpt.get("epoch"))
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Correlate image-conditioned 3D VQ-VAE recon error with true Dice.")
    p.add_argument("--config", default="configs/vae3d_vq_sdf_io.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--head", choices=("binary", "sdfio"), required=True)
    p.add_argument("--quantizer", choices=("vq", "simvq"), required=True)
    p.add_argument("--tag", default=None, help="output filename tag (default: head_quantizer)")
    p.add_argument("--split", default="test")
    p.add_argument("--n-per-mask", type=int, default=5)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--data-source", choices=("nifti", "precomputed"), default="precomputed")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    config = load_config(args.config)
    logging.basicConfig(
        level=getattr(logging, config.get("logging", {}).get("level", "INFO"), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config.setdefault("data", {})["data_source"] = args.data_source
    logger.info("data source: %s | primitives: 3D (%s) | head=%s quantizer=%s",
                config["data"]["data_source"], ", ".join(PRIMITIVES_3D), args.head, args.quantizer)

    set_seed(args.seed)
    device = resolve_device(config["train"]["device"])

    ckpt_path = args.checkpoint
    if not Path(ckpt_path).is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    model = load_imgcond_model(Path(ckpt_path), args.head, args.quantizer, device)

    clean_masks, clean_images = load_clean_image_volumes(config, args.split, args.limit, args.seed)
    degrader = ev3d.calibrate_degrader(config, clean_masks, args.seed)
    masks, images, true_dice, is_clean = build_eval_set_imgcond(
        clean_masks, clean_images, degrader, args.n_per_mask, args.seed)

    score_fns, directions = build_score_registry(args.head, args.quantizer)
    scores = score_masks_imgcond(model, masks, images, score_fns, device)
    summary = summarise_imgcond(scores, true_dice, directions)
    ev3d.print_table(summary)

    best = ev3d.best_score_name(summary)
    rd = summary["recon_dice"]["global"]["spearman"]
    logger.info("best score by |global rho|: %s | recon_dice rho=%.4f", best, rd)

    out_dir = PROJECT_ROOT / "outputs"
    tag = args.tag or f"{args.head}_{args.quantizer}"
    _plot_scatter_local(true_dice, scores[best], is_clean, best, directions,
                        summary[best]["global"], out_dir / f"correlation_scatter_imgcond_{tag}.png")
    ev3d.plot_score_comparison(summary, out_dir / f"score_comparison_imgcond_{tag}.png")

    results = {
        "checkpoint": str(args.checkpoint),
        "tag": tag,
        "model": type(model).__name__,
        "head": args.head,
        "quantizer": args.quantizer,
        "split": args.split,
        "n_per_mask": args.n_per_mask,
        "seed": args.seed,
        "n_total": int(len(true_dice)),
        "n_clean": int(is_clean.sum()),
        "n_degraded": int((~is_clean).sum()),
        "primitive_set": "3D",
        "calibrated_ranges": {name: list(rng) for name, rng in (degrader.type_ranges or {}).items()},
        "best_score": best,
        "recon_dice_rho": float(rd),
        "scores": summary,
    }
    results_path = out_dir / f"correlation_results_imgcond_{tag}.json"
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("wrote %s", results_path)


def _plot_scatter_local(true_dice, scores, is_clean, name, directions, stats, out_path) -> None:
    """Local scatter for scores absent from the mask-only direction table (e.g. sdf_l1)."""
    import matplotlib.pyplot as plt
    from quality.score import DICE_BANDS

    fig, ax = plt.subplots(figsize=(9, 6))
    deg = ~is_clean
    ax.scatter(true_dice[deg], scores[deg], s=12, alpha=0.5, c="#c1121f",
               label=f"degraded (n={int(deg.sum())})")
    ax.scatter(true_dice[is_clean], scores[is_clean], s=24, alpha=0.7, c="#1d3557",
               label=f"clean, Dice=1.0 (n={int(is_clean.sum())})")
    for _, hi in DICE_BANDS[:-1]:
        ax.axvline(hi, color="grey", lw=0.6, ls="--", alpha=0.6)
    order = np.argsort(true_dice)
    coeffs = np.polyfit(true_dice, scores, 1)
    ax.plot(true_dice[order], np.polyval(coeffs, true_dice[order]), color="black", lw=1.2, label="linear trend")
    direction = "higher = better" if directions[name] > 0 else "higher = worse"
    ax.set_xlabel("true Dice")
    ax.set_ylabel(f"{name} score")
    ax.set_title(f"image-conditioned 3D VQ-VAE — best score: {name} ({direction})\n"
                 f"Spearman rho = {stats['spearman']:.3f}   Pearson r = {stats['pearson']:.3f}   n = {stats['n']}")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
