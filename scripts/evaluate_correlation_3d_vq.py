"""Phase 9: does the 3D **VQ-VAE**'s reconstruction error track a volume's true Dice?

The VQ-VAE counterpart of ``scripts/evaluate_correlation_3d.py``. The standard
continuous-latent ConvVAE3D reached global Spearman rho=0.851 *despite*
mean-reversion (its decoder emitted a generic mean shape, so the score was really a
"similarity-to-mean-shape" proxy). The VQ-VAE was shown to CURE mean-reversion
(cross-recon ratio 1.005->1.239, recon-pairwise 0.992->0.744): it now reconstructs
input-specifically. The open question this script answers: does that more honest
reconstruction correlate with true Dice *better*, *worse*, or *the same*?

This is a THIN WRAPPER. It imports the model-agnostic evaluation machinery from
``evaluate_correlation_3d`` unchanged (data loading, degrader calibration, eval-set
construction, the console table) and ``quality.score`` unchanged (four of the five
scores work on the VQ-VAE as-is). It overrides only two things:

* **model loader** -> ``ConvVQVAE3D`` (from the checkpoint's ``vae3d_vq`` config).
* **score registry** -> the ``latent_kl`` score calls ``model._encode_dist`` which
  the VQ-VAE lacks (it has no continuous posterior). It is replaced by
  ``quant_error``: the per-mask mean distance from the encoder output to its nearest
  codebook entry — the VQ analogue of latent_kl (a mask the model finds atypical
  sits far from every learned code). The other four scores (recon_dice, recon_bce,
  recon_l1, topology) are imported and used verbatim.

Because ``evaluate_correlation_3d.py`` and ``quality/score.py`` are not touched, the
127 existing tests stay green.

Outputs
-------
* console: active 3D primitive set, calibrated [dice_min, dice_max] per primitive,
  and a correlation table (global + per band) for every score
* ``outputs/correlation_scatter_vq.png``  — true Dice vs the best score
* ``outputs/score_comparison_vq.png``     — |Spearman rho| per score
* ``outputs/correlation_results_vq.json`` — all rho / r / p / n, global and banded

Usage
-----
    python scripts/evaluate_correlation_3d_vq.py \
        --config configs/vae3d_vq.yaml \
        --checkpoint outputs/checkpoints/vae3d_vqvae_best.pt \
        --split test --n-per-mask 5 --seed 42
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
import matplotlib.pyplot as plt  # noqa: E402

from data import load_config  # noqa: E402
from data.degradations_3d import PRIMITIVES_3D  # noqa: E402
from models.vae3d_vq import ConvVQVAE3D  # noqa: E402
from quality.score import (  # noqa: E402
    DICE_BANDS,
    banded_correlation,
    correlation,
    recon_bce_score,
    recon_dice_score,
    recon_l1_score,
    topology_score,
)
from train import resolve_device, set_seed  # noqa: E402

# Model-agnostic helpers reused verbatim from the standard-VAE 3D evaluator.
import evaluate_correlation_3d as ev3d  # noqa: E402

logger = logging.getLogger("evaluate_correlation_3d_vq")

SCORE_BATCH = ev3d.SCORE_BATCH


# --------------------------------------------------------------------------
# VQ-specific score: quantization distance (the latent_kl analogue)
# --------------------------------------------------------------------------
@torch.no_grad()
def quant_error_score(model: ConvVQVAE3D, masks: torch.Tensor) -> torch.Tensor:
    """Per-mask mean L2^2 distance from the encoder output to its nearest codebook
    entry. A mask the VQ-VAE finds atypical lands far from every learned code, so a
    larger quantization distance = more unusual = worse quality. Direction -1.

    Mirrors the nearest-neighbour computation inside ``VectorQuantizer3D.forward``
    but keeps the per-sample minimum distance instead of collapsing to a scalar
    loss — read-only use of the model's own codebook, no model code is modified.
    """
    z_e, _ = model._encode_features(masks)                 # (B, D, d, h, w)
    b = z_e.shape[0]
    z = z_e.permute(0, 2, 3, 4, 1).contiguous()            # (B, d, h, w, D)
    flat = z.view(-1, model.embedding_dim)                 # (B*d*h*w, D)
    w = model.quantizer.embedding.weight                   # (K, D)
    dist = flat.pow(2).sum(1, keepdim=True) - 2.0 * flat @ w.t() + w.pow(2).sum(1)
    min_d = dist.min(dim=1).values                         # (B*d*h*w,)
    return min_d.view(b, -1).mean(dim=1)                   # (B,)


# recon_dice/bce/l1/topology work on the VQ-VAE unchanged (reconstruct() + the
# 3-tuple forward()); latent_kl is replaced by quant_error.
VQ_SCORE_FNS = {
    "recon_dice": recon_dice_score,
    "recon_bce": recon_bce_score,
    "recon_l1": recon_l1_score,
    "topology": topology_score,
    "quant_error": quant_error_score,
}
SCORE_DIRECTIONS_VQ = {
    "recon_dice": +1,
    "recon_bce": -1,
    "recon_l1": -1,
    "topology": -1,
    "quant_error": -1,
}


def load_model(checkpoint: Path, device: torch.device) -> ConvVQVAE3D:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    v = ckpt["config"]["vae3d_vq"]
    model = ConvVQVAE3D(
        base_channels=int(v["base_channels"]),
        num_embeddings=int(v["num_embeddings"]),
        embedding_dim=int(v["embedding_dim"]),
        commitment_cost=float(v["commitment_cost"]),
        input_size=tuple(v["input_size"]),
        use_skip=bool(v.get("use_skip", False)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    logger.info(
        "loaded %s (ConvVQVAE3D codebook=%dx%d use_skip=%s input_size=%s epoch=%s)",
        checkpoint.name, model.num_embeddings, model.embedding_dim,
        model.use_skip, model.input_size, ckpt.get("epoch"),
    )
    return model


def score_masks_vq(model, masks: np.ndarray, device: torch.device) -> dict[str, np.ndarray]:
    """Run every VQ score over all volumes in batches. ``{name: (M,) array}``."""
    model.eval()
    accum: dict[str, list[np.ndarray]] = {name: [] for name in VQ_SCORE_FNS}
    for start in range(0, len(masks), SCORE_BATCH):
        chunk = masks[start:start + SCORE_BATCH]
        x = torch.from_numpy(chunk).float().unsqueeze(1).to(device)
        for name, fn in VQ_SCORE_FNS.items():
            accum[name].append(fn(model, x).cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in accum.items()}


def summarise_vq(scores: dict[str, np.ndarray], true_dice: np.ndarray) -> dict[str, Any]:
    """Global + banded correlation for each score, with the VQ direction table."""
    result: dict[str, Any] = {}
    for name, values in scores.items():
        result[name] = {
            "direction": SCORE_DIRECTIONS_VQ[name],
            "global": correlation(values, true_dice),
            "bands": banded_correlation(values, true_dice),
        }
    return result


def plot_scatter_vq(
    true_dice: np.ndarray, scores: np.ndarray, is_clean: np.ndarray,
    name: str, stats: dict[str, float], out_path: Path,
) -> None:
    """True Dice vs the best VQ score (local copy so an arbitrary score name never
    KeyErrors the standard-VAE global direction table)."""
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
    ax.plot(true_dice[order], np.polyval(coeffs, true_dice[order]),
            color="black", lw=1.2, label="linear trend")
    direction = "higher = better" if SCORE_DIRECTIONS_VQ[name] > 0 else "higher = worse"
    ax.set_xlabel("true Dice")
    ax.set_ylabel(f"{name} score")
    ax.set_title(
        f"3D VQ-VAE — best score: {name} ({direction})\n"
        f"Spearman rho = {stats['spearman']:.3f}   "
        f"Pearson r = {stats['pearson']:.3f}   n = {stats['n']}"
    )
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Correlate 3D VQ-VAE reconstruction error with true Dice.")
    p.add_argument("--config", default="configs/vae3d_vq.yaml")
    p.add_argument("--checkpoint", default="outputs/checkpoints/vae3d_vqvae_best.pt")
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
    logger.info("data source: %s", config["data"]["data_source"])
    logger.info("degradation primitive set: 3D (%s)", ", ".join(PRIMITIVES_3D))

    set_seed(args.seed)
    device = resolve_device(config["train"]["device"])

    model = load_model(PROJECT_ROOT / args.checkpoint, device)
    # data + degradation reuse the standard-VAE 3D machinery verbatim -> identical
    # protocol (same split, same PRIMITIVES_3D, same calibration, same eval set).
    clean_masks = ev3d.load_clean_volumes(config, args.split, args.limit, args.seed)
    degrader = ev3d.calibrate_degrader(config, clean_masks, args.seed)
    masks, true_dice, is_clean = ev3d.build_eval_set(
        clean_masks, degrader, args.n_per_mask, args.seed
    )

    scores = score_masks_vq(model, masks, device)
    summary = summarise_vq(scores, true_dice)
    ev3d.print_table(summary)  # reads info['direction'] from summary, not the global table

    best = ev3d.best_score_name(summary)
    logger.info("best score by |global rho|: %s", best)
    q = summary["quant_error"]["global"]["spearman"]
    rd = summary["recon_dice"]["global"]["spearman"]
    logger.info("VQ-specific quant_error rho=%.3f vs recon_dice rho=%.3f "
                "(quant_error is a NEW score with no standard-VAE analogue)", q, rd)

    out_dir = PROJECT_ROOT / "outputs"
    plot_scatter_vq(true_dice, scores[best], is_clean, best,
                    summary[best]["global"], out_dir / "correlation_scatter_vq.png")
    ev3d.plot_score_comparison(summary, out_dir / "score_comparison_vq.png")

    results = {
        "checkpoint": str(args.checkpoint),
        "model": "ConvVQVAE3D",
        "split": args.split,
        "n_per_mask": args.n_per_mask,
        "seed": args.seed,
        "n_total": int(len(true_dice)),
        "n_clean": int(is_clean.sum()),
        "n_degraded": int((~is_clean).sum()),
        "n_patients": int(is_clean.sum()),
        "primitive_set": "3D",
        "calibrated_ranges": {name: list(rng) for name, rng in (degrader.type_ranges or {}).items()},
        "best_score": best,
        "scores": summary,
    }
    results_path = out_dir / "correlation_results_vq.json"
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("wrote %s", results_path)


if __name__ == "__main__":
    main()
