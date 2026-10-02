"""Phase 12: does the 3D **SDF-input/SDF-output VQ-VAE**'s reconstruction error track true Dice?

The SDF-IO counterpart of ``scripts/evaluate_correlation_3d_vq.py`` (advisor request).
WP-3 (``vae3d_vq_sdf``) fed a *binary* input and regressed an SDF; here the model both
CONSUMES and PREDICTS the normalised truncated SDF (binary -> SDF -> model -> SDF ->
binary, per Bogensperger et al. eq. 1). The question this script answers under the
byte-identical exam: with the input on the smooth SDF manifold too, does the QC signal
improve, degrade, or hold versus vanilla VQ (quant_error rho 0.862) and SimVQ
(recon_dice 0.843 / quant_error -0.255)?

This is a THIN WRAPPER. It imports the model-agnostic evaluation machinery from
``evaluate_correlation_3d`` unchanged (data loading, degrader calibration, eval-set
construction, console table) and the score/plot helpers from
``evaluate_correlation_3d_vq`` where they apply. SDF-IO needs three deviations, all
because the model's encoder input and decoder output are the *continuous SDF*, not a
binary mask / probability:

* **model loader** -> ``ConvVQVAE3D_sdf_io`` (from the checkpoint's ``vae3d_vq`` block +
  the ``vae3d_vq_sdf_io.delta``).
* **quant_error** -> the base ``quant_error_score`` encodes the *binary* mask, but this
  encoder was trained on the SDF. ``quant_error_score_sdfio`` encodes
  ``model.to_sdf_input(x)`` first, then measures the SAME nearest-code L2^2 distance
  against the (real, inherited) codebook ``embedding.weight``.
* **recon_bce dropped, sdf_l1 added** -> BCE-with-logits is meaningless on a raw SDF
  output; the natural reconstruction error of an SDF autoencoder is the L1 in SDF space,
  ``mean|predict_sdf(x) - to_sdf_input(x)|`` (``sdf_l1``). ``recon_dice`` / ``recon_l1`` /
  ``topology`` route through the overridden ``reconstruct`` (SDF bridge) and so run
  unchanged and stay directly comparable.

Non-clobbering: writes ``correlation_results_sdfio_{tag}.json`` and ``*_sdfio_{tag}.png``.

Usage
-----
    python scripts/evaluate_correlation_3d_sdf_io.py \
        --config configs/vae3d_vq_sdf_io.yaml \
        --checkpoint outputs/checkpoints/vae3d_sdfio_best.pt \
        --tag sdfio --split test --n-per-mask 5 --seed 42
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
from models.vae3d_vq_sdf_io import ConvVQVAE3D_sdf_io  # noqa: E402
from quality.score import (  # noqa: E402
    DICE_BANDS,
    banded_correlation,
    correlation,
    recon_dice_score,
    recon_l1_score,
    topology_score,
)
from quality.sdf_paper import DELTA  # noqa: E402
from train import resolve_device, set_seed  # noqa: E402

import evaluate_correlation_3d as ev3d  # noqa: E402

logger = logging.getLogger("evaluate_correlation_3d_sdf_io")

SCORE_BATCH = ev3d.SCORE_BATCH


# --------------------------------------------------------------------------
# SDF-IO-specific scores
# --------------------------------------------------------------------------
@torch.no_grad()
def quant_error_score_sdfio(model: ConvVQVAE3D_sdf_io, masks: torch.Tensor) -> torch.Tensor:
    """Per-mask nearest-code L2^2 distance, but encoding the SDF INPUT (not the binary
    mask): this encoder was trained on ``to_sdf_input(x)``, so scoring must feed it the
    same field. Codebook is the real inherited ``embedding.weight``. Direction -1.
    """
    sdf_in = model.to_sdf_input(masks)                     # binary -> normalised SDF
    z_e, _ = model._encode_features(sdf_in)                # (B, D, d, h, w)
    b = z_e.shape[0]
    z = z_e.permute(0, 2, 3, 4, 1).contiguous()            # (B, d, h, w, D)
    flat = z.view(-1, model.embedding_dim)                 # (B*d*h*w, D)
    w = model.quantizer.embedding.weight                   # (K, D)
    dist = flat.pow(2).sum(1, keepdim=True) - 2.0 * flat @ w.t() + w.pow(2).sum(1)
    min_d = dist.min(dim=1).values                         # (B*d*h*w,)
    return min_d.view(b, -1).mean(dim=1)                   # (B,)


@torch.no_grad()
def sdf_l1_score(model: ConvVQVAE3D_sdf_io, masks: torch.Tensor) -> torch.Tensor:
    """Per-mask mean ``|predict_sdf(x) - to_sdf_input(x)|`` — the reconstruction error
    in SDF space (the natural error of an SDF autoencoder). A mask the model cannot
    reproduce on the smooth field lands with a larger SDF residual. Direction -1.
    """
    pred_sdf = model.predict_sdf(masks)
    tgt_sdf = model.to_sdf_input(masks)
    dims = tuple(range(1, masks.dim()))
    return (pred_sdf - tgt_sdf).abs().mean(dim=dims)


# recon_dice/l1/topology route through the SDF-bridge ``reconstruct`` unchanged;
# recon_bce is dropped (no probabilistic output); quant_error encodes the SDF;
# sdf_l1 is the SDF-space reconstruction error.
SDFIO_SCORE_FNS = {
    "recon_dice": recon_dice_score,
    "recon_l1": recon_l1_score,
    "topology": topology_score,
    "quant_error": quant_error_score_sdfio,
    "sdf_l1": sdf_l1_score,
}
SCORE_DIRECTIONS_SDFIO = {
    "recon_dice": +1,
    "recon_l1": -1,
    "topology": -1,
    "quant_error": -1,
    "sdf_l1": -1,
}


def load_sdfio_model(checkpoint: Path, device: torch.device) -> ConvVQVAE3D_sdf_io:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    v = ckpt["config"]["vae3d_vq"]
    delta = float(ckpt["config"].get("vae3d_vq_sdf_io", {}).get("delta", DELTA))
    model = ConvVQVAE3D_sdf_io(
        base_channels=int(v["base_channels"]),
        num_embeddings=int(v["num_embeddings"]),
        embedding_dim=int(v["embedding_dim"]),
        commitment_cost=float(v["commitment_cost"]),
        input_size=tuple(v["input_size"]),
        use_skip=bool(v.get("use_skip", False)),
        delta=delta,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    logger.info(
        "loaded %s (ConvVQVAE3D_sdf_io codebook=%dx%d delta=%.2f use_skip=%s epoch=%s)",
        checkpoint.name, model.num_embeddings, model.embedding_dim,
        model.delta, model.use_skip, ckpt.get("epoch"),
    )
    return model


def score_masks_sdfio(model, masks: np.ndarray, device: torch.device) -> dict[str, np.ndarray]:
    """Run every SDF-IO score over all volumes in batches. ``{name: (M,) array}``."""
    model.eval()
    accum: dict[str, list[np.ndarray]] = {name: [] for name in SDFIO_SCORE_FNS}
    for start in range(0, len(masks), SCORE_BATCH):
        chunk = masks[start:start + SCORE_BATCH]
        x = torch.from_numpy(chunk).float().unsqueeze(1).to(device)
        for name, fn in SDFIO_SCORE_FNS.items():
            accum[name].append(fn(model, x).cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in accum.items()}


def summarise_sdfio(scores: dict[str, np.ndarray], true_dice: np.ndarray) -> dict[str, Any]:
    """Global + banded correlation for each score, with the SDF-IO direction table."""
    result: dict[str, Any] = {}
    for name, values in scores.items():
        result[name] = {
            "direction": SCORE_DIRECTIONS_SDFIO[name],
            "global": correlation(values, true_dice),
            "bands": banded_correlation(values, true_dice),
        }
    return result


def plot_scatter_sdfio(
    true_dice: np.ndarray, scores: np.ndarray, is_clean: np.ndarray,
    name: str, stats: dict[str, float], out_path: Path,
) -> None:
    """True Dice vs the best SDF-IO score (local copy so ``sdf_l1`` — a name absent from
    the VQ direction table — never KeyErrors)."""
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
    direction = "higher = better" if SCORE_DIRECTIONS_SDFIO[name] > 0 else "higher = worse"
    ax.set_xlabel("true Dice")
    ax.set_ylabel(f"{name} score")
    ax.set_title(
        f"3D SDF-IO VQ-VAE — best score: {name} ({direction})\n"
        f"Spearman rho = {stats['spearman']:.3f}   "
        f"Pearson r = {stats['pearson']:.3f}   n = {stats['n']}"
    )
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Correlate 3D SDF-IO VQ-VAE reconstruction error with true Dice.")
    p.add_argument("--config", default="configs/vae3d_vq_sdf_io.yaml")
    p.add_argument("--checkpoint", default="outputs/checkpoints/vae3d_sdfio_best.pt")
    p.add_argument("--tag", default="sdfio", help="output filename tag")
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
    logger.info("data source: %s | primitives: 3D (%s)", config["data"]["data_source"],
                ", ".join(PRIMITIVES_3D))

    set_seed(args.seed)
    device = resolve_device(config["train"]["device"])

    ckpt_path = args.checkpoint
    if not Path(ckpt_path).is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    model = load_sdfio_model(Path(ckpt_path), device)

    # data + degradation reuse the standard-VAE 3D machinery verbatim -> identical
    # protocol (same split, same PRIMITIVES_3D, same calibration, same eval set).
    clean_masks = ev3d.load_clean_volumes(config, args.split, args.limit, args.seed)
    degrader = ev3d.calibrate_degrader(config, clean_masks, args.seed)
    masks, true_dice, is_clean = ev3d.build_eval_set(clean_masks, degrader, args.n_per_mask, args.seed)

    scores = score_masks_sdfio(model, masks, device)
    summary = summarise_sdfio(scores, true_dice)
    ev3d.print_table(summary)

    best = ev3d.best_score_name(summary)
    q = summary["quant_error"]["global"]["spearman"]
    rd = summary["recon_dice"]["global"]["spearman"]
    sl = summary["sdf_l1"]["global"]["spearman"]
    logger.info("best score by |global rho|: %s | quant_error rho=%.4f | recon_dice rho=%.4f | sdf_l1 rho=%.4f",
                best, q, rd, sl)

    out_dir = PROJECT_ROOT / "outputs"
    tag = args.tag
    plot_scatter_sdfio(true_dice, scores[best], is_clean, best,
                       summary[best]["global"], out_dir / f"correlation_scatter_sdfio_{tag}.png")
    ev3d.plot_score_comparison(summary, out_dir / f"score_comparison_sdfio_{tag}.png")

    results = {
        "checkpoint": str(args.checkpoint),
        "tag": tag,
        "model": "ConvVQVAE3D_sdf_io",
        "delta": float(model.delta),
        "split": args.split,
        "n_per_mask": args.n_per_mask,
        "seed": args.seed,
        "n_total": int(len(true_dice)),
        "n_clean": int(is_clean.sum()),
        "n_degraded": int((~is_clean).sum()),
        "primitive_set": "3D",
        "calibrated_ranges": {name: list(rng) for name, rng in (degrader.type_ranges or {}).items()},
        "best_score": best,
        "quant_error_rho": float(q),
        "recon_dice_rho": float(rd),
        "sdf_l1_rho": float(sl),
        "scores": summary,
    }
    results_path = out_dir / f"correlation_results_sdfio_{tag}.json"
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("wrote %s", results_path)


if __name__ == "__main__":
    main()
