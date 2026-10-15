"""Does the 3D **VAR-style tokenizer**'s reconstruction error track a volume's true Dice?

The VAR counterpart of ``scripts/evaluate_correlation_3d_vq.py``. The only
architectural difference vs the plain VQ-VAE is the quantizer (single-scale VQ ->
multi-scale residual VQ); encoder/decoder and the whole eval protocol are identical,
so this is a THIN WRAPPER that reuses ``evaluate_correlation_3d`` (data loading,
degrader calibration, eval-set construction, console table) and ``quality.score``
(recon_dice/bce/l1/topology) verbatim. It overrides only two things:

* **model loader** -> ``VARVQVAE3D`` (from the checkpoint's ``vae3d_var`` config).
* **VQ-specific score** -> ``quant_error``: per-mask mean squared residual between the
  encoder output ``f`` and the multi-scale reconstruction ``f_hat``. A mask the model
  finds atypical cannot be reconstructed from the coarse-to-fine code sum, so it leaves
  a larger residual = more unusual = worse quality. Direction -1. This is the direct
  multi-scale analogue of the plain VQ-VAE's nearest-code distance.

Because the shared machinery is untouched, the existing tests stay green.

Usage
-----
    python scripts/evaluate_correlation_3d_var.py \
        --config configs/vae3d_var.yaml \
        --checkpoint outputs/checkpoints/vae3d_var_best.pt \
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

from data import load_config  # noqa: E402
from data.degradations_3d import PRIMITIVES_3D  # noqa: E402
from models.vae3d_var import VARVQVAE3D  # noqa: E402
from quality.score import (  # noqa: E402
    recon_bce_score,
    recon_dice_score,
    recon_l1_score,
    topology_score,
    banded_correlation,
    correlation,
)
from train import resolve_device, set_seed  # noqa: E402

# Model-agnostic helpers reused verbatim from the standard-VAE 3D evaluator.
import evaluate_correlation_3d as ev3d  # noqa: E402
# Reuse the VQ wrapper's scatter plot (it reads SCORE_DIRECTIONS_VQ, which contains
# quant_error with direction -1 -- identical to ours).
from evaluate_correlation_3d_vq import plot_scatter_vq  # noqa: E402

logger = logging.getLogger("evaluate_correlation_3d_var")

SCORE_BATCH = ev3d.SCORE_BATCH


@torch.no_grad()
def quant_error_score(model: VARVQVAE3D, masks: torch.Tensor) -> torch.Tensor:
    """Per-mask mean squared residual ``||f - f_hat||^2`` between the encoder output
    and the multi-scale reconstruction. The VAR analogue of the plain VQ-VAE's
    nearest-code distance: an atypical mask cannot be rebuilt from the coarse-to-fine
    code sum, leaving a larger residual. Direction -1 (larger = worse)."""
    f = model._encode_features(masks)                      # (B, C, d, h, w)
    f_hat, _, _ = model.quantizer(f)                       # (B, C, d, h, w)
    resid = (f - f_hat).pow(2)                             # straight-through residual
    return resid.flatten(start_dim=1).mean(dim=1)          # (B,)


VAR_SCORE_FNS = {
    "recon_dice": recon_dice_score,
    "recon_bce": recon_bce_score,
    "recon_l1": recon_l1_score,
    "topology": topology_score,
    "quant_error": quant_error_score,
}
SCORE_DIRECTIONS_VAR = {
    "recon_dice": +1,
    "recon_bce": -1,
    "recon_l1": -1,
    "topology": -1,
    "quant_error": -1,
}


def load_model(checkpoint: Path, device: torch.device) -> VARVQVAE3D:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    v = ckpt["config"]["vae3d_var"]
    model = VARVQVAE3D(
        base_channels=int(v["base_channels"]),
        num_embeddings=int(v["num_embeddings"]),
        embedding_dim=int(v["embedding_dim"]),
        input_size=tuple(v["input_size"]),
        v_patch_nums=tuple(v["v_patch_nums"]),
        using_znorm=bool(v.get("using_znorm", False)),
        beta=float(v.get("beta", 0.25)),
        quant_resi=float(v.get("quant_resi", 0.5)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    logger.info(
        "loaded %s (VARVQVAE3D codebook=%dx%d scales=%s znorm=%s input_size=%s epoch=%s)",
        checkpoint.name, model.num_embeddings, model.embedding_dim,
        model.v_patch_nums, v.get("using_znorm", False), model.input_size, ckpt.get("epoch"),
    )
    return model


def score_masks_var(model, masks: np.ndarray, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    accum: dict[str, list[np.ndarray]] = {name: [] for name in VAR_SCORE_FNS}
    for start in range(0, len(masks), SCORE_BATCH):
        chunk = masks[start:start + SCORE_BATCH]
        x = torch.from_numpy(chunk).float().unsqueeze(1).to(device)
        for name, fn in VAR_SCORE_FNS.items():
            accum[name].append(fn(model, x).cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in accum.items()}


def summarise_var(scores: dict[str, np.ndarray], true_dice: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, values in scores.items():
        result[name] = {
            "direction": SCORE_DIRECTIONS_VAR[name],
            "global": correlation(values, true_dice),
            "bands": banded_correlation(values, true_dice),
        }
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Correlate 3D VAR tokenizer reconstruction error with true Dice.")
    p.add_argument("--config", default="configs/vae3d_var.yaml")
    p.add_argument("--checkpoint", default="outputs/checkpoints/vae3d_var_best.pt")
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
    clean_masks = ev3d.load_clean_volumes(config, args.split, args.limit, args.seed)
    degrader = ev3d.calibrate_degrader(config, clean_masks, args.seed)
    masks, true_dice, is_clean = ev3d.build_eval_set(
        clean_masks, degrader, args.n_per_mask, args.seed
    )

    scores = score_masks_var(model, masks, device)
    summary = summarise_var(scores, true_dice)
    ev3d.print_table(summary)

    best = ev3d.best_score_name(summary)
    logger.info("best score by |global rho|: %s", best)
    q = summary["quant_error"]["global"]["spearman"]
    rd = summary["recon_dice"]["global"]["spearman"]
    logger.info("VAR quant_error rho=%.3f vs recon_dice rho=%.3f", q, rd)

    out_dir = PROJECT_ROOT / "outputs"
    plot_scatter_vq(true_dice, scores[best], is_clean, best,
                    summary[best]["global"], out_dir / "correlation_scatter_var.png")
    ev3d.plot_score_comparison(summary, out_dir / "score_comparison_var.png")

    results = {
        "checkpoint": str(args.checkpoint),
        "model": "VARVQVAE3D",
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
    results_path = out_dir / "correlation_results_var.json"
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("wrote %s", results_path)


if __name__ == "__main__":
    main()
