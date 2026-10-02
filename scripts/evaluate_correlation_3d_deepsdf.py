"""Stage 2: does the DeepSDF auto-decoder's reconstruction error track true Dice?

Same exam as every other 3D evaluator (``evaluate_correlation_3d`` data loading,
degrader calibration, eval-set construction, correlation table, per-band stats) so
the DeepSDF rho is DIRECTLY comparable to the VQ line (base 0.862, SDF-VQ 0.844).

The one thing that differs is scoring. DeepSDF has no encoder: a mask is "reconstructed"
by **latent inference** — freeze the trained decoder, optimise a fresh latent ``z``
(from zero) so ``f(z, xyz)`` fits the mask's truncated target SDF, then the binary
bridge ``sdf < 0 -> inside`` turns the fitted field back into a mask. Several scores fall
out of that fit and go through the identical ``correlation`` / ``print_table`` machinery:

* ``latent_norm`` (-1) — the norm ``||z*||`` of the inferred latent. An atypical /
  degraded mask must push its latent far from the learned shape prior to be fit, so a
  larger norm means a less plausible shape. This is DeepSDF's native atypicality
  measure and — unlike ``recon_dice`` — it is NOT defeated by the decoder's
  expressiveness, so it is the strongest QC signal here.
* ``fit_residual`` (-1) — mean |fitted - target| SDF error over the near-surface band
  (the far field is trivially fit, so a full-grid residual is diluted). The DeepSDF
  analogue of the VQ ``quant_error``.
* ``recon_dice`` (+1) — Dice(reconstruction, input). Kept for the honest comparison:
  because a continuous decoder can fit almost any blob, this collapses as a QC score
  (the same "best reconstruction != best QC" identity trap seen in the VQ ablation).

Non-clobbering: writes ``correlation_results_deepsdf_{tag}.json`` and ``*_deepsdf_{tag}.png``.
Nothing existing is edited, so the test suite stays green.

Usage
-----
    python scripts/evaluate_correlation_3d_deepsdf.py \
        --checkpoint outputs/checkpoints/deepsdf_deepsdf_best.pt --tag deepsdf \
        --split test --n-per-mask 3 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data import load_config  # noqa: E402
from data.degradations_3d import PRIMITIVES_3D  # noqa: E402
from data.sdf_points import SDFPointSampler  # noqa: E402
from models.deepsdf import DeepSDF  # noqa: E402
from quality.score import banded_correlation, correlation  # noqa: E402
from quality.sdf import TRUNCATION  # noqa: E402
from train import resolve_device, set_seed  # noqa: E402

import evaluate_correlation_3d as ev3d  # noqa: E402

logger = logging.getLogger("evaluate_correlation_3d_deepsdf")

SCORE_DIRECTIONS_DEEPSDF = {"latent_norm": -1, "fit_residual": -1, "recon_dice": +1}
SURFACE_BAND = 2.0  # |target SDF| <= band defines the near-surface residual region


def load_deepsdf(checkpoint: Path, device: torch.device) -> tuple[DeepSDF, dict[str, Any]]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    d = ckpt["config"]["deepsdf"]
    model = DeepSDF(
        num_shapes=int(model_num_shapes(ckpt)),
        latent_dim=int(d["latent_dim"]),
        hidden_dim=int(d["hidden_dim"]),
        num_layers=int(d["num_layers"]),
        skip_layers=tuple(int(s) for s in d["skip_layers"]),
        dropout=float(d["dropout"]),
        latent_init_std=float(d["latent_init_std"]),
        weight_norm=bool(d["weight_norm"]),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    logger.info("loaded %s (DeepSDF latent=%d MLP=%dx%d epoch=%s)",
                checkpoint.name, model.latent_dim, model.hidden_dim,
                model.num_layers, ckpt.get("epoch"))
    return model, d


def model_num_shapes(ckpt: dict[str, Any]) -> int:
    """Recover the training ``num_shapes`` from the saved latent-embedding row count."""
    return int(ckpt["model_state"]["latents.weight"].shape[0])


def infer_latents(
    model: DeepSDF, sampler: SDFPointSampler, indices: list[int], device: torch.device,
    *, steps: int, lr: float, reg: float, n_points: int, near_frac: float,
    latent_dim: int, truncation: float,
) -> torch.Tensor:
    """Optimise one fresh latent per mask in ``indices`` against its target SDF.

    Decoder weights are frozen; only the ``(len(indices), L)`` latent block is a leaf
    parameter. Points are re-sampled (near-surface weighted) every step, exactly like
    training, so the fit is over the same distribution the decoder learned.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    z = torch.zeros(len(indices), latent_dim, device=device, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)
    pos = {s: i for i, s in enumerate(indices)}     # shape id -> row in z
    for _ in range(steps):
        idx, coords, tgt = sampler.sample(indices, n_points, near_frac)
        rows = torch.tensor([pos[int(s)] for s in idx.tolist()], device=device)
        coords = coords.to(device)
        tgt = tgt.to(device).clamp(-truncation, truncation)
        pred = model.decode(z[rows], coords)
        recon = F.smooth_l1_loss(pred.clamp(-truncation, truncation), tgt)
        loss = recon + reg * (z * z).sum(dim=-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return z.detach()


@torch.no_grad()
def decode_scores(
    model: DeepSDF, sampler: SDFPointSampler, indices: list[int], z: torch.Tensor,
    device: torch.device, truncation: float, chunk: int = 65536,
) -> tuple[np.ndarray, np.ndarray]:
    """Full-grid decode each fitted latent -> (recon_dice, near-surface fit_residual)."""
    coords = sampler.coords.to(device)
    dices, residuals = [], []
    for row, s in enumerate(indices):
        preds = []
        zc = z[row:row + 1]
        for start in range(0, coords.shape[0], chunk):
            c = coords[start:start + chunk]
            preds.append(model.decode(zc.expand(c.shape[0], -1), c))
        sdf = torch.cat(preds).clamp(-truncation, truncation)
        target = sampler.sdf_flat[s].to(device).clamp(-truncation, truncation)
        # near-surface residual: the far field is trivially fit and dilutes a full-grid mean
        band = target.abs() <= SURFACE_BAND
        residuals.append((sdf[band] - target[band]).abs().mean().item())
        recon = (sdf < 0)
        tgt_bin = (target < 0)
        inter = (recon & tgt_bin).sum().float()
        denom = recon.sum().float() + tgt_bin.sum().float()
        dices.append(((2.0 * inter + 1.0) / (denom + 1.0)).item())
    return np.asarray(dices), np.asarray(residuals)


def score_masks_deepsdf(
    model: DeepSDF, masks: np.ndarray, device: torch.device, d: dict[str, Any],
    *, steps: int, lr: float, reg: float, n_points: int, near_frac: float, batch: int,
) -> dict[str, np.ndarray]:
    """Latent-infer every mask (in batches) and read off recon_dice + fit_residual."""
    truncation = float(d.get("truncation", TRUNCATION))
    sampler = SDFPointSampler([m.astype(np.float32) for m in masks],
                              truncation=truncation, surface_band=float(d["surface_band"]))
    latent_dim = int(d["latent_dim"])
    all_dice = np.empty(len(masks), dtype=np.float64)
    all_res = np.empty(len(masks), dtype=np.float64)
    all_norm = np.empty(len(masks), dtype=np.float64)
    t0 = time.perf_counter()
    for start in range(0, len(masks), batch):
        idx = list(range(start, min(start + batch, len(masks))))
        z = infer_latents(model, sampler, idx, device, steps=steps, lr=lr, reg=reg,
                          n_points=n_points, near_frac=near_frac,
                          latent_dim=latent_dim, truncation=truncation)
        dice, res = decode_scores(model, sampler, idx, z, device, truncation)
        all_dice[idx] = dice
        all_res[idx] = res
        all_norm[idx] = z.norm(dim=-1).cpu().numpy()
        logger.info("  scored %d/%d masks (%.0fs)", idx[-1] + 1, len(masks), time.perf_counter() - t0)
    return {"latent_norm": all_norm, "fit_residual": all_res, "recon_dice": all_dice}


def summarise_deepsdf(scores: dict[str, np.ndarray], true_dice: np.ndarray) -> dict[str, Any]:
    return {
        name: {
            "direction": SCORE_DIRECTIONS_DEEPSDF[name],
            "global": correlation(values, true_dice),
            "bands": banded_correlation(values, true_dice),
        }
        for name, values in scores.items()
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Correlate DeepSDF reconstruction error with true Dice.")
    p.add_argument("--config", default="configs/deepsdf.yaml")
    p.add_argument("--checkpoint", default="outputs/checkpoints/deepsdf_deepsdf_best.pt")
    p.add_argument("--tag", default="deepsdf", help="output filename tag")
    p.add_argument("--split", default="test")
    p.add_argument("--n-per-mask", type=int, default=3)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--data-source", choices=("nifti", "precomputed"), default="precomputed")
    p.add_argument("--seed", type=int, default=42)
    # latent-inference hyper-parameters
    p.add_argument("--infer-steps", type=int, default=200)
    p.add_argument("--infer-lr", type=float, default=0.01)
    p.add_argument("--infer-reg", type=float, default=1e-4)
    p.add_argument("--infer-points", type=int, default=2048)
    p.add_argument("--infer-near-frac", type=float, default=0.5)
    p.add_argument("--infer-batch", type=int, default=16)
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
    model, dcfg = load_deepsdf(Path(ckpt_path), device)

    clean_masks = ev3d.load_clean_volumes(config, args.split, args.limit, args.seed)
    degrader = ev3d.calibrate_degrader(config, clean_masks, args.seed)
    masks, true_dice, is_clean = ev3d.build_eval_set(clean_masks, degrader, args.n_per_mask, args.seed)

    logger.info("latent inference: steps=%d lr=%g reg=%g points=%d batch=%d over %d masks",
                args.infer_steps, args.infer_lr, args.infer_reg, args.infer_points,
                args.infer_batch, len(masks))
    scores = score_masks_deepsdf(
        model, masks, device, dcfg, steps=args.infer_steps, lr=args.infer_lr,
        reg=args.infer_reg, n_points=args.infer_points, near_frac=args.infer_near_frac,
        batch=args.infer_batch,
    )
    summary = summarise_deepsdf(scores, true_dice)
    ev3d.print_table(summary)

    best = ev3d.best_score_name(summary)
    ln = summary["latent_norm"]["global"]["spearman"]
    fr = summary["fit_residual"]["global"]["spearman"]
    rd = summary["recon_dice"]["global"]["spearman"]
    logger.info("best score by |global rho|: %s | latent_norm rho=%.4f | fit_residual rho=%.4f | recon_dice rho=%.4f",
                best, ln, fr, rd)

    out_dir = PROJECT_ROOT / "outputs"
    ev3d.plot_score_comparison(summary, out_dir / f"score_comparison_deepsdf_{args.tag}.png")

    results = {
        "checkpoint": str(args.checkpoint),
        "tag": args.tag,
        "model": "DeepSDF",
        "split": args.split,
        "n_per_mask": args.n_per_mask,
        "seed": args.seed,
        "n_total": int(len(true_dice)),
        "n_clean": int(is_clean.sum()),
        "n_degraded": int((~is_clean).sum()),
        "primitive_set": "3D",
        "inference": {
            "steps": args.infer_steps, "lr": args.infer_lr, "reg": args.infer_reg,
            "points": args.infer_points, "near_frac": args.infer_near_frac,
        },
        "calibrated_ranges": {name: list(rng) for name, rng in (degrader.type_ranges or {}).items()},
        "best_score": best,
        "latent_norm_rho": float(ln),
        "fit_residual_rho": float(fr),
        "recon_dice_rho": float(rd),
        "scores": summary,
    }
    results_path = out_dir / f"correlation_results_deepsdf_{args.tag}.json"
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("wrote %s", results_path)


if __name__ == "__main__":
    main()
