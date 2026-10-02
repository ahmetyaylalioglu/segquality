"""Phase 12: does the 3D **SimVQ-VAE**'s reconstruction error track true Dice?

The SimVQ counterpart of ``scripts/evaluate_correlation_3d_vq.py``. SimVQ cures
codebook collapse (frozen Gaussian base ``C`` + one learnable linear ``W``, so a
single gradient step moves the entire effective codebook ``C @ W`` jointly),
lifting utilization from ~1-3% to tens of percent. The open question this script
answers under the byte-identical exam: does that near-full, non-collapsed codebook
help, hurt, or leave the QC signal unchanged versus vanilla VQ (base rho 0.862)
and the commit-variant ablation?

This is a THIN WRAPPER. It imports the model-agnostic evaluation machinery from
``evaluate_correlation_3d`` unchanged (data loading, degrader calibration, eval-set
construction, console table) and the VQ scoring/summary/plot machinery from
``evaluate_correlation_3d_vq`` unchanged. It overrides only two things:

* **model loader** -> ``ConvVQVAE3D_SimVQ`` (from the checkpoint's ``vae3d_vq`` config).
* **quant_error score** -> the imported ``quant_error_score`` reads
  ``model.quantizer.embedding.weight``, which SimVQ does NOT have (its codebook is
  the frozen base ``C`` = ``quantizer.codebook`` projected by ``W`` = ``quantizer.proj``).
  It is replaced by ``quant_error_score_simvq``, which computes the SAME per-mask
  nearest-code L2^2 distance but against the EFFECTIVE codebook ``C @ W`` (what the
  encoder is actually quantized against). The other four scores (recon_dice,
  recon_bce, recon_l1, topology) call ``reconstruct``/``forward`` inherited verbatim,
  so they run unchanged and rho stays directly comparable.

Non-clobbering: writes ``correlation_results_simvq_{tag}.json`` and ``*_simvq_{tag}.png``.

Usage
-----
    python scripts/evaluate_correlation_3d_simvq.py \
        --config configs/vae3d_simvq.yaml \
        --checkpoint outputs/checkpoints/vae3d_simvq_best.pt \
        --tag simvq --split test --n-per-mask 5 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data import load_config  # noqa: E402
from data.degradations_3d import PRIMITIVES_3D  # noqa: E402
from models.vae3d_simvq import ConvVQVAE3D_SimVQ  # noqa: E402
from quality.score import (  # noqa: E402
    recon_bce_score,
    recon_dice_score,
    recon_l1_score,
    topology_score,
)
from train import resolve_device, set_seed  # noqa: E402

import evaluate_correlation_3d as ev3d  # noqa: E402
import evaluate_correlation_3d_vq as ev3dvq  # noqa: E402

logger = logging.getLogger("evaluate_correlation_3d_simvq")

SCORE_BATCH = ev3d.SCORE_BATCH


# --------------------------------------------------------------------------
# SimVQ-specific score: quantization distance against the EFFECTIVE codebook C@W
# --------------------------------------------------------------------------
@torch.no_grad()
def quant_error_score_simvq(model: ConvVQVAE3D_SimVQ, masks: torch.Tensor) -> torch.Tensor:
    """Per-mask mean L2^2 distance from the encoder output to its nearest EFFECTIVE
    code (``C @ W``). A mask SimVQ finds atypical lands far from every learned code,
    so a larger distance = more unusual = worse quality. Direction -1.

    Identical to ``ev3dvq.quant_error_score`` except the codebook it compares against
    is ``model.quantizer.effective_codebook()`` (SimVQ has no ``embedding.weight``);
    read-only use of the model's own codebook, no model code is modified.
    """
    z_e, _ = model._encode_features(masks)                 # (B, D, d, h, w)
    b = z_e.shape[0]
    z = z_e.permute(0, 2, 3, 4, 1).contiguous()            # (B, d, h, w, D)
    flat = z.view(-1, model.embedding_dim)                 # (B*d*h*w, D)
    w = model.quantizer.effective_codebook()               # (K, D) — the projected C@W
    dist = flat.pow(2).sum(1, keepdim=True) - 2.0 * flat @ w.t() + w.pow(2).sum(1)
    min_d = dist.min(dim=1).values                         # (B*d*h*w,)
    return min_d.view(b, -1).mean(dim=1)                   # (B,)


# Same four reconstruction scores as VQ (inherited encoder/decoder), with
# quant_error swapped for the effective-codebook version.
SIMVQ_SCORE_FNS = {
    "recon_dice": recon_dice_score,
    "recon_bce": recon_bce_score,
    "recon_l1": recon_l1_score,
    "topology": topology_score,
    "quant_error": quant_error_score_simvq,
}


def load_simvq_model(checkpoint: Path, device: torch.device) -> ConvVQVAE3D_SimVQ:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    v = ckpt["config"]["vae3d_vq"]
    model = ConvVQVAE3D_SimVQ(
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
        "loaded %s (ConvVQVAE3D_SimVQ codebook=%dx%d C frozen+W learnable use_skip=%s epoch=%s)",
        checkpoint.name, model.num_embeddings, model.embedding_dim,
        model.use_skip, ckpt.get("epoch"),
    )
    return model


def score_masks_simvq(model, masks: np.ndarray, device: torch.device) -> dict[str, np.ndarray]:
    """Run every SimVQ score over all volumes in batches. ``{name: (M,) array}``."""
    model.eval()
    accum: dict[str, list[np.ndarray]] = {name: [] for name in SIMVQ_SCORE_FNS}
    for start in range(0, len(masks), SCORE_BATCH):
        chunk = masks[start:start + SCORE_BATCH]
        x = torch.from_numpy(chunk).float().unsqueeze(1).to(device)
        for name, fn in SIMVQ_SCORE_FNS.items():
            accum[name].append(fn(model, x).cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in accum.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Correlate 3D SimVQ-VAE reconstruction error with true Dice.")
    p.add_argument("--config", default="configs/vae3d_simvq.yaml")
    p.add_argument("--checkpoint", default="outputs/checkpoints/vae3d_simvq_best.pt")
    p.add_argument("--tag", default="simvq", help="output filename tag")
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
    model = load_simvq_model(Path(ckpt_path), device)

    # data + degradation reuse the standard-VAE 3D machinery verbatim -> identical
    # protocol (same split, same PRIMITIVES_3D, same calibration, same eval set).
    clean_masks = ev3d.load_clean_volumes(config, args.split, args.limit, args.seed)
    degrader = ev3d.calibrate_degrader(config, clean_masks, args.seed)
    masks, true_dice, is_clean = ev3d.build_eval_set(clean_masks, degrader, args.n_per_mask, args.seed)

    scores = score_masks_simvq(model, masks, device)
    summary = ev3dvq.summarise_vq(scores, true_dice)  # direction table shares the same names
    ev3d.print_table(summary)

    best = ev3d.best_score_name(summary)
    q = summary["quant_error"]["global"]["spearman"]
    rd = summary["recon_dice"]["global"]["spearman"]
    logger.info("best score by |global rho|: %s | quant_error rho=%.4f (effective C@W) vs recon_dice rho=%.4f",
                best, q, rd)

    out_dir = PROJECT_ROOT / "outputs"
    tag = args.tag
    ev3dvq.plot_scatter_vq(true_dice, scores[best], is_clean, best,
                           summary[best]["global"], out_dir / f"correlation_scatter_simvq_{tag}.png")
    ev3d.plot_score_comparison(summary, out_dir / f"score_comparison_simvq_{tag}.png")

    results = {
        "checkpoint": str(args.checkpoint),
        "tag": tag,
        "model": "ConvVQVAE3D_SimVQ",
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
        "scores": summary,
    }
    results_path = out_dir / f"correlation_results_simvq_{tag}.json"
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("wrote %s", results_path)


if __name__ == "__main__":
    main()
