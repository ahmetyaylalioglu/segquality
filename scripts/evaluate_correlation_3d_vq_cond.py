"""Correlate the CONDITIONED 3D VQ-VAE's scores with true Dice (WP-2 evaluator).

Identical protocol to ``evaluate_correlation_3d_vq.py`` — it imports that script's
scoring/summary/plot machinery and the standard-VAE data/degrader/eval-set helpers
verbatim, so the exam (split, PRIMITIVES_3D, calibration, seed) is byte-identical to
every other 3D evaluation. The ONLY thing it overrides is the model loader:
``evaluate_correlation_3d_vq.load_model`` builds a plain ``ConvVQVAE3D`` and cannot
restore the conditioning MLP / standardisation buffers, so this file rebuilds a
:class:`ConvVQVAE3D_cond` from the checkpoint's embedded ``vae3d_vq_cond`` spec.

Because the conditioned model derives its geometric features INSIDE ``forward`` from
the input mask, the imported ``score_masks_vq`` (which only ever passes the mask)
scores it unchanged — no score function is touched.

Non-clobbering: writes ``correlation_results_vq_cond_{tag}.json`` and the two
``*_vq_cond_{tag}.png`` figures, so the canonical ``*_vq.*`` outputs are never
overwritten and each of the three WP-2 runs keeps its own results.

Usage
-----
    python scripts/evaluate_correlation_3d_vq_cond.py \
        --checkpoint outputs/checkpoints/vae3d_wp2_full_best.pt \
        --tag wp2_full --split test --n-per-mask 5 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data import load_config  # noqa: E402
from data.degradations_3d import PRIMITIVES_3D  # noqa: E402
from models.vae3d_vq_cond import ConvVQVAE3D_cond  # noqa: E402
from train import resolve_device, set_seed  # noqa: E402

import evaluate_correlation_3d as ev3d  # noqa: E402
import evaluate_correlation_3d_vq as ev3dvq  # noqa: E402

logger = logging.getLogger("evaluate_correlation_3d_vq_cond")


def load_cond_model(checkpoint: Path, device: torch.device) -> ConvVQVAE3D_cond:
    """Rebuild the conditioned model from the checkpoint's embedded spec and restore
    its weights + standardisation buffers. Falls back to cond_dim=0 (a no-op, i.e. a
    plain VQ-VAE) when no ``vae3d_vq_cond`` block is present, so a baseline checkpoint
    trained by the cond-trainer also loads here."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    v = ckpt["config"]["vae3d_vq"]
    c = ckpt["config"].get("vae3d_vq_cond", {})
    cond_dim = int(c.get("cond_dim", 0))
    feature_names = list(c.get("feature_names", []))
    model = ConvVQVAE3D_cond(
        base_channels=int(v["base_channels"]),
        num_embeddings=int(v["num_embeddings"]),
        embedding_dim=int(v["embedding_dim"]),
        commitment_cost=float(v["commitment_cost"]),
        input_size=tuple(v["input_size"]),
        use_skip=bool(v.get("use_skip", False)),
        cond_dim=cond_dim,
        feature_names=feature_names,
        cond_hidden=int(c.get("cond_hidden", 32)),
    )
    model.load_state_dict(ckpt["model_state"])  # also restores feat_mean/feat_std buffers
    model.to(device).eval()
    logger.info(
        "loaded %s (ConvVQVAE3D_cond codebook=%dx%d cond_dim=%d features=%s epoch=%s)",
        checkpoint.name, model.num_embeddings, model.embedding_dim,
        cond_dim, feature_names or "-", ckpt.get("epoch"),
    )
    if cond_dim > 0:
        logger.info("restored feature stats | mean=%s | std=%s",
                    [round(float(m), 3) for m in model.feat_mean],
                    [round(float(s), 3) for s in model.feat_std])
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Correlate conditioned 3D VQ-VAE scores with true Dice.")
    p.add_argument("--config", default="configs/vae3d_vq.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tag", default="cond", help="output filename tag (per WP-2 run)")
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
    model = load_cond_model(Path(ckpt_path), device)

    # SAME exam as every other 3D evaluation (reused verbatim).
    clean_masks = ev3d.load_clean_volumes(config, args.split, args.limit, args.seed)
    degrader = ev3d.calibrate_degrader(config, clean_masks, args.seed)
    masks, true_dice, is_clean = ev3d.build_eval_set(clean_masks, degrader, args.n_per_mask, args.seed)

    scores = ev3dvq.score_masks_vq(model, masks, device)
    summary = ev3dvq.summarise_vq(scores, true_dice)
    ev3d.print_table(summary)

    best = ev3d.best_score_name(summary)
    rd = summary["recon_dice"]["global"]["spearman"]
    logger.info("best score by |global rho|: %s | recon_dice rho=%.4f", best, rd)

    out_dir = PROJECT_ROOT / "outputs"
    tag = args.tag
    ev3dvq.plot_scatter_vq(true_dice, scores[best], is_clean, best,
                           summary[best]["global"], out_dir / f"correlation_scatter_vq_cond_{tag}.png")
    ev3d.plot_score_comparison(summary, out_dir / f"score_comparison_vq_cond_{tag}.png")

    results = {
        "checkpoint": str(args.checkpoint),
        "tag": tag,
        "model": "ConvVQVAE3D_cond",
        "cond": {"cond_dim": model.cond_dim, "feature_names": model.feature_names,
                 "cond_hidden": model.cond_hidden},
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
    results_path = out_dir / f"correlation_results_vq_cond_{tag}.json"
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("wrote %s", results_path)


if __name__ == "__main__":
    main()
