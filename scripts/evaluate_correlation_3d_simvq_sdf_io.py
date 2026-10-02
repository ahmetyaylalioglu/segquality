"""Does the 3D **SimVQ SDF-in/SDF-out VQ-VAE**'s reconstruction error track true Dice?

Baseline E1 ("SDF-io'yu SimVQ ile eğitelim"). The SimVQ counterpart of
``scripts/evaluate_correlation_3d_sdf_io.py``: same SDF-in/SDF-out interpretation
(binary -> SDF -> model -> SDF -> binary), so the four reconstruction-side scores
(recon_dice / recon_l1 / topology / sdf_l1) and the whole protocol are identical; the
only quantizer-specific deviation is ``quant_error``, which for SimVQ must be measured
against the EFFECTIVE codebook ``C @ W`` (SimVQ has no ``quantizer.embedding``).

So this wrapper reuses ``evaluate_correlation_3d_sdf_io`` verbatim for everything —
the registry, ``sdf_l1``, ``summarise_sdfio``, ``plot_scatter_sdfio``, the eval-set
build — and overrides exactly two things:

* **model loader** -> ``ConvVQVAE3D_SimVQ_sdf_io`` (from the checkpoint's ``vae3d_vq``
  block + the ``vae3d_vq_sdf_io.delta``).
* **quant_error** -> ``quant_error_score_simvq_sdfio``: encode the SDF input (this
  encoder was trained on ``to_sdf_input(x)``), then nearest-code L2^2 against the
  effective codebook ``model.quantizer.effective_codebook()`` (not ``embedding.weight``).

Non-clobbering: writes ``correlation_results_simvq_sdfio_{tag}.json`` and ``*_simvq_sdfio_{tag}.png``.

Usage
-----
    python scripts/evaluate_correlation_3d_simvq_sdf_io.py \
        --config configs/vae3d_vq_sdf_io.yaml \
        --checkpoint outputs/checkpoints/vae3d_simvq_sdfio_best.pt \
        --tag simvq_sdfio --split test --n-per-mask 5 --seed 42
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
from models.vae3d_simvq_sdf_io import ConvVQVAE3D_SimVQ_sdf_io  # noqa: E402
from quality.sdf_paper import DELTA  # noqa: E402
from train import resolve_device, set_seed  # noqa: E402

import evaluate_correlation_3d as ev3d  # noqa: E402
import evaluate_correlation_3d_sdf_io as evsdfio  # noqa: E402

logger = logging.getLogger("evaluate_correlation_3d_simvq_sdf_io")

SCORE_BATCH = ev3d.SCORE_BATCH


# --------------------------------------------------------------------------
# SimVQ-SDF-IO-specific score: SDF-input encoding, EFFECTIVE codebook C@W
# --------------------------------------------------------------------------
@torch.no_grad()
def quant_error_score_simvq_sdfio(model: ConvVQVAE3D_SimVQ_sdf_io, masks: torch.Tensor) -> torch.Tensor:
    """Per-mask nearest-code L2^2 distance, encoding the SDF INPUT (like SDF-io) and
    comparing against the EFFECTIVE codebook ``C @ W`` (like SimVQ). Direction -1.
    """
    sdf_in = model.to_sdf_input(masks)                     # binary -> normalised SDF
    z_e, _ = model._encode_features(sdf_in)                # (B, D, d, h, w)
    b = z_e.shape[0]
    z = z_e.permute(0, 2, 3, 4, 1).contiguous()            # (B, d, h, w, D)
    flat = z.view(-1, model.embedding_dim)                 # (B*d*h*w, D)
    w = model.quantizer.effective_codebook()               # (K, D) — the projected C@W
    dist = flat.pow(2).sum(1, keepdim=True) - 2.0 * flat @ w.t() + w.pow(2).sum(1)
    min_d = dist.min(dim=1).values                         # (B*d*h*w,)
    return min_d.view(b, -1).mean(dim=1)                   # (B,)


# Same registry as SDF-io (recon_dice/recon_l1/topology/sdf_l1), quant_error swapped
# for the effective-codebook SimVQ version.
SIMVQ_SDFIO_SCORE_FNS = {
    "recon_dice": evsdfio.recon_dice_score,
    "recon_l1": evsdfio.recon_l1_score,
    "topology": evsdfio.topology_score,
    "quant_error": quant_error_score_simvq_sdfio,
    "sdf_l1": evsdfio.sdf_l1_score,
}


def load_simvq_sdfio_model(checkpoint: Path, device: torch.device) -> ConvVQVAE3D_SimVQ_sdf_io:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    v = ckpt["config"]["vae3d_vq"]
    delta = float(ckpt["config"].get("vae3d_vq_sdf_io", {}).get("delta", DELTA))
    model = ConvVQVAE3D_SimVQ_sdf_io(
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
        "loaded %s (ConvVQVAE3D_SimVQ_sdf_io codebook=%dx%d C frozen+W learnable delta=%.2f use_skip=%s epoch=%s)",
        checkpoint.name, model.num_embeddings, model.embedding_dim,
        model.delta, model.use_skip, ckpt.get("epoch"),
    )
    return model


def score_masks_simvq_sdfio(model, masks: np.ndarray, device: torch.device) -> dict[str, np.ndarray]:
    """Run every SimVQ-SDF-IO score over all volumes in batches. ``{name: (M,) array}``."""
    model.eval()
    accum: dict[str, list[np.ndarray]] = {name: [] for name in SIMVQ_SDFIO_SCORE_FNS}
    for start in range(0, len(masks), SCORE_BATCH):
        chunk = masks[start:start + SCORE_BATCH]
        x = torch.from_numpy(chunk).float().unsqueeze(1).to(device)
        for name, fn in SIMVQ_SDFIO_SCORE_FNS.items():
            accum[name].append(fn(model, x).cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in accum.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Correlate 3D SimVQ SDF-IO VQ-VAE reconstruction error with true Dice.")
    p.add_argument("--config", default="configs/vae3d_vq_sdf_io.yaml")
    p.add_argument("--checkpoint", default="outputs/checkpoints/vae3d_simvq_sdfio_best.pt")
    p.add_argument("--tag", default="simvq_sdfio", help="output filename tag")
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
    model = load_simvq_sdfio_model(Path(ckpt_path), device)

    # data + degradation reuse the standard-VAE 3D machinery verbatim -> identical
    # protocol (same split, same PRIMITIVES_3D, same calibration, same eval set).
    clean_masks = ev3d.load_clean_volumes(config, args.split, args.limit, args.seed)
    degrader = ev3d.calibrate_degrader(config, clean_masks, args.seed)
    masks, true_dice, is_clean = ev3d.build_eval_set(clean_masks, degrader, args.n_per_mask, args.seed)

    scores = score_masks_simvq_sdfio(model, masks, device)
    summary = evsdfio.summarise_sdfio(scores, true_dice)
    ev3d.print_table(summary)

    best = ev3d.best_score_name(summary)
    q = summary["quant_error"]["global"]["spearman"]
    rd = summary["recon_dice"]["global"]["spearman"]
    sl = summary["sdf_l1"]["global"]["spearman"]
    logger.info("best score by |global rho|: %s | quant_error rho=%.4f (effective C@W) | recon_dice rho=%.4f | sdf_l1 rho=%.4f",
                best, q, rd, sl)

    out_dir = PROJECT_ROOT / "outputs"
    tag = args.tag
    evsdfio.plot_scatter_sdfio(true_dice, scores[best], is_clean, best,
                               summary[best]["global"], out_dir / f"correlation_scatter_simvq_sdfio_{tag}.png")
    ev3d.plot_score_comparison(summary, out_dir / f"score_comparison_simvq_sdfio_{tag}.png")

    results = {
        "checkpoint": str(args.checkpoint),
        "tag": tag,
        "model": "ConvVQVAE3D_SimVQ_sdf_io",
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
    results_path = out_dir / f"correlation_results_simvq_sdfio_{tag}.json"
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("wrote %s", results_path)


if __name__ == "__main__":
    main()
