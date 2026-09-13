"""WP-4 STEP 0 — candidate-diversity measurement (READ-ONLY diagnosis, no new model
code, nothing trained, nothing written to outputs/).

The WP-4 idea: instead of snapping each of the 4x4x4=64 latent positions to its single
nearest codebook vector (argmin), take the nearest K codes -> several candidate
quantizations -> decode each -> several candidate masks -> distance-weighted consensus +
a 3D uncertainty map. STEP 0 only answers the GO/NO-GO question:

    Are the runner-up codes meaningfully different from the argmin, i.e. do they decode
    to DIFFERENT masks? If Dice(M0, Mr) ~ 1.0 the codebook is over-confident and WP-4
    has nothing to aggregate; if Dice < 1 there is real diversity to exploit.

Metrics (over a few real val masks, using the model's OWN codebook, read-only):
  1. distance competitiveness  — mean d[r]/d[0] for r=1..K-1 (how close the runner-up
                                  codes sit to the winner; ~1 = competitive).
  2. candidate divergence       — build candidate r by snapping EVERY position to its
                                  r-th nearest code, decode, and report Dice(M0, Mr) and
                                  voxel-disagreement %% vs the argmin mask M0.
  3. disagreement localisation  — fraction of disagreeing voxels that fall in the
                                  boundary band of M0 (sanity: uncertainty ~ surface).

Usage
-----
    python scripts/wp4_step0_diversity.py --checkpoint outputs/checkpoints/vae3d_vqvae_best.pt \
        --k 5 --n-masks 8
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data import load_config  # noqa: E402
from data.raos_volume import RAOSVolumes  # noqa: E402
from train import resolve_device, set_seed  # noqa: E402
from evaluate_correlation_3d_vq import load_model  # noqa: E402

logger = logging.getLogger("wp4_step0_diversity")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WP-4 STEP 0 candidate-diversity diagnosis.")
    p.add_argument("--config", default="configs/vae3d_vq.yaml")
    p.add_argument("--checkpoint", default="outputs/checkpoints/vae3d_vqvae_best.pt")
    p.add_argument("--split", default="val")
    p.add_argument("--k", type=int, default=5, help="nearest codes per position to inspect")
    p.add_argument("--n-masks", type=int, default=8, help="how many val masks to probe")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_masks(config: dict, split: str, n: int, seed: int) -> np.ndarray:
    ds = RAOSVolumes.from_config(config, split=split)
    set_seed(seed)
    idx = np.random.default_rng(seed).permutation(len(ds))[: min(n, len(ds))]
    masks = np.stack([np.asarray(ds[int(i)]["mask"], dtype=np.float32) for i in idx])
    return masks


@torch.no_grad()
def topk_indices_and_dists(model, x: torch.Tensor, k: int):
    """Nearest-k codes per latent position (mirrors VectorQuantizer3D, read-only).

    Returns (idx_k (B,P,k), dist_k (B,P,k), grid_shape (d,h,w)) where P=d*h*w."""
    z_e, _ = model._encode_features(x)                 # (B, D, d, h, w)
    b, D, d, h, w = z_e.shape
    z = z_e.permute(0, 2, 3, 4, 1).contiguous().view(b, -1, D)   # (B, P, D)
    W = model.quantizer.embedding.weight               # (K, D)
    # squared L2 distance (B,P,K)
    dist = z.pow(2).sum(-1, keepdim=True) - 2.0 * z @ W.t() + W.pow(2).sum(1)
    dist_k, idx_k = torch.topk(dist, k, dim=-1, largest=False, sorted=True)
    return idx_k, dist_k.clamp_min(0.0), (d, h, w)


@torch.no_grad()
def decode_candidate(model, idx_rank: torch.Tensor, grid, feats_none=None) -> torch.Tensor:
    """Decode the quantization that snaps every position to the given code indices.

    idx_rank: (B, P) code index per position. Returns binary mask (B, 1, X, Y, Z)."""
    d, h, w = grid
    b, P = idx_rank.shape
    z_q_flat = model.quantizer.embedding(idx_rank)     # (B, P, D)
    z_q = z_q_flat.view(b, d, h, w, model.embedding_dim).permute(0, 4, 1, 2, 3).contiguous()
    logits = model._decode_features(z_q, feats=[])     # base model use_skip=False
    return (torch.sigmoid(logits) > 0.5).float()


def dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return float((2.0 * inter + 1.0) / (denom + 1.0))


def boundary_band(mask: np.ndarray, width: int = 2) -> np.ndarray:
    """Voxels within `width` of the surface of `mask` (dilation XOR erosion)."""
    m = mask > 0.5
    er = ndimage.binary_erosion(m, iterations=width)
    di = ndimage.binary_dilation(m, iterations=width)
    return di & ~er


def main() -> None:
    args = parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = load_config(args.config)
    device = resolve_device(str(config["train"]["device"]))
    model = load_model(Path(args.checkpoint), device)
    K = model.num_embeddings
    k = min(args.k, K)

    masks = load_masks(config, args.split, args.n_masks, args.seed)
    x = torch.from_numpy(masks).float().unsqueeze(1).to(device)
    logger.info("WP-4 STEP 0 | %s | %d masks | K=%d | probing nearest %d codes/position",
                Path(args.checkpoint).name, len(masks), K, k)

    idx_k, dist_k, grid = topk_indices_and_dists(model, x, k)   # (B,P,k)
    P = idx_k.shape[1]

    # ---- METRIC 1: distance competitiveness d[r]/d[0] --------------------
    d0 = dist_k[..., 0].clamp_min(1e-8)
    ratios = (dist_k / d0.unsqueeze(-1)).mean(dim=(0, 1))        # (k,)
    logger.info("[M1] mean d[r]/d[0] over %d positions: %s", P,
                ", ".join(f"r{r}={ratios[r].item():.3f}" for r in range(k)))
    # how often is the runner-up within 1.5x of the winner (=> real competition)
    frac_close = (dist_k[..., 1] < 1.5 * d0).float().mean().item() if k > 1 else float("nan")
    logger.info("[M1] positions where 2nd-nearest < 1.5x nearest = %.1f%% (higher = more diverse)",
                100.0 * frac_close)
    # how many DISTINCT codes are actually used by argmin across all positions
    distinct = int(torch.unique(idx_k[..., 0]).numel())
    logger.info("[M1] distinct argmin codes across %d positions = %d / %d", P, distinct, K)

    # ---- METRIC 2 + 3: candidate divergence & localisation ---------------
    m0 = decode_candidate(model, idx_k[..., 0], grid).cpu().numpy()[:, 0]   # argmin masks
    self_dice = np.mean([dice(m0[i] > 0.5, masks[i] > 0.5) for i in range(len(masks))])
    logger.info("[M2] argmin recon self-Dice vs input = %.4f (sanity: matches known ~0.9x)", self_dice)

    for r in range(1, k):
        mr = decode_candidate(model, idx_k[..., r], grid).cpu().numpy()[:, 0]
        dices, disagree, on_band = [], [], []
        for i in range(len(masks)):
            a, b = m0[i] > 0.5, mr[i] > 0.5
            dices.append(dice(a, b))
            diff = np.logical_xor(a, b)
            disagree.append(diff.mean())
            if diff.sum() > 0:
                band = boundary_band(m0[i])
                on_band.append(diff[band].sum() / diff.sum())
        logger.info("[M2] candidate r=%d vs argmin: Dice(M0,Mr)=%.4f | voxel-disagree=%.3f%% | "
                    "[M3] disagreement-in-boundary-band=%.1f%%",
                    r, float(np.mean(dices)), 100.0 * float(np.mean(disagree)),
                    100.0 * float(np.mean(on_band)) if on_band else float("nan"))

    logger.info("──────── STEP 0 VERDICT ────────")
    logger.info("If Dice(M0,Mr) < ~0.98 for r>=1 -> runner-up codes decode to DIFFERENT masks "
                "-> WP-4 has real diversity to aggregate (GO). If ~1.0 -> codebook over-confident "
                "-> distance-weighting would just reproduce argmin (NO-GO / need temperature).")


if __name__ == "__main__":
    main()
