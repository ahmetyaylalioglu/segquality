"""WP-4 — distance-weighted multi-candidate consensus + 3D uncertainty map (INFERENCE
ONLY; no retraining, no existing file edited, uses the trained base VQ-VAE's own codebook).

Idea
----
STEP 0 proved the runner-up codes decode to genuinely different masks (Dice(M0,M1)=0.79).
Here we exploit that. For each of the P=4x4x4 latent positions we take the nearest
``k`` codes; rank ``r`` gives a full candidate quantization -> decode -> candidate
probability map ``p_r``. The advisor's distance-weighted average combines them:

    P_consensus = Σ_r w_r · p_r ,   w_r = (1/d_r) / Σ_j (1/d_j)

where ``d_r`` is the candidate's mean nearest-code distance (closer candidate = more
trusted). The consensus mask is ``P_consensus > 0.5``; the 3D **uncertainty** map is the
per-voxel binary entropy ``H(P) = -P·logP - (1-P)·log(1-P)`` (0 = all candidates agree,
max at P=0.5 = candidates split).

Validation (does uncertainty mark the REAL errors?)
---------------------------------------------------
On the canonical eval set (clean liver + synthetic degradations, so the clean shape is a
known reference), the true error region of an input is ``input XOR reference``:
  * voxel-level: AUROC of uncertainty vs the error mask (subsampled, degraded vols only).
  * volume-level: Spearman(mean uncertainty, 1 - true_dice) — worse masks = more unsure.
Plus an ortho figure (input | consensus | uncertainty | error) for a few volumes.

Usage
-----
    python scripts/multi_candidate_wp4.py --checkpoint outputs/checkpoints/vae3d_vqvae_best.pt \
        --k 5 --split test --n-per-mask 2 --figure figures/wp4/wp4_uncertainty.png
"""
from __future__ import annotations

import argparse
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
from train import resolve_device, set_seed  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
import evaluate_correlation_3d as ev3d  # noqa: E402
from evaluate_correlation_3d_vq import load_model  # noqa: E402

logger = logging.getLogger("multi_candidate_wp4")
EPS = 1e-6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WP-4 multi-candidate consensus + uncertainty.")
    p.add_argument("--config", default="configs/vae3d_vq.yaml")
    p.add_argument("--checkpoint", default="outputs/checkpoints/vae3d_vqvae_best.pt")
    p.add_argument("--split", default="test")
    p.add_argument("--k", type=int, default=5, help="candidates (nearest codes) per position")
    p.add_argument("--n-per-mask", type=int, default=2)
    p.add_argument("--limit", type=int, default=None, help="cap clean volumes (smoke)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--figure", default="figures/wp4/wp4_uncertainty.png")
    p.add_argument("--fig-n", type=int, default=6,
                   help="degraded figure rows; each row is a DIFFERENT patient (one degraded vol per block)")
    p.add_argument("--fig-clean", type=int, default=2,
                   help="clean (un-degraded) rows shown FIRST as a contrast (uncertainty should be low)")
    p.add_argument("--voxel-sample", type=int, default=200_000,
                   help="max error/non-error voxels to pool for the AUROC")
    return p.parse_args()


# --------------------------------------------------------------------------
# WP-4 core: consensus probability + entropy uncertainty (read-only on model)
# --------------------------------------------------------------------------
@torch.no_grad()
def consensus_and_uncertainty(model, x: torch.Tensor, k: int):
    """Return (P_consensus (B,X,Y,Z), uncertainty (B,X,Y,Z), cand_std (B,X,Y,Z)).

    ``P_consensus`` is the distance-weighted blend of the k candidate probability maps;
    ``uncertainty`` is its per-voxel binary entropy; ``cand_std`` is the weighted std
    across candidates (a second, model-internal disagreement signal)."""
    z_e, _ = model._encode_features(x)                       # (B, D, d, h, w)
    b, D, d, h, w = z_e.shape
    z = z_e.permute(0, 2, 3, 4, 1).contiguous().view(b, -1, D)   # (B, P, D)
    W = model.quantizer.embedding.weight                      # (K, D)
    dist = z.pow(2).sum(-1, keepdim=True) - 2.0 * z @ W.t() + W.pow(2).sum(1)
    dist_k, idx_k = torch.topk(dist, k, dim=-1, largest=False, sorted=True)  # (B,P,k)
    dist_k = dist_k.clamp_min(0.0)

    # candidate weight w_r = (1/d_r) / Σ (1/d_j), d_r = mean nearest-code dist of cand r
    d_r = dist_k.mean(dim=1) + EPS                            # (B, k)
    inv = 1.0 / d_r
    cw = inv / inv.sum(dim=1, keepdim=True)                   # (B, k) candidate weights

    P = None
    probs = []
    for r in range(k):
        z_q_flat = model.quantizer.embedding(idx_k[..., r])  # (B, P, D)
        z_q = z_q_flat.view(b, d, h, w, D).permute(0, 4, 1, 2, 3).contiguous()
        p_r = torch.sigmoid(model._decode_features(z_q, feats=[]))[:, 0]  # (B,X,Y,Z)
        probs.append(p_r)
        wr = cw[:, r].view(b, 1, 1, 1)
        P = p_r * wr if P is None else P + p_r * wr
    P = P.clamp(EPS, 1.0 - EPS)
    uncertainty = -(P * P.log() + (1.0 - P) * (1.0 - P).log())  # binary entropy (nats)

    stack = torch.stack(probs, dim=0)                         # (k,B,X,Y,Z)
    wexp = cw.t().view(k, b, 1, 1, 1)
    mean = (stack * wexp).sum(0)
    cand_std = ((stack - mean).pow(2) * wexp).sum(0).sqrt()
    return P, uncertainty, cand_std


# --------------------------------------------------------------------------
# validation helpers
# --------------------------------------------------------------------------
def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via the Mann-Whitney U rank statistic (no sklearn dependency)."""
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    allv = np.concatenate([pos, neg])
    sorter = np.argsort(allv, kind="mergesort")
    sv = allv[sorter]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[sorter[i:j + 1]] = np.mean(ranks[sorter[i:j + 1]])
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def save_ortho_figure(items: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(items)
    fig, axes = plt.subplots(n, 5, figsize=(15, 3 * n))
    if n == 1:
        axes = axes[None, :]
    titles = ["input (axial mid)", "ground truth (ref)", "consensus (P>0.5)",
              "uncertainty (entropy)", "error = input XOR ref"]
    # Fix the uncertainty colour scale across ALL rows (binary entropy maxes at ln 2 nats),
    # so a clean mask's low uncertainty truly looks darker than a degraded one's — otherwise
    # per-panel auto-scaling would make every row equally bright and hide the contrast.
    umax = float(np.log(2.0))
    for row, it in enumerate(items):
        z = it["input"].shape[0] // 2
        panels = [it["input"][z], it["reference"][z], it["consensus"][z],
                  it["uncertainty"][z], it["error"][z]]
        cmaps = ["gray", "gray", "gray", "magma", "Reds"]
        for col, (panel, cm) in enumerate(zip(panels, cmaps)):
            ax = axes[row, col]
            if col == 3:                                   # uncertainty: shared 0..ln2 scale
                ax.imshow(panel, cmap=cm, vmin=0.0, vmax=umax)
            else:
                ax.imshow(panel, cmap=cm)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(titles[col], fontsize=10)
        tag = "CLEAN" if it.get("clean") else "DEGRADED"
        axes[row, 0].set_ylabel(f"{tag}\nD={it['true_dice']:.2f}", fontsize=9)
    fig.suptitle("WP-4 distance-weighted consensus + uncertainty "
                 "(clean contrast rows first, then degraded)", fontsize=12, y=1.0)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved figure -> %s", path)


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
    k = min(args.k, model.num_embeddings)

    # ---- eval set (clean + degraded); references recovered from block structure ----
    set_seed(args.seed)
    clean = ev3d.load_clean_volumes(config, args.split, args.limit, args.seed)
    degrader = ev3d.calibrate_degrader(config, clean, args.seed)
    masks, true_dice, is_clean = ev3d.build_eval_set(clean, degrader, args.n_per_mask, args.seed)
    block = 1 + args.n_per_mask   # each clean mask is followed by n_per_mask degradations
    logger.info("WP-4 | %s | %d volumes | k=%d candidates/position", Path(args.checkpoint).name,
                len(masks), k)

    # ---- run WP-4 over all volumes ----
    vol_uncert = np.zeros(len(masks), dtype=np.float64)   # mean uncertainty per volume
    vox_scores: list[np.ndarray] = []                     # pooled uncertainty (degraded)
    vox_labels: list[np.ndarray] = []                     # pooled error mask (degraded)
    fig_items: list[dict] = []                            # degraded rows
    fig_blocks: set[int] = set()                          # blocks (patients) already in the figure
    fig_clean: list[dict] = []                            # clean (un-degraded) contrast rows
    for start in range(0, len(masks), args.batch):
        chunk = masks[start:start + args.batch]
        x = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(1).to(device)
        P, U, _ = consensus_and_uncertainty(model, x, k)
        Pn = P.cpu().numpy()
        Un = U.cpu().numpy()
        for j in range(len(chunk)):
            gi = start + j
            vol_uncert[gi] = float(Un[j].mean())
            blk = gi // block                             # block index == patient index
            ref = masks[blk * block]                      # clean reference (ground truth) of this block
            err = np.logical_xor(chunk[j].astype(bool), ref.astype(bool))
            if not is_clean[gi]:
                vox_scores.append(Un[j].ravel())
                vox_labels.append(err.ravel())
                # one row per DIFFERENT patient: skip a block already shown in the figure
                if blk not in fig_blocks and len(fig_items) < args.fig_n and err.sum() > 0:
                    fig_blocks.add(blk)
                    fig_items.append({
                        "input": chunk[j].astype(np.float32),
                        "reference": ref.astype(np.float32),
                        "consensus": (Pn[j] > 0.5).astype(np.float32),
                        "uncertainty": Un[j],
                        "error": err.astype(np.float32),
                        "true_dice": float(true_dice[gi]),
                    })
            elif len(fig_clean) < args.fig_clean:
                # clean (un-degraded) contrast: input == reference, so error is empty and
                # uncertainty should be low (only baseline boundary) -> shows the system
                # does NOT flag a good mask.
                fig_clean.append({
                    "input": chunk[j].astype(np.float32),
                    "reference": ref.astype(np.float32),
                    "consensus": (Pn[j] > 0.5).astype(np.float32),
                    "uncertainty": Un[j],
                    "error": err.astype(np.float32),
                    "true_dice": float(true_dice[gi]),
                    "clean": True,
                })

    # ---- VALIDATION 1: voxel-level AUROC (uncertainty predicts the error region) ----
    scores = np.concatenate(vox_scores)
    labels = np.concatenate(vox_labels)
    # subsample to keep the AUROC tractable while preserving prevalence
    rng = np.random.default_rng(args.seed)
    pos_idx = np.where(labels)[0]
    neg_idx = np.where(~labels)[0]
    cap = args.voxel_sample // 2
    if len(pos_idx) > cap:
        pos_idx = rng.choice(pos_idx, cap, replace=False)
    if len(neg_idx) > cap:
        neg_idx = rng.choice(neg_idx, cap, replace=False)
    sel = np.concatenate([pos_idx, neg_idx])
    vox_auroc = auroc(scores[sel], labels[sel])
    logger.info("[VALID 1] voxel AUROC(uncertainty -> input XOR reference) = %.4f "
                "(0.5 = chance; >0.7 = uncertainty localises real errors) | pooled %d err / %d ok voxels",
                vox_auroc, int(labels.sum()), int((~labels).size - labels.sum()))

    # ---- VALIDATION 2: volume-level Spearman(mean uncertainty, 1 - true_dice) ----
    err_level = 1.0 - true_dice
    rho_all = spearmanr(vol_uncert, err_level).statistic
    deg = ~is_clean
    rho_deg = spearmanr(vol_uncert[deg], err_level[deg]).statistic
    logger.info("[VALID 2] Spearman(mean uncertainty, 1-true_Dice): all=%.4f | degraded-only=%.4f "
                "(positive = more corrupt -> more uncertain)", float(rho_all), float(rho_deg))

    # ---- figure (clean contrast rows first, then degraded) ----
    fig_rows = fig_clean + fig_items
    if fig_rows:
        logger.info("──────── FIGURE ROW METRICS ────────")
        logger.info("%-4s %-9s %-10s %-14s %-12s", "row", "kind", "true_Dice",
                    "consensus_Dice", "mean_uncert")
        for i, it in enumerate(fig_rows):
            cons = it["consensus"].astype(bool)
            ref = it["reference"].astype(bool)
            inter = np.logical_and(cons, ref).sum()
            cdice = 2.0 * inter / (cons.sum() + ref.sum() + EPS)
            kind = "CLEAN" if it.get("clean") else "DEGRADED"
            logger.info("%-4d %-9s %-10.4f %-14.4f %-12.5f", i + 1, kind,
                        it["true_dice"], cdice, float(it["uncertainty"].mean()))
        save_ortho_figure(fig_rows, PROJECT_ROOT / args.figure)

    logger.info("──────── WP-4 SUMMARY ────────")
    logger.info("voxel AUROC=%.4f | volume rho(all)=%.4f rho(deg)=%.4f | k=%d | n=%d",
                vox_auroc, float(rho_all), float(rho_deg), k, len(masks))


if __name__ == "__main__":
    main()
