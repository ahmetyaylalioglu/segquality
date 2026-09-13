"""Diagnose 3D VAE reconstruction mean-reversion / mode-collapse (İş 1).

The advisor noticed the 3D reconstruction previews look "column-similar": different
livers seem to reconstruct to a similar generic shape. That could be genuine
mean-reversion (the VAE ignoring its input and always emitting the dataset mean),
or it could be a *visual illusion*: the squashed cube's centre slices are broadly
similar across patients, and soft/blurry probability maps look alike to the eye.

This script measures which it is, honestly, with numbers rather than opinion. It
touches NO training/model code — it only *reads* a trained checkpoint.

Three measurements
------------------
1. **Cross-reconstruction Dice matrix.** ``M[i][j] = Dice(binarised recon_i,
   input_j)``. The diagonal is each liver reconstructed against *its own* input;
   the off-diagonal is a recon against *other* livers' inputs. If reconstruction
   is input-specific, the diagonal should sit clearly above the off-diagonal, so
   ``ratio = diag_mean / offdiag_mean > 1``. If the VAE emits the same shape for
   everyone, diag ~= offdiag and ``ratio ~= 1``.

2. **Latent diversity.** Mean pairwise L2 distance between posterior means
   ``mu``; per-dimension std of ``mu`` across the K livers (dims with ~0 std are
   "dead" — collapsed/unused). A collapsed latent is the mechanism behind
   mean-reversion. We also relate each input-pair's Dice to that pair's latent
   distance (similar inputs should sit closer in latent space).

3. **Input-pairwise vs recon-pairwise Dice (control).** If reconstructions are
   *more* similar to each other than the inputs are to each other
   (``recon_pairwise_dice > input_pairwise_dice``), the VAE is actively reducing
   diversity — a clear mean-reversion sign. If they are comparable, the recons
   simply inherit the inputs' own similarity (innocent).

A labelled orthogonal-slice preview (patient id + that liver's self-recon Dice per
row) is also written so the "column-similar" claim can be inspected directly.

Interpretation printed at the end
---------------------------------
* ``ratio`` clearly > ~1.3  -> reconstruction is input-specific, NO mean-reversion
  (the visual similarity is the squashed-cube / soft-probability illusion).
* ``ratio`` ~= 1.0          -> mean-reversion is REAL (recon ignores the input).

Usage
-----
    python scripts/diagnose_reconstruction.py \
        --checkpoint outputs/checkpoints/vae3d_best_latent64.pt \
        --config configs/vae3d.yaml --k 20
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import load_config  # noqa: E402
from data.degradations import dice  # noqa: E402
from data.raos_volume import RAOSVolumes  # noqa: E402
from models.vae3d import ConvVAE3D  # noqa: E402

logger = logging.getLogger("diagnose_reconstruction")

DEAD_DIM_STD = 0.01  # a latent dim whose std across livers is below this is "dead"


# --------------------------------------------------------------------------
# model / data loading (read-only; mirrors evaluate_correlation_3d.load_model)
# --------------------------------------------------------------------------
def load_model(checkpoint: Path, device: torch.device) -> ConvVAE3D:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    vcfg = ckpt["config"]["vae3d"]
    model = ConvVAE3D(
        latent_dim=int(vcfg["latent_dim"]),
        base_channels=int(vcfg["base_channels"]),
        input_size=tuple(vcfg["input_size"]),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    logger.info(
        "loaded %s | ConvVAE3D latent=%d input_size=%s (epoch %s)",
        checkpoint.name, model.latent_dim, model.input_size, ckpt.get("epoch"),
    )
    return model


def select_indices(n: int, k: int, seed: int) -> list[int]:
    """Deterministic sorted selection of ``k`` of ``n`` items (shared by the
    standalone diagnosis and the in-training diagnosis so both pick the *same*
    livers for a given (n, k, seed) — a precondition for apples-to-apples ``ratio``
    comparison between the standard VAE and the VQ-VAE)."""
    if k > n:
        raise ValueError(f"requested K={k} but only {n} available")
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(n, size=k, replace=False).tolist())


def load_val_volumes(config: dict, model: ConvVAE3D, split: str, k: int, seed: int):
    """Return (masks (K,D,H,W) float32, patient_ids list) from the precomputed split."""
    # The checkpoint's input_size is authoritative; make the dataset resize match it
    # so model and data can never silently disagree.
    config = json.loads(json.dumps(config))  # cheap deep copy
    config.setdefault("data", {})["data_source"] = "precomputed"
    config["vae3d"]["input_size"] = list(model.input_size)
    ds = RAOSVolumes.from_config(config, split=split, group="train")
    n = len(ds)
    idx = select_indices(n, k, seed)
    masks = np.stack([np.asarray(ds[i]["mask"], dtype=np.float32) for i in idx])
    pids = [str(ds[i]["patient_id"]) for i in idx]
    logger.info("selected %d/%d val livers (seed=%d): %s", k, n, seed, ", ".join(pids))
    return masks, pids


# --------------------------------------------------------------------------
# measurements
# --------------------------------------------------------------------------
def _pairwise_mean_dice(binaries: np.ndarray) -> float:
    """Mean Dice over all unordered off-diagonal pairs of a (K,...) bool stack."""
    k = len(binaries)
    vals = [dice(binaries[i], binaries[j]) for i in range(k) for j in range(i + 1, k)]
    return float(np.mean(vals)) if vals else float("nan")


def cross_recon_matrix(recon_bin: np.ndarray, inputs_bin: np.ndarray) -> np.ndarray:
    """M[i][j] = Dice(recon_i, input_j)."""
    k = len(recon_bin)
    m = np.empty((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            m[i, j] = dice(recon_bin[i], inputs_bin[j])
    return m


def meanreversion_stats(recon_bin: np.ndarray, inputs_bin: np.ndarray) -> dict:
    """The core mean-reversion numbers, from binarised recons + inputs.

    Single source of truth for the cross-reconstruction ``ratio`` and the
    input-/recon-pairwise Dice, called by BOTH the standalone diagnosis and the
    in-training diagnosis in ``train_vae3d_vq.py``. Because both paths use this
    exact function (same :func:`dice`, same matrix), the VQ-VAE's ``ratio`` is
    directly comparable to the standard VAE's 1.005.
    """
    m = cross_recon_matrix(recon_bin, inputs_bin)
    diag = np.diag(m)
    off = m[~np.eye(len(m), dtype=bool)]
    diag_mean, off_mean = float(diag.mean()), float(off.mean())
    ratio = diag_mean / off_mean if off_mean > 0 else float("inf")
    return {
        "matrix": m,
        "self_recon_dice": diag,
        "diag_mean": diag_mean,
        "offdiag_mean": off_mean,
        "ratio": ratio,
        "input_pairwise_dice": _pairwise_mean_dice(inputs_bin),
        "recon_pairwise_dice": _pairwise_mean_dice(recon_bin),
    }


def latent_stats(mu: np.ndarray, inputs_bin: np.ndarray) -> dict:
    """Latent diversity + how input similarity relates to latent distance."""
    k = len(mu)
    # mean pairwise L2 between posterior means
    pair_l2, pair_input_dice = [], []
    for i in range(k):
        for j in range(i + 1, k):
            pair_l2.append(float(np.linalg.norm(mu[i] - mu[j])))
            pair_input_dice.append(dice(inputs_bin[i], inputs_bin[j]))
    per_dim_std = mu.std(axis=0)
    dead = int((per_dim_std < DEAD_DIM_STD).sum())
    # More-similar inputs should be closer in latent space -> negative correlation
    # between input-pair Dice and latent-pair distance.
    if len(pair_l2) >= 2 and np.std(pair_l2) > 0:
        rho, pval = spearmanr(pair_input_dice, pair_l2)
    else:
        rho, pval = float("nan"), float("nan")
    return {
        "mean_latent_distance": float(np.mean(pair_l2)),
        "latent_std_per_dim": per_dim_std.tolist(),
        "latent_std_mean": float(per_dim_std.mean()),
        "latent_std_min": float(per_dim_std.min()),
        "latent_std_max": float(per_dim_std.max()),
        "dead_dims": dead,
        "n_dims": int(mu.shape[1]),
        "inputdice_vs_latentdist_spearman": float(rho),
        "inputdice_vs_latentdist_pval": float(pval),
    }


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def plot_matrix(m: np.ndarray, pids: list[str], out_path: Path) -> None:
    k = len(pids)
    fig, ax = plt.subplots(figsize=(1.0 + 0.5 * k, 1.0 + 0.5 * k))
    im = ax.imshow(m, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels(pids, rotation=90, fontsize=6)
    ax.set_yticklabels(pids, fontsize=6)
    ax.set_xlabel("input j"); ax.set_ylabel("reconstruction i")
    ax.set_title("Cross-reconstruction Dice  M[i,j]=Dice(recon_i, input_j)\n"
                 "bright diagonal = input-specific reconstruction", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def _ortho(vol: np.ndarray) -> list[np.ndarray]:
    """Axial / coronal / sagittal centre slices (mirrors train_vae3d.save_recon_preview_3d)."""
    d, h, w = vol.shape
    return [vol[d // 2, :, :], vol[:, h // 2, :], vol[:, :, w // 2]]


def plot_labeled_preview(
    inputs: np.ndarray, recon: np.ndarray, pids: list[str],
    self_dice: np.ndarray, out_path: Path,
) -> None:
    """One row per liver: [in axial|coronal|sagittal | rec axial|coronal|sagittal],
    each row labelled with the patient id and that liver's self-reconstruction Dice."""
    n = len(pids)
    col_titles = ["in-axial", "in-coronal", "in-sagittal",
                  "rec-axial", "rec-coronal", "rec-sagittal"]
    fig, axes = plt.subplots(n, 6, figsize=(12, 2 * n))
    axes = np.atleast_2d(axes)
    for row in range(n):
        panels = _ortho(inputs[row]) + _ortho(recon[row])
        for col, img in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(col_titles[col], fontsize=8)
        axes[row, 0].set_ylabel(f"{pids[row]}\nDice={self_dice[row]:.3f}", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", out_path)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose 3D VAE reconstruction mean-reversion.")
    p.add_argument("--checkpoint", default="outputs/checkpoints/vae3d_best_latent64.pt")
    p.add_argument("--config", default="configs/vae3d.yaml")
    p.add_argument("--split", default="val")
    p.add_argument("--k", type=int, default=20, help="number of clean livers to diagnose")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    model = load_model(PROJECT_ROOT / args.checkpoint, device)
    masks, pids = load_val_volumes(config, model, args.split, args.k, args.seed)

    # single deterministic forward pass (eval -> reparameterize returns mu)
    x = torch.from_numpy(masks).unsqueeze(1).to(device)
    with torch.no_grad():
        probs = model.reconstruct(x).cpu().numpy()[:, 0]
        mu = model._encode_dist(x)[0].cpu().numpy()
    inputs_bin = masks > 0.5
    recon_bin = probs > 0.5

    # (1) + (3) cross-reconstruction matrix and input-/recon-pairwise Dice via the
    # shared aggregator (same function the VQ-VAE training loop calls).
    mr = meanreversion_stats(recon_bin, inputs_bin)
    m = mr["matrix"]
    diag = mr["self_recon_dice"]
    diag_mean, off_mean, ratio = mr["diag_mean"], mr["offdiag_mean"], mr["ratio"]
    input_pairwise = mr["input_pairwise_dice"]
    recon_pairwise = mr["recon_pairwise_dice"]

    # (2) latent diversity
    lat = latent_stats(mu, inputs_bin)

    # figures
    plot_matrix(m, pids, out_dir / "cross_recon_matrix.png")
    plot_labeled_preview(masks, probs, pids, diag, out_dir / "recon_labeled.png")

    mean_reversion = ratio < 1.15  # decision threshold (see interpretation below)
    result = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "k": args.k,
        "seed": args.seed,
        "patient_ids": pids,
        "self_recon_dice": diag.tolist(),
        "self_recon_dice_mean": diag_mean,
        "diag_mean": diag_mean,
        "offdiag_mean": off_mean,
        "ratio": ratio,
        "input_pairwise_dice": input_pairwise,
        "recon_pairwise_dice": recon_pairwise,
        "latent": lat,
        "mean_reversion_detected": bool(mean_reversion),
    }
    out_json = out_dir / "diagnose_reconstruction.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("wrote %s", out_json)

    # -- human-readable report ---------------------------------------------
    print("\n" + "=" * 72)
    print("3D VAE RECONSTRUCTION DIAGNOSIS")
    print("=" * 72)
    print(f"livers (K)                : {args.k}  from split '{args.split}' (seed {args.seed})")
    print(f"self-recon Dice (mean)    : {diag_mean:.4f}  "
          f"[min {diag.min():.3f}, max {diag.max():.3f}]")
    print("\n(1) CROSS-RECONSTRUCTION Dice matrix")
    print(f"    diagonal  mean (recon_i vs own input_i)   : {diag_mean:.4f}")
    print(f"    off-diag  mean (recon_i vs other inputs)  : {off_mean:.4f}")
    print(f"    ratio = diag/offdiag                      : {ratio:.4f}")
    print("\n(2) LATENT diversity")
    print(f"    mean pairwise L2 between mu               : {lat['mean_latent_distance']:.4f}")
    print(f"    per-dim std: mean {lat['latent_std_mean']:.4f} "
          f"(min {lat['latent_std_min']:.4f}, max {lat['latent_std_max']:.4f})")
    print(f"    dead dims (std < {DEAD_DIM_STD}): {lat['dead_dims']} / {lat['n_dims']}")
    print(f"    input-Dice vs latent-distance Spearman    : "
          f"{lat['inputdice_vs_latentdist_spearman']:+.3f} "
          f"(p={lat['inputdice_vs_latentdist_pval']:.2g})  "
          f"[negative = similar inputs sit closer in latent]")
    print("\n(3) INPUT-pairwise vs RECON-pairwise Dice (mean-reversion control)")
    print(f"    input  pairwise Dice (inputs vs each other): {input_pairwise:.4f}")
    print(f"    recon  pairwise Dice (recons vs each other): {recon_pairwise:.4f}")
    delta = recon_pairwise - input_pairwise
    print(f"    delta (recon - input)                      : {delta:+.4f}  "
          f"[strongly positive = VAE reduces diversity]")
    print("\n" + "-" * 72)
    print("VERDICT")
    print("-" * 72)
    if mean_reversion:
        print(f"  ratio = {ratio:.3f} ~= 1  ->  MEAN-REVERSION LIKELY REAL.")
        print("  The reconstruction reconstructs other livers about as well as its")
        print("  own; the latent is not steering the output enough. Levers: larger")
        print("  latent, weaker KL (lower beta), less aggressive squash, augmentation.")
    else:
        print(f"  ratio = {ratio:.3f} > 1  ->  reconstruction is INPUT-SPECIFIC; NO")
        print("  mean-reversion. Each liver reconstructs its OWN input markedly better")
        print("  than others'. The 'column-similar' look is the squashed-cube / soft-")
        print("  probability visual illusion (similar centre slices + blurry maps),")
        print("  NOT the VAE emitting one generic shape.")
    if recon_pairwise > input_pairwise + 0.05:
        print(f"  NOTE: recons ARE more similar to each other than inputs are "
              f"(delta {delta:+.3f}); some diversity reduction is present even if the")
        print("  diagonal still dominates.")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
