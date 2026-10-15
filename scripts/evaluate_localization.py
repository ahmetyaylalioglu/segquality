"""Voxel-level error-LOCALIZATION benchmark (Part 1 of the diffusion workstream).

The global Dice-correlation (est_dice ρ) answers "is this mask good or bad?". The advisor also
wants to *mark* WHERE the mask is wrong. That is a voxel-level localization task, and global ρ
does not measure it. This script scores the per-voxel error maps that the EXISTING trained
models already produce, against the true error map (``broken XOR clean``), using voxel-AUROC and
average precision (PR-AUC, robust to the heavy foreground/background imbalance of error voxels).

Three localizers are compared on the SAME eval set (seed 42, n_per_mask 5), all additive -- no
training, no changes to existing files:

  - ``recon_diff``  : |broken - reconstruct(broken)|           (shape-only, image-free; VAR/VQ)
  - ``fpfn``        : max(FP, FN) predicted correction maps     (image-conditioned; G-VAR etc.)
  - ``cascade``     : |refine(broken) - broken|                 (Faz-4 reference-mask difference)

Each produces a soft map in [0, 1]; metrics are computed per DEGRADED row then averaged, and also
reported in true-Dice bands so the hard cases are visible. Nothing here is gated on reconstruction
fidelity -- only on how well the signal localizes the injected error.

Run (full):
    PYTHONIOENCODING=utf-8 KMP_DUPLICATE_LIB_OK=TRUE python scripts/evaluate_localization.py \
        --var outputs/checkpoints/vae3d_var_best.pt \
        --fpfn-var outputs/checkpoints/vae3d_g_var_best.pt \
        --cascade outputs/checkpoints/vae3d_h_cascade_best.pt
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from data import load_config  # noqa: E402
from data.degradations_3d import PRIMITIVES_3D  # noqa: E402
from train import resolve_device, set_seed  # noqa: E402

import evaluate_correlation_3d as ev3d  # noqa: E402
import evaluate_correlation_3d_var as evv  # noqa: E402
import evaluate_correlation_3d_vq as evq  # noqa: E402
import evaluate_correlation_cascade as evc  # noqa: E402
import evaluate_correlation_imgcond as evimg  # noqa: E402
import evaluate_correlation_imgcond_fpfn as evfpfn  # noqa: E402
import evaluate_correlation_var_fpfn as evvf  # noqa: E402

logger = logging.getLogger("evaluate_localization")

SCORE_BATCH = ev3d.SCORE_BATCH


# --------------------------------------------------------------------------
# Localizers: (model, broken[b,D,H,W] float, images[b,D,H,W] float) -> soft error map [b,D,H,W]
# --------------------------------------------------------------------------
@torch.no_grad()
def recon_diff_map(model, broken: np.ndarray, images: np.ndarray, device) -> np.ndarray:
    x = torch.from_numpy(broken[:, None].astype(np.float32)).to(device)   # mask-only, image-free
    recon = model.reconstruct(x)[:, 0]                                    # prob in [0,1]
    bt = torch.from_numpy(broken.astype(np.float32)).to(device)
    return (bt - recon).abs().cpu().numpy()


@torch.no_grad()
def fpfn_map(model, broken: np.ndarray, images: np.ndarray, device) -> np.ndarray:
    x = torch.from_numpy(np.stack([images, broken], axis=1).astype(np.float32)).to(device)
    fp, fn = model.predict_fpfn(x)                                        # each [b,1,D,H,W] prob
    return torch.maximum(fp[:, 0], fn[:, 0]).cpu().numpy()                # union of predicted edits


@torch.no_grad()
def cascade_map(model, broken: np.ndarray, images: np.ndarray, device) -> np.ndarray:
    x = torch.from_numpy(np.stack([images, broken], axis=1).astype(np.float32)).to(device)
    ref = model.predict_reference(x)[:, 0]                               # refined reference prob
    bt = torch.from_numpy(broken.astype(np.float32)).to(device)
    return (ref - bt).abs().cpu().numpy()


@torch.no_grad()
def diffusion_map(bundle, broken: np.ndarray, images: np.ndarray, device) -> np.ndarray:
    """AnoDDPM localizer (Part 2): partial-noise the broken mask's frozen-VAR latent
    then reverse-denoise toward the learned clean-shape manifold; the anomaly map is
    |broken - decode(denoised)|. ``bundle`` is ``(frozen_var, ddpm, t_start)``."""
    var, ddpm, t_start = bundle
    x = torch.from_numpy(broken[:, None].astype(np.float32)).to(device)   # mask-only, image-free
    f_hat, _, _ = var.quantize(x)                                         # quantized VAR latent
    denoised = ddpm.partial_denoise(f_hat, t_start)
    recon = torch.sigmoid(var.decode(denoised))[:, 0]                     # prob in [0,1]
    bt = torch.from_numpy(broken.astype(np.float32)).to(device)
    return (bt - recon).abs().cpu().numpy()


def load_diffusion(path: Path, device):
    """Load a latent-DDPM checkpoint plus the FROZEN VAR it was trained over.
    Returns the ``(var, ddpm, t_start)`` bundle :func:`diffusion_map` expects."""
    import models.diffusion3d as dm  # local import: only needed when --diffusion is given

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    dcfg = cfg["diffusion3d"]
    var_ckpt = Path(dcfg["var_checkpoint"])
    if not var_ckpt.is_absolute():
        var_ckpt = PROJECT_ROOT / var_ckpt
    var = evv.load_model(var_ckpt, device)
    ddpm = dm.build_from_config(cfg, var.embedding_dim, var.feature_shape[0]).to(device)
    ddpm.load_state_dict(ckpt["model_state"])
    ddpm.eval()
    return (var, ddpm, int(dcfg["locate_t"]))


# --------------------------------------------------------------------------
def _band_of(d: float) -> str:
    if d >= 0.85:
        return "kolay [0.85-1.0]"
    if d >= 0.70:
        return "orta [0.70-0.85]"
    if d >= 0.55:
        return "zor [0.55-0.70]"
    return "cok zor [<0.55]"


def score_localizer(
    localizer: Callable, model, broken: np.ndarray, images: np.ndarray, clean_ref: np.ndarray,
    true_dice: np.ndarray, is_clean: np.ndarray, device,
) -> dict[str, Any]:
    """Voxel-AUROC / AP per degraded row vs the true error map (broken XOR clean)."""
    aurocs: list[float] = []
    aps: list[float] = []
    dices: list[float] = []
    for start in range(0, len(broken), SCORE_BATCH):
        sl = slice(start, start + SCORE_BATCH)
        soft = localizer(model, broken[sl].astype(np.float32), images[sl].astype(np.float32), device)
        for j in range(soft.shape[0]):
            gi = start + j
            if is_clean[gi]:
                continue
            true_err = (broken[gi].astype(bool) ^ clean_ref[gi].astype(bool)).ravel()
            if true_err.sum() == 0 or true_err.all():
                continue  # AUROC undefined without both classes
            pred = soft[j].ravel().astype(np.float64)
            aurocs.append(float(roc_auc_score(true_err, pred)))
            aps.append(float(average_precision_score(true_err, pred)))
            dices.append(float(true_dice[gi]))
    aurocs_a, aps_a, dices_a = np.asarray(aurocs), np.asarray(aps), np.asarray(dices)
    bands: dict[str, Any] = {}
    band_labels = np.array([_band_of(d) for d in dices_a])
    for b in ("kolay [0.85-1.0]", "orta [0.70-0.85]", "zor [0.55-0.70]", "cok zor [<0.55]"):
        m = band_labels == b
        if m.any():
            bands[b] = {"n": int(m.sum()), "auroc": float(aurocs_a[m].mean()),
                        "ap": float(aps_a[m].mean())}
    return {
        "n": int(len(aurocs)),
        "auroc_mean": float(aurocs_a.mean()) if len(aurocs_a) else float("nan"),
        "auroc_std": float(aurocs_a.std()) if len(aurocs_a) else float("nan"),
        "ap_mean": float(aps_a.mean()) if len(aps_a) else float("nan"),
        "ap_std": float(aps_a.std()) if len(aps_a) else float("nan"),
        "bands": bands,
    }


# --------------------------------------------------------------------------
def build_registry(args, device) -> list[tuple[str, Callable, Any]]:
    """Return [(label, localizer_fn, model), ...] for every checkpoint that is present."""
    reg: list[tuple[str, Callable, Any]] = []

    def _p(path: str | None) -> Path | None:
        if not path:
            return None
        pp = Path(path)
        if not pp.is_absolute():
            pp = PROJECT_ROOT / pp
        return pp if pp.exists() else None

    vq = _p(args.vq)
    if vq is not None:
        reg.append(("recon_diff/VQ", recon_diff_map, evq.load_model(vq, device)))
    var = _p(args.var)
    if var is not None:
        reg.append(("recon_diff/VAR", recon_diff_map, evv.load_model(var, device)))
    fvar = _p(args.fpfn_var)
    if fvar is not None:
        reg.append(("fpfn/G-VAR", fpfn_map, evvf.load_var_fpfn_model(fvar, device)))
    fvq = _p(args.fpfn_vq)
    if fvq is not None:
        reg.append(("fpfn/G-VQ", fpfn_map, evfpfn.load_fpfn_model(fvq, "binary", "vq", device)))
    cas = _p(args.cascade)
    if cas is not None:
        cfg_cas = load_config(args.cascade_config)
        cfg_cas.setdefault("data", {})["data_source"] = args.data_source
        reg.append(("cascade/Faz4", cascade_map, evc.load_cascade(cfg_cas, cas, device)))
    dif = _p(args.diffusion)
    if dif is not None:
        reg.append(("diffusion/latent-DDPM", diffusion_map, load_diffusion(dif, device)))
    return reg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Voxel-level error-localization benchmark (Part 1).")
    p.add_argument("--config", default="configs/vae3d_imgcond_fpfn.yaml",
                   help="config that provides the image+mask eval set")
    p.add_argument("--cascade-config", default="configs/vae3d_cascade.yaml")
    p.add_argument("--var", default="outputs/checkpoints/vae3d_var_best.pt")
    p.add_argument("--vq", default="outputs/checkpoints/vae3d_vq_best.pt")
    p.add_argument("--fpfn-var", default="outputs/checkpoints/vae3d_g_var_best.pt")
    p.add_argument("--fpfn-vq", default="outputs/checkpoints/vae3d_g_vq_best.pt")
    p.add_argument("--cascade", default="outputs/checkpoints/vae3d_h_cascade_best.pt")
    p.add_argument("--diffusion", default="outputs/checkpoints/vae3d_diffusion_best.pt",
                   help="latent-DDPM checkpoint (Part 2); scored only if present")
    p.add_argument("--split", default="test")
    p.add_argument("--n-per-mask", type=int, default=5)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--data-source", choices=("nifti", "precomputed"), default="precomputed")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="localization")
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
    set_seed(args.seed)
    device = resolve_device(config["train"]["device"])

    clean_masks, clean_images = evimg.load_clean_image_volumes(config, args.split, args.limit, args.seed)
    degrader = ev3d.calibrate_degrader(config, clean_masks, args.seed)
    broken, images, clean_ref, true_dice, is_clean = evfpfn.build_eval_set_fpfn(
        clean_masks, clean_images, degrader, args.n_per_mask, args.seed)
    logger.info("eval set: %d volumes (%d degraded)", len(broken), int((~is_clean).sum()))

    registry = build_registry(args, device)
    if not registry:
        logger.error("no checkpoints found -- nothing to score")
        return

    results: dict[str, Any] = {}
    print(f"\n{'localizer':18s} {'n':>4s}  {'AUROC':>14s}  {'AP (PR-AUC)':>14s}")
    print("-" * 56)
    for label, fn, model in registry:
        r = score_localizer(fn, model, broken, images, clean_ref, true_dice, is_clean, device)
        results[label] = r
        print(f"{label:18s} {r['n']:>4d}  {r['auroc_mean']:.3f} +/- {r['auroc_std']:.3f}  "
              f"{r['ap_mean']:.3f} +/- {r['ap_std']:.3f}")

    out = PROJECT_ROOT / "outputs" / f"{args.tag}_results.json"
    out.write_text(json.dumps({
        "config": args.config, "split": args.split, "n_per_mask": args.n_per_mask,
        "seed": args.seed, "primitives": list(PRIMITIVES_3D), "results": results,
    }, indent=2), encoding="utf-8")
    print("\nsaved:", out)


if __name__ == "__main__":
    main()
