"""Qualitative panel for the 2-stage VAR refiner (2026-10-20 direction).

Picks one test patient, applies a light -> heavy degradation sweep, and shows the
reconstructed mask (TP OR FN) at stage-1 and after each refiner iteration, with the
residual error (recon XOR clean) overlaid in red and Dice annotated. The red region
shrinks and Dice rises across iterations -- the figure we embed in the Word report.

Inference only (no training). Reuses the cascade loader from ``evaluate_cascade_tp``
and the degradation primitives from the localization notebook.

Usage
-----
    python scripts/visualize_cascade_tp.py \
        --config configs/vae3d_cascade_tp.yaml \
        --refiner outputs/checkpoints/vae3d_h_cascade_tp_best.pt \
        --patient 0 --tag h_cascade_tp
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

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
from data.degradations import dice as _dice  # noqa: E402
from data.degradations_3d import PRIMITIVES_3D  # noqa: E402
from data.degradations_partial_3d import PARTIAL_PRIMITIVES_3D  # noqa: E402
from data.fpfntp_targets import reconstruct_from_fntp  # noqa: E402
from train import resolve_device, set_seed  # noqa: E402

import evaluate_cascade_tp as evc  # noqa: E402
import evaluate_correlation_imgcond as evic  # noqa: E402

ALL_FNS = {**PRIMITIVES_3D, **PARTIAL_PRIMITIVES_3D}
SPECS = [("erode", 0.25, "hafif erozyon / light erosion"),
         ("erode", 0.55, "orta erozyon / medium erosion"),
         ("center_only", 0.60, "ağır kayıp / heavy loss")]


@torch.no_grad()
def recon_sequence(model, img, broken, device):
    """Return (base_recon, [iter1, iter2, ...]) reconstructed masks (bool, D,H,W)."""
    x = torch.from_numpy(np.stack([img.astype(np.float32), broken.astype(np.float32)], axis=0)[None]).to(device)
    fp0, fn0, tp0 = model.stage1_products(x)
    base = (reconstruct_from_fntp(fn0, tp0) > 0.5).cpu().numpy()[0, 0].astype(bool)
    outputs, _ = model.forward(x)
    iters = [(reconstruct_from_fntp(fn_i, tp_i) > 0.5).cpu().numpy()[0, 0].astype(bool)
             for (_, fn_i, tp_i) in outputs]
    return base, iters


def _show_mask_with_error(ax, mask, gt, title_dice=None):
    err = mask ^ gt
    ax.imshow(mask, cmap="gray", vmin=0, vmax=1)
    ax.imshow(np.ma.masked_where(~err, err), cmap="autumn_r", alpha=0.9, vmin=0, vmax=1)
    if title_dice is not None:
        ax.set_xlabel(f"Dice={title_dice:.2f}", fontsize=8)


def build_panel(model, clean_masks, clean_images, patient, iters, seed, out_path):
    gt = clean_masks[patient].astype(bool)
    img = clean_images[patient]
    zc = gt.shape[0] // 2
    cols = ["CT görüntü\nCT image", "temiz (GT)\nclean (GT)", "bozuk\nbroken",
            "Aşama-1\nStage-1"] + [f"iter {i}" for i in range(1, iters + 1)]
    fig, axes = plt.subplots(len(SPECS), len(cols),
                             figsize=(2.15 * len(cols), 2.2 * len(SPECS)), squeeze=False)
    for ri, (name, sev, lvl) in enumerate(SPECS):
        b = np.asarray(ALL_FNS[name](gt, sev, np.random.default_rng(seed))) > 0
        d_broken = float(_dice(gt, b))
        base, iter_recons = recon_sequence(model, img, b, model.refiner.parameters().__next__().device)
        axes[ri, 0].imshow(img[zc], cmap="gray")
        axes[ri, 1].imshow(gt[zc], cmap="gray", vmin=0, vmax=1)
        _show_mask_with_error(axes[ri, 2], b[zc], gt[zc], d_broken)
        _show_mask_with_error(axes[ri, 3], base[zc], gt[zc], float(_dice(gt, base)))
        for i, rec in enumerate(iter_recons, start=1):
            _show_mask_with_error(axes[ri, 3 + i], rec[zc], gt[zc], float(_dice(gt, rec)))
        for ci in range(len(cols)):
            axes[ri, ci].set_xticks([]); axes[ri, ci].set_yticks([])
        axes[ri, 0].set_ylabel(f"{lvl}\n{name}/{sev}", fontsize=9)
    for ci, t in enumerate(cols):
        axes[0, ci].set_title(t, fontsize=8.5)
    fig.suptitle(f"İki aşamalı VAR düzeltici — hasta {patient}  |  kırmızı = kalan hata (recon XOR GT)\n"
                 f"Two-stage VAR refiner — patient {patient}  |  red = residual error", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qualitative refiner panel (inference only).")
    p.add_argument("--config", default="configs/vae3d_cascade_tp.yaml")
    p.add_argument("--refiner", required=True)
    p.add_argument("--patients", default="0", help="comma-separated patient indices, e.g. 0,1,2")
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--data-source", choices=("nifti", "precomputed"), default="precomputed")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="h_cascade_tp")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    config = load_config(args.config)
    config.setdefault("data", {})["data_source"] = args.data_source
    set_seed(args.seed)
    device = resolve_device(config["train"]["device"])

    ref_path = args.refiner
    if not Path(ref_path).is_absolute():
        ref_path = PROJECT_ROOT / ref_path
    model = evc.load_cascade(config, Path(ref_path), device)

    clean_masks, clean_images = evic.load_clean_image_volumes(config, args.split, args.limit, args.seed)
    patients = [int(s) for s in str(args.patients).split(",") if s.strip() != ""]
    for patient in patients:
        out_path = PROJECT_ROOT / "outputs" / f"cascade_tp_qualitative_p{patient}.png"
        build_panel(model, clean_masks, clean_images, patient, model.iters, args.seed, out_path)


if __name__ == "__main__":
    main()
