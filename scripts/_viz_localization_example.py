"""Save a qualitative localization panel: for a few HARD degraded livers, show the
image-free shape-only localizers side by side -- recon_diff/VAR (single quantizer
pass) vs diffusion/latent-DDPM (AnoDDPM). Reuses evaluate_localization verbatim;
purely additive, no training. Picks, per selected example, the axial slice with
the most true-error voxels so the difference is visible.

    python scripts/_viz_localization_example.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data import load_config  # noqa: E402
from train import resolve_device, set_seed  # noqa: E402

import evaluate_localization as loc  # noqa: E402
import evaluate_correlation_3d as ev3d  # noqa: E402
import evaluate_correlation_3d_var as evv  # noqa: E402
import evaluate_correlation_imgcond as evimg  # noqa: E402
import evaluate_correlation_imgcond_fpfn as evfpfn  # noqa: E402

SEED = 42
N_PER_MASK = 5
N_SHOW = 3  # hard examples to display


def main() -> None:
    cfg = load_config("configs/vae3d_imgcond_fpfn.yaml")
    cfg.setdefault("data", {})["data_source"] = "precomputed"
    set_seed(SEED)
    device = resolve_device(cfg["train"]["device"])

    clean_masks, clean_images = evimg.load_clean_image_volumes(cfg, "test", None, SEED)
    degrader = ev3d.calibrate_degrader(cfg, clean_masks, SEED)
    broken, images, clean_ref, true_dice, is_clean = evfpfn.build_eval_set_fpfn(
        clean_masks, clean_images, degrader, N_PER_MASK, SEED)

    var = evv.load_model(PROJECT_ROOT / "outputs/checkpoints/vae3d_var_best.pt", device)
    bundle = loc.load_diffusion(PROJECT_ROOT / "outputs/checkpoints/vae3d_diffusion_best.pt", device)

    # Pick from the "zor" band (Dice 0.55-0.80): the mask is mostly intact but
    # carries a clearly LOCALIZED error -- the regime where the band table shows
    # diffusion's biggest gain and where a per-voxel map is visually meaningful.
    # (The absolute-hardest near-total-erosion cases leave no shape to localize
    # from and make a confusing picture.)
    deg = [i for i in np.where(~is_clean)[0] if 0.55 <= true_dice[i] <= 0.80]
    deg = sorted(deg, key=lambda i: true_dice[i])
    picks = []
    for i in deg:
        true_err = broken[i].astype(bool) ^ clean_ref[i].astype(bool)
        if 50 < true_err.sum() < 0.5 * true_err.size:  # localized, not near-total
            picks.append(i)
        if len(picks) == N_SHOW:
            break

    rd = loc.recon_diff_map(var, broken[picks].astype(np.float32),
                            images[picks].astype(np.float32), device)
    df = loc.diffusion_map(bundle, broken[picks].astype(np.float32),
                           images[picks].astype(np.float32), device)

    cols = ["kesit (CT)", "bozuk maske", "gercek hata", "recon_diff/VAR", "diffusion/DDPM"]
    fig, axes = plt.subplots(N_SHOW, 5, figsize=(14, 3 * N_SHOW))
    axes = np.atleast_2d(axes)
    for r, gi in enumerate(picks):
        true_err = (broken[gi].astype(bool) ^ clean_ref[gi].astype(bool))
        z = int(true_err.sum(axis=(1, 2)).argmax())  # axial slice with most error
        panels = [
            (images[gi][z], "gray", None),
            (broken[gi][z], "gray", (0, 1)),
            (true_err[z].astype(float), "autumn_r", (0, 1)),
            (rd[r][z], "Reds", (0, 1)),
            (df[r][z], "Reds", (0, 1)),
        ]
        for c, (img, cmap, lim) in enumerate(panels):
            ax = axes[r, c]
            if lim is None:
                ax.imshow(img, cmap=cmap)
            else:
                ax.imshow(img, cmap=cmap, vmin=lim[0], vmax=lim[1])
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=11)
        axes[r, 0].set_ylabel(f"Dice={true_dice[gi]:.2f}\nz={z}", fontsize=10)

    fig.suptitle("Goruntusuz sekil-tabanli hata lokalizasyonu: recon_diff vs diffusion (zor vakalar)",
                 fontsize=13)
    out = PROJECT_ROOT / "outputs" / "localization_diffusion_example.png"
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    main()
