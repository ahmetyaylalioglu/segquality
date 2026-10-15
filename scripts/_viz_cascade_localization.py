"""Cascade error-localization on MANY examples, 2D multi-slice AND 3D isosurface.

The cascade (Faz-4) localizer is ``|refine(broken) - broken|``: stage-1 (frozen VAR
FP/FN) + stage-2 refiner turn ``[image, broken]`` into a refined reference mask, and
the voxels where that reference moves away from the broken mask ARE the error. It is
the strongest localizer in the benchmark (voxel-AUROC 0.980). This script shows WHERE
it fires, qualitatively, on many degraded livers.

Two figures, both additive (reuses evaluate_localization; no training, no protected-file
edits):
  * outputs/cascade_localization_multi2d.png -- N examples x several axial slices,
    true error (yellow) vs cascade map (red), so the per-voxel agreement is visible.
  * outputs/cascade_localization_3d.png      -- a few examples as 3D marching-cubes
    isosurfaces: broken mask (grey), true error region (gold), cascade-predicted error
    region (red). Answers "can the visualization be 3D?" -- yes.

    python scripts/_viz_cascade_localization.py
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401  (registers 3d projection)
from skimage import measure  # noqa: E402

from data import load_config  # noqa: E402
from train import resolve_device, set_seed  # noqa: E402

import evaluate_localization as loc  # noqa: E402
import evaluate_correlation_3d as ev3d  # noqa: E402
import evaluate_correlation_imgcond as evimg  # noqa: E402
import evaluate_correlation_imgcond_fpfn as evfpfn  # noqa: E402

SEED = 42
N_PER_MASK = 5
CASCADE_CKPT = "outputs/checkpoints/vae3d_h_cascade_best.pt"


def _load_cascade_and_set(device):
    cfg = load_config("configs/vae3d_imgcond_fpfn.yaml")
    cfg.setdefault("data", {})["data_source"] = "precomputed"
    set_seed(SEED)
    clean_masks, clean_images = evimg.load_clean_image_volumes(cfg, "test", None, SEED)
    degrader = ev3d.calibrate_degrader(cfg, clean_masks, SEED)
    broken, images, clean_ref, true_dice, is_clean = evfpfn.build_eval_set_fpfn(
        clean_masks, clean_images, degrader, N_PER_MASK, SEED)

    cfg_cas = load_config("configs/vae3d_cascade.yaml")
    cfg_cas.setdefault("data", {})["data_source"] = "precomputed"
    model = loc.evc.load_cascade(cfg_cas, PROJECT_ROOT / CASCADE_CKPT, device)
    return model, broken, images, clean_ref, true_dice, is_clean


def _pick(broken, clean_ref, true_dice, is_clean, n, lo=0.55, hi=0.85):
    """Degraded rows in a mid-hard Dice band with a localized (not near-total) error."""
    cand = [i for i in np.where(~is_clean)[0] if lo <= true_dice[i] <= hi]
    cand.sort(key=lambda i: true_dice[i])
    out = []
    for i in cand:
        err = broken[i].astype(bool) ^ clean_ref[i].astype(bool)
        if 100 < err.sum() < 0.4 * err.size:
            out.append(i)
        if len(out) == n:
            break
    return out


# --------------------------------------------------------------------------
def fig_multi2d(model, broken, images, clean_ref, true_dice, is_clean, device, n=5, k=3):
    picks = _pick(broken, clean_ref, true_dice, is_clean, n)
    cmaps = loc.cascade_map(model, broken[picks].astype(np.float32),
                            images[picks].astype(np.float32), device)
    ncol = 1 + 2 * k  # bozuk(orta kesit) + k*(true, cascade)
    fig, axes = plt.subplots(n, ncol, figsize=(2.2 * ncol, 2.2 * n))
    axes = np.atleast_2d(axes)
    for r, gi in enumerate(picks):
        err = broken[gi].astype(bool) ^ clean_ref[gi].astype(bool)
        zc = err.sum(axis=(1, 2))
        zs = np.argsort(zc)[::-1][:k]            # k slices with most error
        zs = sorted(zs)
        axes[r, 0].imshow(broken[gi][broken[gi].shape[0] // 2], cmap="gray", vmin=0, vmax=1)
        axes[r, 0].set_ylabel(f"Dice={true_dice[gi]:.2f}", fontsize=10)
        if r == 0:
            axes[r, 0].set_title("bozuk (orta)", fontsize=10)
        for c, z in enumerate(zs):
            at = axes[r, 1 + 2 * c]
            ac = axes[r, 2 + 2 * c]
            at.imshow(err[z].astype(float), cmap="autumn_r", vmin=0, vmax=1)
            ac.imshow(cmaps[r][z], cmap="Reds", vmin=0, vmax=1)
            if r == 0:
                at.set_title(f"gercek hata z={z}", fontsize=9)
                ac.set_title(f"cascade z={z}", fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Cascade (Faz-4) hata lokalizasyonu -- goruntu-kosullu, voxel-AUROC 0.980",
                 fontsize=13)
    out = PROJECT_ROOT / "outputs" / "cascade_localization_multi2d.png"
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out, picks


# --------------------------------------------------------------------------
def _add_surface(ax, vol, color, alpha, level=0.5):
    """Marching-cubes isosurface rendered with ``plot_trisurf`` (handles lighting, so
    solid regions read as 3D instead of flat black silhouettes)."""
    if vol.sum() < 10:
        return
    try:
        verts, faces, _, _ = measure.marching_cubes(vol.astype(np.float32), level=level)
    except (ValueError, RuntimeError):
        return
    ax.plot_trisurf(
        verts[:, 0], verts[:, 1], faces, verts[:, 2],
        color=color, alpha=alpha, linewidth=0, antialiased=True, shade=True,
    )


def fig_3d(model, broken, images, clean_ref, true_dice, is_clean, device, n=3):
    picks = _pick(broken, clean_ref, true_dice, is_clean, n)
    cmaps = loc.cascade_map(model, broken[picks].astype(np.float32),
                            images[picks].astype(np.float32), device)
    fig = plt.figure(figsize=(4.2 * 3, 4.6 * n))
    for r, gi in enumerate(picks):
        b = broken[gi].astype(bool)
        true_err = (b ^ clean_ref[gi].astype(bool))
        pred_err = cmaps[r] > 0.5
        d = b.shape
        cols = [
            ("bozuk maske (yuzey)", [(b, (0.62, 0.66, 0.78), 1.0)]),
            ("gercek hata", [(b, (0.70, 0.74, 0.85), 0.15), (true_err, (1.0, 0.75, 0.0), 1.0)]),
            ("cascade tahmini", [(b, (0.70, 0.74, 0.85), 0.15), (pred_err, (0.92, 0.15, 0.15), 1.0)]),
        ]
        for c, (title, layers) in enumerate(cols):
            ax = fig.add_subplot(n, 3, r * 3 + c + 1, projection="3d")
            for vol, color, alpha in layers:
                _add_surface(ax, vol, color, alpha)
            ax.set_xlim(0, d[0]); ax.set_ylim(0, d[1]); ax.set_zlim(0, d[2])
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.view_init(elev=18, azim=-60)
            if r == 0:
                ax.set_title(title, fontsize=11)
            if c == 0:
                ax.text2D(-0.1, 0.5, f"Dice={true_dice[gi]:.2f}", transform=ax.transAxes,
                          fontsize=10, rotation=90, va="center")
    fig.suptitle("Cascade hata lokalizasyonu -- 3B izoyuzey (marching cubes)", fontsize=13)
    out = PROJECT_ROOT / "outputs" / "cascade_localization_3d.png"
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out, picks


def main() -> None:
    device = resolve_device("auto")
    model, broken, images, clean_ref, true_dice, is_clean = _load_cascade_and_set(device)
    o1, p1 = fig_multi2d(model, broken, images, clean_ref, true_dice, is_clean, device)
    print("saved:", o1, "| examples:", list(map(int, p1)))
    o2, p2 = fig_3d(model, broken, images, clean_ref, true_dice, is_clean, device)
    print("saved:", o2, "| examples:", list(map(int, p2)))


if __name__ == "__main__":
    main()
