# segquality — ground-truth-free 3D segmentation quality estimation

A shape prior (a **3D VQ-VAE**) is trained only on *good* anatomical masks. At test
time an arbitrary segmentation mask is passed through the model and its
**reconstruction error is used as a quality score**: masks that look like the
learned anatomy reconstruct well (high Dice against themselves), corrupted masks do
not. No ground truth and no image intensities are needed at scoring time — the score
is purely shape-based, so it is organ- and modality-agnostic in principle.

The reconstruction quality is validated by correlating the score against the *true*
Dice of controlled synthetic degradations: a good score tracks true Dice
(Spearman rho ~0.85).

## What is in here (and the main findings)

| Phase | Model | File | Result |
|-------|-------|------|--------|
| Baseline | Continuous-latent 3D VAE | `models/vae3d.py` | rho~0.85 **but mean-reverts** (decoder emits a generic mean shape) |
| **Fix** | **3D VQ-VAE** (spatial discrete codebook) | `models/vae3d_vq.py` | **cures mean-reversion** (cross-recon ratio 1.005 -> ~1.24), rho~0.85, val recon Dice ~0.91 |
| WP-2 | VQ-VAE + geometric conditioning | `models/vae3d_vq_cond.py` | **negative**: external geometry did not help (rho 0.850 -> 0.800/0.822) |
| WP-3 | VQ-VAE with **SDF** target | `models/vae3d_vq_sdf.py` | **lossless** (rho 0.844 vs 0.850); geometry baked into the target, foundation for modality independence |

- The VQ-VAE quantizer (`VectorQuantizer3D`) is a hand-written van den Oord (2017)
  vector quantizer generalised to a 5D `(B, C, d, h, w)` spatial grid: nearest-code
  lookup + straight-through estimator + codebook/commitment loss. No KL term.
- WP-3's SDF model regresses a **signed distance function** (outside +, inside -,
  boundary 0) with `SmoothL1`; a binary bridge `sdf < 0 -> inside` turns the
  prediction back into a mask so the identical scoring/correlation machinery applies.

See `ADVISOR_QA.md` for line-referenced answers to the technical review questions,
and `figures/` for the reconstruction previews, cross-reconstruction matrix and
correlation scatter plots.

## Clone and run (checkpoints + data included via Git LFS)

The three canonical checkpoints and the precomputed 64^3 mask splits ship in the
repo through **Git LFS**, so you can clone and evaluate without downloading raw RAOS.

```bash
# 1. Git LFS must be installed once on your machine (https://git-lfs.com)
git lfs install

# 2. Clone — LFS files (*.pt, *.npz) are fetched automatically
git clone https://github.com/ahmetyaylalioglu/segquality.git
cd segquality
git lfs pull            # ensure the LFS blobs are materialised

# 3. Python environment (a CUDA GPU is recommended for scoring)
conda create -n segquality python=3.10 -y
conda activate segquality
pip install -r requirements.txt
```

What LFS brings down:

```
outputs/checkpoints/vae3d_vqvae_best.pt     # base 3D VQ-VAE   (rho ~0.85)
outputs/checkpoints/vae3d_wp2_base_best.pt  # WP-2 baseline    (rho 0.850, binary reference)
outputs/checkpoints/vae3d_wp3_sdf_best.pt   # WP-3 SDF-VQ-VAE  (rho 0.844)
outputs/volumes/{train,val,test}.npz        # precomputed squashed 64^3 masks
```

## Evaluate (reproduce the correlation numbers)

The precomputed splits are the default data source, so these run out of the box:

```bash
# Base 3D VQ-VAE: reconstruction error vs true Dice
python scripts/evaluate_correlation_3d_vq.py \
    --config configs/vae3d_vq.yaml \
    --checkpoint outputs/checkpoints/vae3d_vqvae_best.pt \
    --split test --n-per-mask 5 --seed 42

# WP-3 SDF-VQ-VAE (scored through the sdf<0 binary bridge)
python scripts/evaluate_correlation_3d_sdf.py \
    --config configs/vae3d_vq.yaml \
    --checkpoint outputs/checkpoints/vae3d_wp3_sdf_best.pt \
    --split test --n-per-mask 5 --seed 42
```

Each writes a correlation table to the console plus `outputs/correlation_results_*.json`
and scatter/comparison PNGs.

## Train from scratch (optional)

```bash
# base VQ-VAE
python scripts/train_vae3d_vq.py --config configs/vae3d_vq.yaml --num-workers 0
# WP-3 SDF target
python scripts/train_vae3d_vq_sdf.py --tag wp3_sdf --num-workers 0
```

On Windows use `--num-workers 0` (multiprocessing DataLoader + subprocess can crash
the Torch DLL). Training reads only *clean* masks; degradation happens exclusively in
the evaluation scripts.

## Testing

```bash
pytest -p no:logfire
```

The suite (162 tests) uses synthetic masks and passes without the raw data.

## Folder layout

```
segquality/
  configs/     YAML configs (vae3d_vq.yaml is the VQ-VAE config)
  data/        config loader, RAOS dataset, transforms, 2D/3D degradations
  models/      vae3d, vae3d_vq (+ _cond, _sdf variants)
  quality/     score functions, geometric features, SDF helpers
  scripts/     train_* and evaluate_correlation_3d* entry points
  tests/       pytest suite (synthetic data only)
  figures/     curated result figures for the write-up
  outputs/     generated artifacts; only the 3 canonical checkpoints + 3 npz are tracked (LFS)
```

## Notes

- The raw RAOS dataset lives **outside** the repo; its path comes from
  `configs/raos.yaml` (`dataset.root`). Only needed for `--data-source nifti` or to
  regenerate the precomputed npz — the shipped npz cover the evaluation.
- Everything operates in the squashed 64^3 mask space so degradation, reconstruction
  and true-Dice measurement all share one coordinate frame.
