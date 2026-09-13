# Result figures

Curated figures for the write-up / advisor review. Regenerate any of these with the
`scripts/evaluate_correlation_3d_*.py`, `scripts/diagnose_reconstruction_vq.py` and
`scripts/train_vae3d_vq*.py` entry points (see the top-level README).

## `vqvae/` — base 3D VQ-VAE (mean-reversion cure)

| File | What it shows |
|------|---------------|
| `reconstruction_preview.png` | Labeled input/reconstruction grid over several val patients — each liver reconstructs to *its own* shape (input-specific), not a generic mean. |
| `reconstruction_epoch100_ortho.png` | Axial/coronal/sagittal centre slices, input vs reconstruction, at epoch 100. |
| `cross_reconstruction_matrix.png` | Cross-reconstruction Dice matrix: strong diagonal, weak off-diagonal = the decoder is input-specific (ratio ~1.24), i.e. mean-reversion is cured (baseline VAE was ~1.005). |
| `correlation_scatter.png` | Reconstruction Dice vs true Dice, **Spearman rho = 0.862** (n=282, 47 clean + 235 degraded). |
| `score_comparison.png` | \|Spearman rho\| per quality score — recon_dice wins. |
| `training_curves.png` | Train/val loss, val recon Dice, codebook perplexity over training. |

## `sdf/` — WP-3 SDF-target VQ-VAE

| File | What it shows |
|------|---------------|
| `sdf_reconstruction_preview.png` | input mask \| predicted signed distance map (RdBu, ±3) \| bridged binary (`sdf<0`) with self-Dice, for a few patients. |
| `correlation_scatter.png` | Bridged reconstruction Dice vs true Dice, **Spearman rho = 0.844**. |
| `binary_vs_sdf_comparison.png` | Binary baseline (rho 0.850) vs SDF-VQ (rho 0.844), global + per-Dice-band — SDF is lossless. |
| `training_curves.png` | SDF SmoothL1 loss, val bridge Dice, codebook perplexity. |

## `conditioning/` — WP-2 geometric conditioning (negative result)

| File | What it shows |
|------|---------------|
| `wp2_conditioning_ablation.png` | Baseline 0.850 vs topology-only 0.800 vs full-feature 0.822 — external geometry did **not** help; codebook perplexity rose (latent not crushed). |
