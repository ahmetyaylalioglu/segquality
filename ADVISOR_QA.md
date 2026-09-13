# Advisor Q&A — technical review

Line-referenced answers to the review questions about the VQ-VAE implementation.
All references are to the committed code on `main`.

## 1. Verifications

### 1a. Is there a double sigmoid? — **NO**

The training BCE receives **raw logits**; the sigmoid in `reconstruct()` is a
separate scoring/visualisation path never used during training.

- `models/vae3d_vq.py:222-226` — `forward()` returns raw `logits` from
  `_decode_features`; the final decoder stage (`vae3d_vq.py:174-176`) is a bare
  `ConvTranspose3d` with **no** BatchNorm/activation ("final stage -> raw logits").
- `scripts/train_vae3d_vq.py:114-115` — training uses those raw logits:
  `logits, vq_term, info = model(x)` → `vq_loss(logits, x, ...)`.
- `losses/vae_losses.py:43` — `bce = F.binary_cross_entropy_with_logits(logits, target)`.
  `binary_cross_entropy_with_logits` applies sigmoid **once, internally**, to raw
  logits — the correct, numerically stable usage.
- `losses/vae_losses.py:44` — the Dice term applies `torch.sigmoid(logits)` **once**,
  only for soft-Dice (which needs probabilities); it is a separate branch from BCE.
- `models/vae3d_vq.py:228-231` — `reconstruct()` = `sigmoid(logits)`, used **only** in
  evaluation/visualisation (`evaluate_correlation_3d_vq.py`), never in the train loop.

Variants:
- **cond** (`models/vae3d_vq_cond.py:160-167`): `forward` decodes through the inherited
  bare final `ConvTranspose3d` → raw logits; same `run_epoch_vq` → same `vq_loss`.
- **SDF** (`scripts/train_vae3d_vq_sdf.py:98-99`): loss is
  `F.smooth_l1_loss(pred_sdf, target_sdf)` — **no BCE, no sigmoid** on the SDF path.

**Verdict: correct — no double sigmoid; results are sound.**

### 1b. Is the training input clean or degraded? — **CLEAN only**

- `scripts/train_vae3d.py:114-124` — `make_loaders` loads clean GT masks from
  `RAOSVolumes.from_config(split="train"/"val")`.
- `transform=_VolumeTransform()` (`train_vae3d.py:117`) routes through `augment_volume`
  (`train_vae3d.py:77`) which is `return mask` — a no-op.
- `scripts/train_vae3d_vq.py:112-115` — the reconstruction target is the input `x`
  itself (pure autoencoding of clean masks). `SyntheticDegrader` appears **only** in
  `evaluate_correlation_3d.build_eval_set` (`:160-186`), i.e. test/eval time.

**Verdict: correct — the model never sees degraded masks in training.**

### 1c. Where is the SDF→binary comparison? — **after the bridge, in binary space**

- Bridge: `quality/sdf.py:50-59` `sdf_to_binary(sdf) = (sdf < 0)` (inside).
- Training val Dice: `scripts/train_vae3d_vq_sdf.py:113-114` —
  `pred_bin = (pred_sdf < 0).float()` then `_hard_dice_from_binary(pred_bin, x)`.
- Evaluation: `models/vae3d_vq_sdf.py:52-58` overrides `reconstruct` to
  `sdf_to_binary(predict_sdf(x))`, so the existing `recon_dice_score` scores binary.

**Verdict: correct — Dice is always computed after `sdf<0`, in binary space.**

### 1d. Straight-through estimator `z_q = z + (z_q - z).detach()` — **correct**

- `models/vae3d_vq.py:92`. Forward value = `z + (z_q - z) = z_q` (the quantized code).
  Backward: `(z_q - z).detach()` has no gradient, so the gradient of `z_q` flows
  unchanged to `z` (the encoder output), bypassing the non-differentiable `argmin`.
  This is the canonical van den Oord (2017) straight-through estimator.

**Verdict: correct — canonical STE.**

## 2. `VectorQuantizer3D` reference

`models/vae3d_vq.py:44-102`. Hand-written (no MONAI available), following
**van den Oord et al., 2017, "Neural Discrete Representation Learning" (VQ-VAE)**,
generalised from a 2D feature map to a **3D spatial codebook** over a
`(B, C, d, h, w)` grid (here 4×4×4 = 64 positions per volume):

- **Nearest-neighbour lookup** (`vae3d_vq.py:77-84`): squared-L2 distance from each
  position's encoder vector to every codebook entry, then `argmin` selects the code.
- **Straight-through estimator** (`vae3d_vq.py:92`): see 1d.
- **Codebook + commitment loss** (`vae3d_vq.py:87-89`):
  `codebook_loss = MSE(z_q, z.detach())` moves the embeddings toward the encoder;
  `commitment_loss = MSE(z, z_q.detach())` (weighted by `commitment_cost`, default
  0.25) moves the encoder toward its chosen code. No KL term (VQ replaces it).
- **Collapse diagnostics** (`vae3d_vq.py:97-101`): perplexity `exp(-Σ p·log p)` and
  `codebook_used`; logged every epoch as the codebook-collapse early warning.

## 3. Pending changes (dev branch)

Planned experiments, to be developed on `dev` so `main` stays the validated baseline:

- **Quantizer objective tweak** — drop the explicit `codebook_loss` and set
  `commitment_cost = 1` (relying on EMA-style codebook updates / a single-term VQ
  objective) to test whether it improves codebook usage without hurting rho.
- **WP-4: multi-candidate / uncertainty** — sample multiple candidate masks from the
  VQ-VAE (distance-weighted probabilistic quantization over the codebook, temperature
  `T`, `T→0` = current deterministic argmin) to produce a **consensus mask** + a **3D
  uncertainty map**, then validate that the uncertainty overlaps the truly-degraded
  regions (`degraded XOR clean`). Inference-only; no retraining of `main`.
