# segquality — RAOS data pipeline

Ground-truth-free segmentation quality estimation. A shape-VAE is trained on
good anatomical masks; at test time a mask is passed through the VAE and its
reconstruction error is used as a quality score (good masks reconstruct well,
corrupted masks do not). **This repository currently implements only the data
pipeline** (Phase 1) — no model, training loop, or VAE code yet.

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux:    source .venv/bin/activate
pip install -r requirements.txt
```

No GPU is required for this phase (there is no training).

## Where the RAOS data goes

The raw dataset lives **outside** this repo; its location comes from the config,
not from code. Point `dataset.root` in `configs/raos.yaml` at your RAOS folder:

```yaml
dataset:
  root: "C:/Users/you/.../datasets/RAOS"   # use forward slashes in YAML
```

RAOS is organized in nnU-Net style with three subsets — `CancerImages(Set1)`,
`AfterSurgery(Set2)`, `MissingOrgans(Set3)` — each containing
`imagesTr/labelsTr`, `imagesVal/labelsVal`, `imagesTs/labelsTs`. Image and label
files share the same filename (a DICOM UID), so `patient_id` = filename stem.

Training uses only the clean anatomy in Set1. Set2/Set3 (post-surgical and
missing-organ cases) are the real evaluation ground for a quality model — they
are indexed but held out from training.

## Run order

Run each script with `--config configs/raos.yaml`.

1. **`scripts/inspect_raos.py`** — discovers file naming, label ids, spacings and
   the label-value histogram. Writes `outputs/raos_inspection.json` and prints a
   copy-paste-ready `organs:` YAML block.
2. **Fill `configs/raos.yaml`** — paste the discovered `organs:` map and set
   `target:` to the organ you want (e.g. `liver`).
3. **`scripts/build_index.py`** — precomputes a per-slice index for the target
   organ (skips slices below `min_organ_area`) so the dataset never opens a full
   NIfTI volume in `__getitem__`. Indexes both training and held-out subsets.
   Re-running is a no-op unless `--force` is given.
4. **`scripts/dataset_stats.py`** — per-split patient/slice counts, organ-area
   distribution, patient-leakage check, and the Dice distribution of 1000 random
   degradations. Writes `outputs/stats.json` and `outputs/dice_distribution.png`.
5. **`scripts/visualize_samples.py`** — renders `outputs/samples_{split}.png`: a
   grid of original masks alongside degraded versions with their true Dice.

## Testing

```bash
pytest
```

All tests use synthetic masks and pass **without** the raw data.

## Folder layout

```
segquality/
  configs/        default.yaml (common) + raos.yaml (inherits via _base_)
  data/           config loader (__init__), dataset, transforms, degradations, splits
  scripts/        inspect / build_index / dataset_stats / visualize entry points
  tests/          pytest suite (synthetic data only)
  outputs/        generated indexes, splits, stats and plots (git-ignored)
```

## Design notes

- All file access uses `pathlib.Path`; moving to a server only changes
  `dataset.root` in the config.
- The slicing `axis` is a parameter, never hard-coded, so the later move to 3D
  volumes is painless.
- Splits are patient-based and deterministic (fixed seed); every indexed slice
  records its `source_subset` and `original_split` for later comparison against
  published RAOS benchmarks.
- **Degradations act in resized (128×128) space** (`degradation.space: resized`),
  and true Dice is measured on that same resized pair — so the VAE input and the
  Dice label live in one space and no resize-induced difference leaks into the
  quality score as noise.
- Degradations follow a `mask -> (degraded_mask, true_dice)` interface
  (`DegradationSource`); the synthetic corruptor is one implementation, so real
  segmentation-model outputs can later be fed through the identical downstream
  code. Targets are drawn first and severity is solved by binary search, keeping
  true Dice spread across ~[0.4, 0.99] instead of piling up near 1.0.
