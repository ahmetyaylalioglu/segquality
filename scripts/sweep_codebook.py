"""Codebook-size ablation for the 3D VQ-VAE (Phase 9, WP-1).

Sweeps ``num_embeddings`` K in {128, 256, 512, 1024, 2048} and, for every K,
reports three numbers on the SAME exam:

* **recon Dice**   — best val reconstruction Dice (from the trained checkpoint).
* **codebook use** — perplexity and %usage (perplexity / K) at that best epoch.
* **rho**          — Spearman correlation of ``recon_dice`` (the primary GT-free
  score) against true Dice on the held-out *test* split.

Why this script only ADDS files
-------------------------------
* Training is delegated to ``scripts/train_vae3d_vq.py`` unchanged, via subprocess,
  passing the two new backward-compatible flags ``--num-embeddings K`` and
  ``--tag sweep_k{K}``. The per-K tag means each run writes its own
  ``vae3d_sweep_k{K}_best.pt`` and **never overwrites the canonical
  ``vae3d_vqvae_best.pt``** (the existing K=512 full run).
* rho is measured IN-PROCESS by importing the exact functions the canonical
  evaluator uses (``evaluate_correlation_3d`` for data/degrader/eval-set and
  ``evaluate_correlation_3d_vq`` for the model loader + VQ scores). Every K is
  scored with the identical ``--seed`` / ``--split`` / ``--n-per-mask``, so the
  degradation exam is byte-identical across K — the comparison is fair. This path
  does NOT write the canonical ``correlation_results_vq.json`` / PNGs, so no
  existing output is clobbered.

Outputs
-------
* ``outputs/codebook_ablation.csv`` — one row per K: K, recon_dice, perplexity,
  usage_pct, rho, quant_rho, n.
* ``outputs/codebook_ablation.png`` — recon Dice / usage% / rho vs log2(K).

Usage
-----
    # STEP-2 smoke: one K, one epoch, just prove the pipeline runs end-to-end
    python scripts/sweep_codebook.py --ks 128 --epochs 1 --train-limit 24

    # full sweep (user runs; ~30 min per K)
    python scripts/sweep_codebook.py --ks 128,256,512,1024,2048
"""
from __future__ import annotations

import argparse
import csv
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

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
from train import resolve_device, set_seed  # noqa: E402
import evaluate_correlation_3d as ev3d  # noqa: E402
import evaluate_correlation_3d_vq as ev3dvq  # noqa: E402

logger = logging.getLogger("sweep_codebook")

DEFAULT_KS = (128, 256, 512, 1024, 2048)


def train_one(k: int, config_path: str, epochs: int | None, train_limit: int | None) -> Path:
    """Train a single K via the unchanged trainer, tagged so it never clobbers the
    canonical checkpoint. Returns the path to that K's best checkpoint."""
    tag = f"sweep_k{k}"
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "train_vae3d_vq.py"),
        "--config", config_path,
        "--num-embeddings", str(k),
        "--tag", tag,
        # num_workers=0: no DataLoader worker spawn -> avoids the Windows
        # "WinError 6 / c10_cuda.dll" crash when training runs as a subprocess.
        "--num-workers", "0",
    ]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    if train_limit is not None:
        cmd += ["--limit", str(train_limit)]
    logger.info("[K=%d] training: %s", k, " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    ckpt = PROJECT_ROOT / "outputs" / "checkpoints" / f"vae3d_{tag}_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"[K={k}] expected checkpoint not found: {ckpt}")
    return ckpt


def read_train_metrics(checkpoint: Path) -> dict[str, float]:
    """recon Dice + codebook usage recorded at the best epoch inside the checkpoint."""
    meta = torch.load(checkpoint, map_location="cpu", weights_only=False)["metrics"]
    return {
        "recon_dice": float(meta["recon_dice"]),
        "perplexity": float(meta["perplexity"]),
        "codebook_used": float(meta.get("codebook_used", float("nan"))),
    }


def eval_rho(
    checkpoint: Path, config: dict[str, Any], device: torch.device,
    seed: int, split: str, n_per_mask: int,
) -> dict[str, float]:
    """Spearman rho of recon_dice (and quant_error) vs true Dice, computed with the
    SAME functions and SAME seed the canonical evaluator uses -> identical exam for
    every K. Non-destructive: writes no files."""
    set_seed(seed)
    model = ev3dvq.load_model(checkpoint, device)
    clean = ev3d.load_clean_volumes(config, split, None, seed)
    degrader = ev3d.calibrate_degrader(config, clean, seed)
    masks, true_dice, _ = ev3d.build_eval_set(clean, degrader, n_per_mask, seed)
    scores = ev3dvq.score_masks_vq(model, masks, device)
    summary = ev3dvq.summarise_vq(scores, true_dice)
    return {
        "rho": float(summary["recon_dice"]["global"]["spearman"]),
        "quant_rho": float(summary["quant_error"]["global"]["spearman"]),
        "n": int(len(true_dice)),
    }


CSV_COLS = ["K", "recon_dice", "perplexity", "usage_pct", "rho", "quant_rho", "n"]


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in CSV_COLS})
    logger.info("wrote %s", out_path)


def read_existing_rows(path: Path) -> dict[int, dict[str, Any]]:
    """Load already-completed K rows so an interrupted sweep can resume. A K present
    here (with a real recon_dice) is treated as done and skipped."""
    if not path.exists():
        return {}
    done: dict[int, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                done[int(r["K"])] = {
                    "K": int(r["K"]),
                    "recon_dice": float(r["recon_dice"]),
                    "perplexity": float(r["perplexity"]),
                    "usage_pct": float(r["usage_pct"]),
                    "rho": float(r["rho"]),
                    "quant_rho": float(r["quant_rho"]),
                    "n": int(r["n"]),
                }
            except (KeyError, ValueError):
                continue
    return done


def plot_sweep(rows: list[dict[str, Any]], out_path: Path) -> None:
    ks = [r["K"] for r in rows]
    x = np.log2(ks)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(x, [r["recon_dice"] for r in rows], marker="o", color="C2")
    axes[0].set_title("val reconstruction Dice"); axes[0].set_ylim(0, 1)
    axes[1].plot(x, [r["usage_pct"] for r in rows], marker="o", color="C4")
    axes[1].set_title("codebook usage % (perplexity / K)")
    axes[2].plot(x, [r["rho"] for r in rows], marker="o", color="C0", label="recon_dice")
    axes[2].plot(x, [r["quant_rho"] for r in rows], marker="s", color="C1", label="quant_error")
    axes[2].set_title("QC Spearman rho vs true Dice"); axes[2].legend()
    for ax in axes:
        ax.set_xlabel("codebook size K (log2 axis)")
        ax.set_xticks(x); ax.set_xticklabels([str(k) for k in ks])
        ax.grid(alpha=0.3)
    fig.suptitle("3D VQ-VAE codebook-size ablation", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Codebook-size ablation for the 3D VQ-VAE.")
    p.add_argument("--config", default="configs/vae3d_vq.yaml")
    p.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS),
                   help="comma-separated codebook sizes, e.g. 128,256,512,1024,2048")
    p.add_argument("--epochs", type=int, default=None, help="override train epochs (smoke: 1)")
    p.add_argument("--train-limit", type=int, default=None, help="cap volumes/split for a fast smoke")
    p.add_argument("--split", default="test")
    p.add_argument("--n-per-mask", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-train", action="store_true",
                   help="reuse existing vae3d_sweep_k{K}_best.pt checkpoints (eval only)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    device = resolve_device(config["train"]["device"])
    ks = [int(k) for k in args.ks.split(",") if k.strip()]
    out_dir = PROJECT_ROOT / "outputs"
    csv_path = out_dir / "codebook_ablation.csv"
    png_path = out_dir / "codebook_ablation.png"

    # Resume: any K already recorded in the CSV is skipped, so an interrupted 2.5h
    # sweep continues from where it stopped instead of restarting from K=128.
    results: dict[int, dict[str, Any]] = read_existing_rows(csv_path)
    if results:
        logger.info("resume: found %d completed K in %s -> %s",
                    len(results), csv_path.name, sorted(results))
    logger.info("codebook sweep K=%s | split=%s seed=%d n_per_mask=%d (same exam for every K)",
                ks, args.split, args.seed, args.n_per_mask)

    def flush() -> None:
        """Persist everything completed so far — called after EVERY K so a crash
        loses at most the K in flight."""
        ordered = [results[kk] for kk in sorted(results)]
        write_csv(ordered, csv_path)
        plot_sweep(ordered, png_path)

    for k in ks:
        if k in results:
            r = results[k]
            logger.info("[K=%d] already in CSV (recon_dice=%.4f rho=%.4f) -> skip (resume)",
                        k, r["recon_dice"], r["rho"])
            continue
        tag = f"sweep_k{k}"
        ckpt = PROJECT_ROOT / "outputs" / "checkpoints" / f"vae3d_{tag}_best.pt"
        if not args.skip_train:
            ckpt = train_one(k, args.config, args.epochs, args.train_limit)
        elif not ckpt.exists():
            raise FileNotFoundError(f"--skip-train but {ckpt} missing")
        tm = read_train_metrics(ckpt)
        ev = eval_rho(ckpt, config, device, args.seed, args.split, args.n_per_mask)
        results[k] = {
            "K": k,
            "recon_dice": round(tm["recon_dice"], 4),
            "perplexity": round(tm["perplexity"], 3),
            "usage_pct": round(100.0 * tm["perplexity"] / k, 2),
            "rho": round(ev["rho"], 4),
            "quant_rho": round(ev["quant_rho"], 4),
            "n": ev["n"],
        }
        row = results[k]
        logger.info("[K=%d] recon_dice=%.4f perplexity=%.2f usage=%.1f%% rho=%.4f quant_rho=%.4f (n=%d)",
                    k, row["recon_dice"], row["perplexity"], row["usage_pct"],
                    row["rho"], row["quant_rho"], row["n"])
        flush()  # durable partial result after THIS K
        logger.info("[K=%d] partial results saved (%d K done in %s)", k, len(results), csv_path.name)

    flush()
    logger.info("done. %d K values in %s.", len(results), csv_path.name)


if __name__ == "__main__":
    main()
