"""Standalone mean-reversion diagnosis for the trained 3D VQ-VAE (Aşama 9 kanıtı).

The standard continuous-latent ``ConvVAE3D`` was shown to mean-revert
(``ratio = 1.005``, recon-pairwise Dice ``0.992`` vs input-pairwise ``0.733``):
its decoder emitted a generic mean shape regardless of the input. The VQ-VAE with a
discrete *spatial* codebook was built to cure this. This script produces the
*definitive* apples-to-apples proof by running the EXACT same measurement on the
trained VQ-VAE checkpoint.

It touches NO training/model code and, crucially, imports the single-source
diagnosis primitives (:func:`select_indices`, :func:`load_val_volumes`,
:func:`meanreversion_stats`, :func:`cross_recon_matrix`, :func:`plot_matrix`,
:func:`plot_labeled_preview`) straight from ``diagnose_reconstruction.py`` — the very
functions the standard-VAE diagnosis used. Same K, same seed, same ``dice``, same
matrix. The only difference is the model loader (ConvVQVAE3D) and an added codebook
perplexity/usage readout (VQ's own collapse trap).

Usage
-----
    python scripts/diagnose_reconstruction_vq.py \
        --checkpoint outputs/checkpoints/vae3d_vqvae_best.pt \
        --config configs/vae3d_vq.yaml --k 20 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data import load_config  # noqa: E402
from models.vae3d_vq import ConvVQVAE3D  # noqa: E402

# single-source diagnosis primitives — identical computation to the standard-VAE run
from diagnose_reconstruction import (  # noqa: E402
    load_val_volumes,
    meanreversion_stats,
    plot_labeled_preview,
    plot_matrix,
)

logger = logging.getLogger("diagnose_reconstruction_vq")


def load_model(checkpoint: Path, device: torch.device) -> ConvVQVAE3D:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    vcfg = ckpt["config"]["vae3d_vq"]
    model = ConvVQVAE3D(
        base_channels=int(vcfg["base_channels"]),
        num_embeddings=int(vcfg["num_embeddings"]),
        embedding_dim=int(vcfg["embedding_dim"]),
        commitment_cost=float(vcfg["commitment_cost"]),
        input_size=tuple(vcfg["input_size"]),
        use_skip=bool(vcfg.get("use_skip", False)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    logger.info(
        "loaded %s | ConvVQVAE3D codebook=%dx%d use_skip=%s input_size=%s (epoch %s)",
        checkpoint.name, model.num_embeddings, model.embedding_dim,
        model.use_skip, model.input_size, ckpt.get("epoch"),
    )
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose 3D VQ-VAE reconstruction mean-reversion.")
    p.add_argument("--checkpoint", default="outputs/checkpoints/vae3d_vqvae_best.pt")
    p.add_argument("--config", default="configs/vae3d_vq.yaml")
    p.add_argument("--split", default="val")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--baseline-json", default="outputs/diagnose_reconstruction.json",
                   help="standard-VAE diagnosis JSON for the side-by-side table")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # Turkish output on Windows cp1252
    except Exception:
        pass
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    model = load_model(PROJECT_ROOT / args.checkpoint, device)
    # load_val_volumes only needs model.input_size (duck-typed) -> same K livers as
    # the standard-VAE run for this (n, k, seed).
    masks, pids = load_val_volumes(config, model, args.split, args.k, args.seed)

    x = torch.from_numpy(masks).unsqueeze(1).to(device)
    with torch.no_grad():
        logits, _vq, info = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[:, 0]
    inputs_bin = masks > 0.5
    recon_bin = probs > 0.5

    # identical single-source aggregator -> ratio directly comparable to 1.005
    mr = meanreversion_stats(recon_bin, inputs_bin)
    m = mr["matrix"]
    diag = mr["self_recon_dice"]
    diag_mean, off_mean, ratio = mr["diag_mean"], mr["offdiag_mean"], mr["ratio"]
    input_pairwise = mr["input_pairwise_dice"]
    recon_pairwise = mr["recon_pairwise_dice"]

    perplexity = float(info["perplexity"])
    codebook_used = int(info["codebook_used"])
    usage_pct = 100.0 * codebook_used / model.num_embeddings

    # figures (regenerate the heatmap; VQ variants so the standard-VAE files stay put)
    plot_matrix(m, pids, out_dir / "cross_recon_matrix_vq.png")
    plot_labeled_preview(masks, probs, pids, diag, out_dir / "recon_labeled_vq.png")

    mean_reversion = ratio < 1.15
    result = {
        "checkpoint": str(args.checkpoint),
        "model": "ConvVQVAE3D",
        "split": args.split, "k": args.k, "seed": args.seed,
        "patient_ids": pids,
        "self_recon_dice": diag.tolist(),
        "self_recon_dice_mean": diag_mean,
        "diag_mean": diag_mean, "offdiag_mean": off_mean, "ratio": ratio,
        "input_pairwise_dice": input_pairwise,
        "recon_pairwise_dice": recon_pairwise,
        "codebook": {
            "num_embeddings": model.num_embeddings,
            "perplexity": perplexity,
            "codebook_used": codebook_used,
            "usage_pct": usage_pct,
        },
        "mean_reversion_detected": bool(mean_reversion),
    }
    (out_dir / "diagnose_reconstruction_vq.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    logger.info("wrote %s", out_dir / "diagnose_reconstruction_vq.json")

    # -- side-by-side table -------------------------------------------------
    base = None
    base_path = PROJECT_ROOT / args.baseline_json
    if base_path.exists():
        base = json.loads(base_path.read_text(encoding="utf-8"))

    print("\n" + "=" * 72)
    print("3D VQ-VAE RECONSTRUCTION DIAGNOSIS (mean-reversion kanıtı)")
    print("=" * 72)
    print(f"livers (K)={args.k}  split='{args.split}'  seed={args.seed}")
    print(f"codebook perplexity : {perplexity:.1f}/{model.num_embeddings} "
          f"= {usage_pct:.1f}% ({codebook_used} kod aktif)")

    def _row(label: str, vq_val: str, base_val: str) -> str:
        return f"  {label:<40}{base_val:>14}{vq_val:>14}"

    print("\n" + "-" * 72)
    print(f"  {'metric':<40}{'standart-VAE':>14}{'VQ-VAE':>14}")
    print("-" * 72)
    bget = (lambda k: f"{base[k]:.4f}") if base else (lambda k: "n/a")
    print(_row("self-recon Dice (diag mean)", f"{diag_mean:.4f}", bget("diag_mean")))
    print(_row("off-diagonal Dice mean", f"{off_mean:.4f}", bget("offdiag_mean")))
    print(_row("cross-recon ratio = diag/off", f"{ratio:.4f}", bget("ratio")))
    print(_row("input-pairwise Dice", f"{input_pairwise:.4f}", bget("input_pairwise_dice")))
    print(_row("recon-pairwise Dice", f"{recon_pairwise:.4f}", bget("recon_pairwise_dice")))
    print("-" * 72)

    print("\nYORUM")
    print("-" * 72)
    if base:
        d_ratio = ratio - base["ratio"]
        d_rpw = recon_pairwise - base["recon_pairwise_dice"]
        print(f"  ratio           : {base['ratio']:.4f} -> {ratio:.4f}  ({d_ratio:+.4f})")
        print(f"  recon-pairwise  : {base['recon_pairwise_dice']:.4f} -> {recon_pairwise:.4f}"
              f"  ({d_rpw:+.4f})   [hedef: input-pairwise {input_pairwise:.4f}]")
    if not mean_reversion:
        print(f"  ratio = {ratio:.3f} > 1.15  ->  MEAN-REVERSION COZULDU. Her karaciger")
        print("  kendi girdisini digerlerinden belirgin sekilde daha iyi reconstruct")
        print("  ediyor; ayrik uzamsal kodlar ortalama sekle cokmuyor.")
    else:
        print(f"  ratio = {ratio:.3f} ~ 1  ->  mean-reversion hala var; VQ tek basina")
        print("  yetmedi. Sonraki kaldirac: use_skip=True (encoder detayini decoder'a zorla).")
    if recon_pairwise <= input_pairwise + 0.05:
        print(f"  recon-pairwise ({recon_pairwise:.3f}) artik input-pairwise "
              f"({input_pairwise:.3f}) seviyesinde: cesitlilik geri geldi.")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
