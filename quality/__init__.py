"""Quality scoring: turn a trained shape-VAE's reconstruction of a mask into a
ground-truth-free quality score (Phase 6).

The public surface is the score functions (``recon_dice_score`` et al.), the
``SCORE_FNS`` registry / ``compute_all_scores`` batch helper, and the correlation
utilities (``correlation`` / ``banded_correlation``) used by
``scripts/evaluate_correlation.py`` to test the core hypothesis: does
reconstruction error correlate with a mask's true Dice?
"""
from quality.score import (
    DICE_BANDS,
    SCORE_FNS,
    SCORE_DIRECTIONS,
    banded_correlation,
    compute_all_scores,
    correlation,
    latent_kl_score,
    recon_bce_score,
    recon_dice_score,
    recon_l1_score,
    topology_score,
)

__all__ = [
    "DICE_BANDS",
    "SCORE_FNS",
    "SCORE_DIRECTIONS",
    "banded_correlation",
    "compute_all_scores",
    "correlation",
    "latent_kl_score",
    "recon_bce_score",
    "recon_dice_score",
    "recon_l1_score",
    "topology_score",
]
