"""Tests for the pure training helpers in train.py.

These cover the pieces the training loop's correctness hinges on but that are
easy to get subtly wrong: the KL warmup schedule, the early-stopping
"improvement" rule (with its ``min_delta`` dead-band), the hard reconstruction
Dice, and the blur/sharpness metric added to catch a VAE that collapses masks to
a mean grey blob. All run on tiny CPU tensors — no data, no GPU.
"""
from __future__ import annotations

import torch

from train import beta_at, hard_dice, improved, sharpness_penalty


def test_beta_at_warmup_then_holds() -> None:
    assert beta_at(0, 0.1, 5) == 0.1 * (1 / 5)
    assert beta_at(4, 0.1, 5) == 0.1              # last warmup epoch reaches target
    assert beta_at(20, 0.1, 5) == 0.1             # held after warmup
    assert beta_at(0, 0.1, 0) == 0.1              # warmup disabled -> target immediately


def test_improved_respects_min_delta() -> None:
    assert improved(0.50, 0.40, 0.001) is True
    assert improved(0.4005, 0.40, 0.001) is False   # 0.0005 < min_delta: not an improvement
    assert improved(0.4020, 0.40, 0.001) is True    # 0.002 > min_delta
    assert improved(0.40, 0.40, 0.0) is False       # equal is never an improvement


def test_sharpness_penalty_zero_when_crisp_half_when_blurred() -> None:
    crisp = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    blurred = torch.full((2, 2), 0.5)
    assert sharpness_penalty(crisp).item() == 0.0
    assert abs(sharpness_penalty(blurred).item() - 0.5) < 1e-6
    # A partly-confident map sits between the two extremes.
    mid = torch.tensor([[0.9, 0.1], [0.5, 0.5]])
    val = sharpness_penalty(mid).item()
    assert 0.0 < val < 0.5


def test_hard_dice_perfect_and_disjoint() -> None:
    target = torch.zeros(1, 1, 8, 8)
    target[..., 2:6, 2:6] = 1.0
    perfect = torch.where(target > 0.5, 0.99, 0.01)  # probs threshold to the target
    assert hard_dice(perfect, target).item() > 0.99
    # Reconstruction that misses the organ entirely (all background) scores low.
    empty = torch.zeros_like(target) + 0.01
    assert hard_dice(empty, target).item() < 0.1
