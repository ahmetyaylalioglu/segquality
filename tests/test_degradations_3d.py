"""Tests for data.degradations_3d and the dimension-agnostic SyntheticDegrader.

Synthetic 3D masks only (solid balls / two-lobe shapes), no raw data required.
These mirror tests/test_degradations.py one axis higher and, crucially, prove:
* the four 3D geometric primitives edit a volume, stay binary, stay non-empty;
* the SAME SyntheticDegrader engine, handed PRIMITIVES_3D, calibrates / targets /
  chains / guards on volumes exactly as it does on slices;
* passing ``primitives=None`` leaves the engine bound to the 2D PRIMITIVES
  (behaviour-preserving — the guarantee that the 97 existing 2D tests stay green).
"""
from __future__ import annotations

import numpy as np

from data.degradations import (
    PRIMITIVES,
    DegradationParams,
    SyntheticDegrader,
    dice,
)
from data.degradations_3d import PRIMITIVES_3D, _ball


def make_ball(shape: tuple[int, int, int], center, radius: float) -> np.ndarray:
    return _ball(shape, center, radius)


def big_ball() -> np.ndarray:
    return make_ball((48, 48, 48), (24, 24, 24), 15)


def two_lobe() -> np.ndarray:
    """Two overlapping balls — a lumpier, liver-ish 3D shape for chaining tests."""
    mask = make_ball((48, 48, 48), (22, 20, 24), 13)
    mask |= make_ball((48, 48, 48), (28, 32, 24), 9)
    return mask


def is_binary(mask: np.ndarray) -> bool:
    return bool(np.all(np.isin(np.unique(mask), (0, 1))))


def _degrader_3d(params: DegradationParams | None = None) -> SyntheticDegrader:
    return SyntheticDegrader(params, primitives=PRIMITIVES_3D)


# --- backward-compat guarantee ---------------------------------------------
def test_default_primitives_is_2d_set():
    # primitives=None must keep the exact 2D module-global set — this is what
    # keeps every existing 2D caller and test byte-for-byte unchanged.
    assert SyntheticDegrader().primitives is PRIMITIVES
    assert SyntheticDegrader.from_config({}).primitives is PRIMITIVES


def test_primitives_3d_has_same_keys_as_2d():
    # Name-based module constants (_STRONG_TYPES/_CHAIN_TYPES/_LIGHT_ONLY_TYPES)
    # rely on the 3D set using identical type names.
    assert set(PRIMITIVES_3D) == set(PRIMITIVES)


# --- dice in 3D ------------------------------------------------------------
def test_dice_identical_is_one_3d():
    mask = big_ball()
    assert dice(mask, mask) == 1.0


def test_dice_disjoint_is_zero_3d():
    a = np.zeros((16, 16, 16), dtype=bool)
    b = np.zeros((16, 16, 16), dtype=bool)
    a[0:4, 0:4, 0:4] = True
    b[12:16, 12:16, 12:16] = True
    assert dice(a, b) == 0.0


# --- primitives ------------------------------------------------------------
def test_every_primitive_3d_changes_mask_and_stays_binary():
    mask = big_ball()
    for name, fn in PRIMITIVES_3D.items():
        out = fn(mask, 0.5, np.random.default_rng(0))
        assert is_binary(out), f"{name} produced a non-binary volume"
        assert out.any(), f"{name} produced an empty volume at severity 0.5"
        assert not np.array_equal(out, mask), f"{name} did not change the volume"
        assert out.shape == mask.shape, f"{name} changed the volume shape"


def test_severity_is_roughly_monotonic_3d():
    mask = big_ball()
    for name in ("erode", "translate", "boundary_noise"):
        fn = PRIMITIVES_3D[name]
        low = dice(mask, fn(mask, 0.15, np.random.default_rng(0)))
        high = dice(mask, fn(mask, 0.6, np.random.default_rng(0)))
        assert high < low, f"{name}: dice did not drop with severity ({high} !< {low})"


def test_ball_helper_is_spherical():
    shape = (32, 32, 32)
    ball = _ball(shape, (16, 16, 16), 8)
    # A voxel at radius 8 along an axis is inside; one at radius 9 is outside.
    assert ball[16 + 8, 16, 16]
    assert not ball[16 + 9, 16, 16]
    # Corner voxel (far outside) is empty.
    assert not ball[0, 0, 0]


# --- engine on volumes: targeting, spread, determinism ---------------------
def test_pair_dice_is_consistent_with_output_3d():
    mask = big_ball()
    degrader = _degrader_3d()
    degraded, true_dice = degrader.make_pair(mask, np.random.default_rng(1), target_dice=0.8)
    assert is_binary(degraded)
    assert degraded.any()
    assert abs(dice(mask, degraded) - true_dice) < 1e-9


def test_pair_hits_requested_targets_3d():
    mask = two_lobe()
    degrader = _degrader_3d()
    for target in (0.6, 0.75, 0.9):
        errors = []
        for seed in range(6):
            _, achieved = degrader.make_pair(mask, np.random.default_rng(seed), target)
            errors.append(abs(achieved - target))
        assert min(errors) < 0.1, f"target {target}: best error {min(errors):.3f}"


def test_pair_distribution_spreads_across_range_3d():
    mask = two_lobe()
    degrader = _degrader_3d()
    rng = np.random.default_rng(7)
    dices = [degrader.make_pair(mask, rng)[1] for _ in range(200)]
    assert min(dices) < 0.65, f"distribution too high: min={min(dices):.3f}"
    assert max(dices) > 0.9, f"distribution too low: max={max(dices):.3f}"
    assert np.std(dices) > 0.08, "distribution too narrow to discriminate quality"


def test_same_seed_gives_identical_degradation_3d():
    mask = big_ball()
    degrader = _degrader_3d()
    d1, dice1 = degrader.make_pair(mask, np.random.default_rng(123))
    d2, dice2 = degrader.make_pair(mask, np.random.default_rng(123))
    assert np.array_equal(d1, d2)
    assert dice1 == dice2


def test_record_replays_exactly_3d():
    mask = two_lobe()
    degrader = _degrader_3d()
    record = degrader.sample_record(
        mask, np.random.default_rng(42), patient_id="p0", slice_index=0
    )
    replayed = degrader.apply_record(mask, record)
    assert abs(dice(mask, replayed) - record.true_dice) < 1e-9
    assert np.array_equal(replayed, degrader.apply_record(mask, record))


# --- degenerate-output protection ------------------------------------------
def test_pairs_are_never_degenerate_3d():
    mask = two_lobe()
    degrader = _degrader_3d()
    rng = np.random.default_rng(11)
    floor = degrader.params.min_valid_dice
    for _ in range(120):
        degraded, achieved = degrader.make_pair(mask, rng)
        assert degraded.any(), "degraded volume is empty"
        assert not degraded.all(), "degraded volume is completely filled"
        assert achieved >= floor - 1e-9


# --- calibration -----------------------------------------------------------
def test_calibrate_reports_ranges_for_every_primitive_3d():
    masks = [big_ball(), two_lobe()]
    ranges = _degrader_3d().calibrate(masks)
    assert set(ranges) == set(PRIMITIVES_3D)
    for name, (lo, hi) in ranges.items():
        assert 0.0 <= lo <= hi <= 1.0, f"{name}: bad range {(lo, hi)}"


# --- edge cases ------------------------------------------------------------
def test_single_voxel_mask_does_not_crash_3d():
    mask = np.zeros((16, 16, 16), dtype=bool)
    mask[8, 8, 8] = True
    degraded, achieved = _degrader_3d().make_pair(mask, np.random.default_rng(0))
    assert is_binary(degraded)
    assert degraded.any()
    assert 0.0 <= achieved <= 1.0


def test_completely_filled_volume_3d():
    mask = np.ones((24, 24, 24), dtype=bool)
    degraded, achieved = _degrader_3d().make_pair(mask, np.random.default_rng(5))
    assert is_binary(degraded)
    assert degraded.any()
    assert not degraded.all(), "a fully-filled input should be degraded to non-full"
