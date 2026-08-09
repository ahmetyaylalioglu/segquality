"""Tests for data.degradations — synthetic masks only, no raw data required."""
from __future__ import annotations

import numpy as np

from data.degradations import (
    PRIMITIVES,
    DegradationParams,
    DegradationRecord,
    SyntheticDegrader,
    apply_random_degradation,
    dice,
    make_degraded_pair,
)


def make_disk(shape: tuple[int, int], cy: float, cx: float, radius: float) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2


def big_disk() -> np.ndarray:
    return make_disk((128, 128), 64, 64, 30)


def is_binary(mask: np.ndarray) -> bool:
    return bool(np.all(np.isin(np.unique(mask), (0, 1))))


# --- dice ------------------------------------------------------------------
def test_dice_identical_is_one():
    mask = big_disk()
    assert dice(mask, mask) == 1.0


def test_dice_disjoint_is_zero():
    a = np.zeros((20, 20), dtype=bool)
    b = np.zeros((20, 20), dtype=bool)
    a[0:5, 0:5] = True
    b[15:20, 15:20] = True
    assert dice(a, b) == 0.0


def test_dice_known_half_overlap():
    a = np.zeros((10, 20), dtype=bool)
    b = np.zeros((10, 20), dtype=bool)
    a[:, 0:10] = True
    b[:, 5:15] = True
    assert abs(dice(a, b) - 0.5) < 1e-9


def test_dice_both_empty_is_one():
    z = np.zeros((10, 10), dtype=bool)
    assert dice(z, z) == 1.0


# --- primitives ------------------------------------------------------------
def test_every_primitive_changes_mask_and_stays_binary():
    mask = big_disk()
    for name, fn in PRIMITIVES.items():
        out = fn(mask, 0.5, np.random.default_rng(0))
        assert is_binary(out), f"{name} produced a non-binary mask"
        assert out.any(), f"{name} produced an empty mask at severity 0.5"
        assert not np.array_equal(out, mask), f"{name} did not change the mask"


def test_severity_is_roughly_monotonic():
    mask = big_disk()
    for name in ("erode", "translate", "boundary_noise"):
        fn = PRIMITIVES[name]
        low = dice(mask, fn(mask, 0.15, np.random.default_rng(0)))
        high = dice(mask, fn(mask, 0.6, np.random.default_rng(0)))
        assert high < low, f"{name}: dice did not drop with severity ({high} !< {low})"


# --- make_degraded_pair: inverse Dice targeting ----------------------------
def test_pair_dice_is_consistent_with_output():
    mask = big_disk()
    rng = np.random.default_rng(1)
    degraded, true_dice = make_degraded_pair(mask, rng, target_dice=0.8)
    assert is_binary(degraded)
    assert degraded.any()
    assert abs(dice(mask, degraded) - true_dice) < 1e-9


def test_pair_hits_requested_targets():
    mask = big_disk()
    for target in (0.6, 0.75, 0.9):
        # Average over a few RNGs; binary search should land close to target.
        errors = []
        for seed in range(5):
            _, achieved = make_degraded_pair(mask, np.random.default_rng(seed), target)
            errors.append(abs(achieved - target))
        assert min(errors) < 0.1, f"target {target}: best error {min(errors):.3f}"


def test_pair_distribution_spreads_across_range():
    # Condition 3: true Dice must spread over ~[0.4, 0.99], not pile up near 1.0.
    mask = big_disk()
    rng = np.random.default_rng(7)
    dices = [make_degraded_pair(mask, rng)[1] for _ in range(300)]
    assert min(dices) < 0.6, f"distribution too high: min={min(dices):.3f}"
    assert max(dices) > 0.9, f"distribution too low: max={max(dices):.3f}"
    assert np.std(dices) > 0.1, "distribution too narrow to discriminate quality"


# --- determinism / frozen eval (condition 5) -------------------------------
def test_same_seed_gives_identical_degradation():
    mask = big_disk()
    d1, dice1 = make_degraded_pair(mask, np.random.default_rng(123))
    d2, dice2 = make_degraded_pair(mask, np.random.default_rng(123))
    assert np.array_equal(d1, d2)
    assert dice1 == dice2


def test_record_replays_exactly():
    mask = big_disk()
    degrader = SyntheticDegrader()
    record = degrader.sample_record(
        mask, np.random.default_rng(42), patient_id="p0", slice_index=5
    )
    replayed = degrader.apply_record(mask, record)
    assert abs(dice(mask, replayed) - record.true_dice) < 1e-9
    # A second replay is identical too.
    assert np.array_equal(replayed, degrader.apply_record(mask, record))


def test_apply_random_degradation_is_binary_and_nonempty():
    mask = big_disk()
    out = apply_random_degradation(mask, np.random.default_rng(3))
    assert is_binary(out)
    assert out.any()


# --- degenerate-output protection (condition 4) ----------------------------
def test_pairs_are_never_degenerate():
    mask = big_disk()
    rng = np.random.default_rng(11)
    params = DegradationParams()
    for _ in range(200):
        degraded, achieved = make_degraded_pair(mask, rng)
        assert degraded.any(), "degraded mask is empty"
        assert not degraded.all(), "degraded mask is completely filled"
        assert achieved >= params.min_valid_dice - 1e-9


# --- edge cases ------------------------------------------------------------
def test_single_pixel_mask_does_not_crash():
    mask = np.zeros((32, 32), dtype=bool)
    mask[16, 16] = True
    degraded, achieved = make_degraded_pair(mask, np.random.default_rng(0))
    assert is_binary(degraded)
    assert degraded.any()
    assert 0.0 <= achieved <= 1.0


def test_two_disjoint_components_mask():
    mask = np.zeros((64, 64), dtype=bool)
    mask |= make_disk((64, 64), 16, 16, 8)
    mask |= make_disk((64, 64), 48, 48, 8)
    degraded, achieved = make_degraded_pair(mask, np.random.default_rng(2))
    assert is_binary(degraded)
    assert degraded.any()


def test_edge_touching_mask():
    mask = np.zeros((64, 64), dtype=bool)
    mask[0:20, 0:64] = True  # runs along the top edge
    degraded, achieved = make_degraded_pair(mask, np.random.default_rng(4))
    assert is_binary(degraded)
    assert degraded.any()


def test_completely_filled_mask():
    mask = np.ones((48, 48), dtype=bool)
    degraded, achieved = make_degraded_pair(mask, np.random.default_rng(5))
    assert is_binary(degraded)
    assert degraded.any()
    assert not degraded.all(), "a fully-filled input should be degraded to non-full"


# --- A fix: chaining, reachable-type selection, target_reached -------------
def _irregular_liver() -> np.ndarray:
    """A big, lobed shape (two overlapping disks) closer to real liver geometry."""
    mask = make_disk((128, 128), 60, 55, 34)
    mask |= make_disk((128, 128), 70, 90, 22)
    return mask


def test_low_target_reached_via_chaining():
    # Single primitives floor out well above 0.45 on a big shape; chaining should
    # push Dice down to the low target and mark it reached.
    mask = _irregular_liver()
    degrader = SyntheticDegrader()
    best = None
    for seed in range(8):
        rec = degrader.sample_record(
            mask, np.random.default_rng(seed), patient_id="p", slice_index=0,
            target_dice=0.45,
        )
        if best is None or abs(rec.true_dice - 0.45) < abs(best - 0.45):
            best = rec.true_dice
        if rec.target_reached:
            assert len(rec.steps) >= 1
            assert rec.true_dice >= DegradationParams().min_valid_dice - 1e-9
            return
    raise AssertionError(f"low target 0.45 never reached (best achieved {best:.3f})")


def test_chained_output_is_not_degenerate():
    mask = _irregular_liver()
    degrader = SyntheticDegrader()
    for seed in range(10):
        rec = degrader.sample_record(
            mask, np.random.default_rng(seed), patient_id="p", slice_index=0,
            target_dice=0.42,
        )
        replayed = degrader.apply_record(mask, rec)
        assert replayed.any(), "chained mask collapsed to empty"
        assert not replayed.all(), "chained mask filled the frame"
        assert dice(mask, replayed) >= DegradationParams().min_valid_dice - 1e-9


def test_chained_record_replays_exactly():
    mask = _irregular_liver()
    degrader = SyntheticDegrader()
    # Seek a record that actually chained (len(steps) > 1), then verify replay.
    for seed in range(20):
        rec = degrader.sample_record(
            mask, np.random.default_rng(seed), patient_id="p", slice_index=0,
            target_dice=0.45,
        )
        if rec.steps and len(rec.steps) > 1:
            replayed = degrader.apply_record(mask, rec)
            assert abs(dice(mask, replayed) - rec.true_dice) < 1e-9
            assert np.array_equal(replayed, degrader.apply_record(mask, rec))
            return
    raise AssertionError("no chained (multi-step) record produced to test replay")


def test_legacy_single_step_record_still_replays():
    # A record with no `steps` (older format) must replay from the flat fields.
    mask = big_disk()
    degrader = SyntheticDegrader()
    seed = 7
    name, severity = "erode", 0.4
    expected = degrader.apply(name, mask, severity, np.random.default_rng(seed))
    legacy = DegradationRecord(
        patient_id="p", slice_index=0, degradation_type=name,
        severity=severity, seed=seed, true_dice=dice(mask, expected),
    )
    assert legacy.steps is None  # legacy shape
    replayed = degrader.apply_record(mask, legacy)
    assert np.array_equal(replayed, expected)


def test_target_reached_flag_matches_tolerance():
    mask = _irregular_liver()
    degrader = SyntheticDegrader()
    tol = degrader.params.dice_tolerance
    for seed in range(15):
        rec = degrader.sample_record(
            mask, np.random.default_rng(seed), patient_id="p", slice_index=0,
        )
        assert rec.target_reached == (abs(rec.true_dice - rec.target_dice) <= tol)


def test_calibrate_reports_ranges_for_every_primitive():
    masks = [big_disk(), _irregular_liver()]
    ranges = SyntheticDegrader().calibrate(masks)
    assert set(ranges) == set(PRIMITIVES)
    for name, (lo, hi) in ranges.items():
        assert 0.0 <= lo <= hi <= 1.0, f"{name}: bad range {(lo, hi)}"


def test_boundary_noise_is_light_only_after_calibration():
    # With measured ranges, boundary_noise must never be the first step for a low
    # target (it would fragment the mask), but may be for a high target.
    degrader = SyntheticDegrader()
    degrader.type_ranges = {
        "erode": (0.04, 0.98),
        "translate": (0.0, 0.98),
        "boundary_noise": (0.50, 0.99),  # safe floor = 0.50+0.7*0.49 = 0.843
    }
    rng = np.random.default_rng(0)
    low = [degrader._choose_type(rng, 0.45) for _ in range(200)]
    assert "boundary_noise" not in low, "boundary_noise chosen for a low target"
    high = [degrader._choose_type(rng, 0.95) for _ in range(200)]
    assert "boundary_noise" in high, "boundary_noise never chosen even at high Dice"


def test_chaining_terminates_on_tiny_mask():
    # A near-empty mask cannot reach a very low target; the loop must stop
    # (honest target_reached=False), not hang.
    mask = np.zeros((32, 32), dtype=bool)
    mask[15:18, 15:18] = True  # 9 px
    degrader = SyntheticDegrader()
    rec = degrader.sample_record(
        mask, np.random.default_rng(0), patient_id="p", slice_index=0,
        target_dice=0.40,
    )
    assert len(rec.steps) <= degrader.params.max_chain_steps
    assert 0.0 <= rec.true_dice <= 1.0
