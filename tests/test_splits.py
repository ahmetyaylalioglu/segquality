"""Tests for data.splits — synthetic patient ids only, no raw data required."""
from __future__ import annotations

from data.splits import load_splits, make_splits, save_splits

RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


def ids(n: int) -> list[str]:
    return [f"patient_{i:04d}" for i in range(n)]


def test_deterministic_same_seed():
    a = make_splits(ids(100), RATIOS, seed=42)
    b = make_splits(ids(100), RATIOS, seed=42)
    assert a == b


def test_no_leakage_between_splits():
    splits = make_splits(ids(250), RATIOS, seed=0)
    train, val, test = (set(splits[k]) for k in ("train", "val", "test"))
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    # Every id is covered exactly once.
    assert len(train) + len(val) + len(test) == 250
    assert train | val | test == set(ids(250))


def test_ratio_accuracy():
    n = 1000
    splits = make_splits(ids(n), RATIOS, seed=1)
    assert abs(len(splits["train"]) - 700) <= 2
    assert abs(len(splits["val"]) - 150) <= 2
    assert abs(len(splits["test"]) - 150) <= 2


def test_different_seed_gives_different_split():
    a = make_splits(ids(200), RATIOS, seed=1)
    b = make_splits(ids(200), RATIOS, seed=2)
    assert a != b


def test_save_load_round_trip(tmp_path):
    splits = make_splits(ids(120), RATIOS, seed=7)
    path = tmp_path / "splits.json"
    save_splits(splits, path, seed=7, ratios=RATIOS)
    assert load_splits(path) == splits


def test_bad_ratios_raise():
    for bad in ({"train": 0.7, "val": 0.15, "test": 0.2}, {"train": 0.5, "val": 0.2, "test": 0.1}):
        try:
            make_splits(ids(10), bad, seed=0)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for ratios {bad}")


def test_missing_ratio_key_raises():
    try:
        make_splits(ids(10), {"train": 0.8, "val": 0.2}, seed=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for missing 'test' key")


def test_duplicate_ids_are_deduplicated():
    splits = make_splits(ids(50) + ids(50), RATIOS, seed=3)
    total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total == 50
