"""Patient-based train/val/test splitting.

Splitting is by *patient*, never by slice: the same patient must never appear in
two splits (that would leak information from train into val/test). The split is
deterministic given a seed, and once written to disk it is never silently
regenerated, so experiments stay comparable across runs.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_RATIO_TOLERANCE = 1e-6
_SPLIT_NAMES = ("train", "val", "test")


def make_splits(
    patient_ids: list[str], ratios: dict[str, float], seed: int
) -> dict[str, list[str]]:
    """Split unique patient ids into train/val/test by the given ratios.

    Deterministic: ids are sorted, then shuffled with ``random.Random(seed)``.
    Sizes are computed by rounding train/val and giving the remainder to test, so
    every id lands in exactly one split (no loss, no double counting).

    Raises ``ValueError`` if the ratios do not sum to 1.0 or a key is missing.
    """
    missing = set(_SPLIT_NAMES) - set(ratios)
    if missing:
        raise ValueError(f"ratios is missing keys: {sorted(missing)}")
    total = sum(ratios[name] for name in _SPLIT_NAMES)
    if abs(total - 1.0) > _RATIO_TOLERANCE:
        raise ValueError(f"ratios must sum to 1.0, got {total} ({ratios})")

    unique_ids = sorted(set(patient_ids))
    if len(unique_ids) != len(patient_ids):
        logger.warning(
            "make_splits received %d ids but only %d are unique; using unique set",
            len(patient_ids),
            len(unique_ids),
        )

    shuffled = list(unique_ids)
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * ratios["train"])
    n_val = round(n * ratios["val"])
    # Remainder goes to test so the three sizes always add up to n exactly.
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }
    _assert_no_leakage(splits, expected_total=n)
    logger.info(
        "made splits (seed=%d): train=%d val=%d test=%d",
        seed,
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )
    return splits


def _assert_no_leakage(splits: dict[str, list[str]], expected_total: int) -> None:
    """Assert the three splits are pairwise disjoint and cover every id once."""
    train, val, test = (set(splits[name]) for name in _SPLIT_NAMES)
    assert train.isdisjoint(val), "patient leakage between train and val"
    assert train.isdisjoint(test), "patient leakage between train and test"
    assert val.isdisjoint(test), "patient leakage between val and test"
    covered = len(train) + len(val) + len(test)
    assert covered == expected_total, (
        f"splits cover {covered} ids but expected {expected_total}"
    )


def save_splits(splits: dict[str, list[str]], path: str | Path, **metadata: Any) -> None:
    """Write splits to JSON with metadata (seed, ratios, timestamp, counts)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "counts": {name: len(splits[name]) for name in _SPLIT_NAMES},
            **metadata,
        },
        "splits": {name: splits[name] for name in _SPLIT_NAMES},
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("saved splits to %s", path.resolve())


def load_splits(path: str | Path) -> dict[str, list[str]]:
    """Load the split dict from a JSON file written by :func:`save_splits`."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return {name: list(payload["splits"][name]) for name in _SPLIT_NAMES}


def get_or_make_splits(
    patient_ids: list[str], config: dict[str, Any]
) -> dict[str, list[str]]:
    """Load splits from ``cache.splits_path`` if present, else make and save them.

    An existing split file is never silently regenerated (experiment
    consistency); delete it to force a new split.
    """
    splits_path = Path(config["cache"]["splits_path"])
    if splits_path.exists():
        logger.info("loading existing splits from %s", splits_path.resolve())
        return load_splits(splits_path)

    split_cfg = config["split"]
    ratios = {name: float(split_cfg[name]) for name in _SPLIT_NAMES}
    seed = int(split_cfg["seed"])
    splits = make_splits(patient_ids, ratios, seed)
    save_splits(splits, splits_path, seed=seed, ratios=ratios, by=split_cfg.get("by"))
    return splits
