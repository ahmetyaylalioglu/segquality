"""Guard the RAOS config against drift: every organ id must exist in the
discovered inventory, and the target organ must be present in every patient.

Requires ``outputs/raos_inspection.json`` (produced by inspect_raos.py). On a
fresh checkout without the raw data this file is absent, so the test skips
rather than failing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from data import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "raos.yaml"
INSPECTION_PATH = PROJECT_ROOT / "outputs" / "raos_inspection.json"


@pytest.fixture
def inventory() -> dict:
    if not INSPECTION_PATH.exists():
        pytest.skip("outputs/raos_inspection.json not found — run inspect_raos.py first")
    with INSPECTION_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)["inventory"]["labels"]  # keys are stringified ints


def test_every_config_organ_id_is_in_the_inventory(inventory):
    config = load_config(CONFIG_PATH)
    for name, label_id in config["organs"].items():
        assert str(label_id) in inventory, (
            f"organ '{name}' -> id {label_id} is not present in the RAOS inventory"
        )


def test_target_is_defined_and_present_in_all_patients(inventory):
    config = load_config(CONFIG_PATH)
    target = config["target"]
    assert target in config["organs"], f"target '{target}' is not in organs map"

    info = inventory[str(config["organs"][target])]
    assert info["present_in_patients"] == info["total_patients"], (
        f"target '{target}' is missing in some patients "
        f"({info['present_in_patients']}/{info['total_patients']})"
    )
