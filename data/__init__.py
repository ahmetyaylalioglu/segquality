"""segquality data package: config loading, mask slice dataset, transforms,
degradations and patient-based splits.

The config loader lives here (rather than in a separate module) to keep the
package tree exactly as specified. It implements a minimal `_base_` inheritance
mechanism — deliberately not a heavyweight registry.
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_BASE_KEY = "_base_"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` (override wins).

    Nested dicts are merged key-by-key; any non-dict value is replaced wholesale.
    Neither input is mutated.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving a single-level ``_base_`` inheritance chain.

    A config may declare ``_base_: "other.yaml"`` (relative to its own folder).
    The base is loaded first (recursively) and the current file is deep-merged
    on top. The ``_base_`` key is removed from the returned config.
    """
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    base_ref = raw.pop(_BASE_KEY, None)
    if base_ref is None:
        return raw

    base_path = (config_path.parent / base_ref).resolve()
    base_cfg = load_config(base_path)
    return _deep_merge(base_cfg, raw)
