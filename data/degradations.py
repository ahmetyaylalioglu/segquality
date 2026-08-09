"""Deliberate mask degradation for the evaluation side of the project.

Architecture note
-----------------
Everything is organised around a single interface:

    mask -> (degraded_mask, true_dice)

expressed by the :class:`DegradationSource` protocol. The synthetic corruptor
(:class:`SyntheticDegrader`) is *one implementation* of that interface, not the
only path. Synthetic morphological edits do not perfectly mimic the errors a
real nnU-Net makes, so a future :class:`ModelPredictionDegrader` can implement
the same ``make_pair`` contract and feed real model outputs through the exact
same downstream code (Dice measurement, frozen eval sets, stats, plots).

Design decisions baked in here
------------------------------
* ``rng`` is always passed explicitly (never global ``np.random``) so an
  evaluation set can be frozen and reproduced bit-for-bit.
* Dice targets are sampled first, then severity is solved by binary search
  (inverse sampling), spreading true Dice across ~[0.4, 0.99] instead of piling
  up near 1.0.
* Every produced pair is guarded against degenerate output (empty / full /
  Dice below ``min_valid_dice``): severity is backed off, then a gentle
  fallback is used, so the loader never emits an empty mask (which would surface
  as NaN in training).
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    gaussian_filter,
)
from scipy.ndimage import shift as ndi_shift

logger = logging.getLogger(__name__)

# --- Synthetic degradation strength defaults ------------------------------
# These map severity in [0, 1] to a physical amount of corruption. They are the
# *defaults* of the synthetic generator (overridable via DegradationParams); the
# binary search over severity compensates for their exact values, so they only
# set the reachable range, not the final Dice.
MAX_MORPH_ITERS = 12        # erode/dilate iterations at severity 1.0
MAX_TRANSLATE_FRAC = 0.30   # translate shift as a fraction of the frame side
MAX_BOUNDARY_FRAC = 0.12    # boundary displacement amplitude, fraction of side
MAX_BLOB_FRAC = 0.50        # remove_blob: max fraction of organ area removed
MAX_FP_FRAC = 0.40          # add_false_positive: blob area as fraction of organ area
_FP_GAP_ITERS = 2           # dilation gap that keeps a false positive disconnected

# --- Chaining (compose) defaults ------------------------------------------
# When a single primitive hits its geometric Dice floor above the target, a
# second (and possibly third) degradation is composed on top to push Dice down.
# Only shape-coherent strong types are chained, so a chained low-Dice mask still
# looks like a *broken liver* rather than scattered noise (add_false_positive /
# boundary_noise are excluded from chaining for that reason).
_CHAIN_TYPES = ("erode", "dilate", "remove_blob", "translate")
_MIN_CHAIN_GAIN = 0.01      # if a chain step drops Dice by less than this, that
#                             type is exhausted on this mask -> stop (honest fail)

# Types that fragment the mask into scattered noise when pushed hard, so they are
# only allowed as a *light* first-step degradation in the upper part of their
# measured Dice range. boundary_noise below ~0.85 shatters the organ into pixel
# speckle (not anatomically plausible); remove_blob is NOT here because it removes
# one contiguous chunk (a realistic, controlled fragmentation).
_LIGHT_ONLY_TYPES = ("boundary_noise",)
# Fraction of a type's measured [dice_min, dice_max] span that counts as its
# "safe" upper band. 0.7 turns boundary_noise's [0.504, 0.997] into a ~0.85 floor
# — derived from the calibration table, not hardcoded.
_LIGHT_ONLY_FRAC = 0.7


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _to_bool(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask) > 0


def _as_input(reference: np.ndarray, out_bool: np.ndarray) -> np.ndarray:
    """Cast a boolean result back to the reference array's dtype."""
    return out_bool.astype(reference.dtype)


def _is_empty(mask: np.ndarray) -> bool:
    return not bool(mask.any())


def _is_full(mask: np.ndarray) -> bool:
    return bool(mask.all())


def _disk(shape: tuple[int, int], cy: float, cx: float, radius: float) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice overlap of two masks. Two empty masks score 1.0; one empty, 0.0."""
    a_bool = _to_bool(a)
    b_bool = _to_bool(b)
    total = int(a_bool.sum()) + int(b_bool.sum())
    if total == 0:
        return 1.0
    inter = int(np.logical_and(a_bool, b_bool).sum())
    return 2.0 * inter / total


# --------------------------------------------------------------------------
# Primitive degradations: (mask, severity, rng) -> mask, values in {0, 1}
# --------------------------------------------------------------------------
def erode(mask: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    """Morphological erosion; severity scales the number of iterations."""
    iters = max(1, int(round(severity * MAX_MORPH_ITERS)))
    out = binary_erosion(_to_bool(mask), iterations=iters)
    return _as_input(mask, out)


def dilate(mask: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    """Morphological dilation; severity scales the number of iterations."""
    iters = max(1, int(round(severity * MAX_MORPH_ITERS)))
    out = binary_dilation(_to_bool(mask), iterations=iters)
    return _as_input(mask, out)


def remove_blob(mask: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    """Erase a random-located disk from the mask (can break topology)."""
    binary = _to_bool(mask)
    fg = np.argwhere(binary)
    if fg.size == 0:
        return _as_input(mask, binary)
    area = len(fg)
    radius = max(1.0, np.sqrt((severity * MAX_BLOB_FRAC * area) / np.pi))
    cy, cx = fg[rng.integers(len(fg))]
    out = binary & ~_disk(binary.shape, cy, cx, radius)
    return _as_input(mask, out)


def add_false_positive(
    mask: np.ndarray, severity: float, rng: np.random.Generator
) -> np.ndarray:
    """Add a disconnected spurious blob outside the mask."""
    binary = _to_bool(mask)
    area = max(int(binary.sum()), 1)
    radius = max(1.0, np.sqrt((severity * MAX_FP_FRAC * area) / np.pi))
    # Keep a gap so the blob stays disconnected from the true mask.
    forbidden = binary_dilation(binary, iterations=_FP_GAP_ITERS)
    background = np.argwhere(~forbidden)
    if background.size == 0:
        return _as_input(mask, binary)
    cy, cx = background[rng.integers(len(background))]
    out = binary | _disk(binary.shape, cy, cx, radius)
    return _as_input(mask, out)


def boundary_noise(
    mask: np.ndarray, severity: float, rng: np.random.Generator
) -> np.ndarray:
    """Roughen the boundary by displacing the signed distance field with noise."""
    binary = _to_bool(mask)
    if _is_empty(binary):
        return _as_input(mask, binary)
    signed = distance_transform_edt(binary) - distance_transform_edt(~binary)
    side = min(binary.shape)
    amplitude = severity * MAX_BOUNDARY_FRAC * side
    sigma = max(1.0, 0.03 * side)
    noise = gaussian_filter(rng.standard_normal(binary.shape), sigma)
    std = noise.std() or 1.0
    out = (signed + noise / std * amplitude) > 0
    return _as_input(mask, out)


def translate(mask: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    """Shift the mask by a severity-scaled amount in a random direction."""
    binary = _to_bool(mask).astype(np.uint8)
    side = min(binary.shape)
    magnitude = severity * MAX_TRANSLATE_FRAC * side
    angle = rng.uniform(0.0, 2.0 * np.pi)
    shifted = ndi_shift(
        binary,
        (magnitude * np.sin(angle), magnitude * np.cos(angle)),
        order=0,
        mode="constant",
        cval=0,
    )
    return _as_input(mask, shifted > 0)


PRIMITIVES: dict[str, Callable[[np.ndarray, float, np.random.Generator], np.ndarray]] = {
    "erode": erode,
    "dilate": dilate,
    "remove_blob": remove_blob,
    "add_false_positive": add_false_positive,
    "boundary_noise": boundary_noise,
    "translate": translate,
}

# Types strong enough to reach low Dice targets; used when target < STRONG_ONLY.
_STRONG_TYPES = ("erode", "dilate", "remove_blob", "translate")
_STRONG_ONLY_BELOW = 0.7


# --------------------------------------------------------------------------
# The interface + a frozen, replayable record
# --------------------------------------------------------------------------
@runtime_checkable
class DegradationSource(Protocol):
    """Anything that turns a mask into a (degraded_mask, true_dice) pair.

    ``SyntheticDegrader`` implements this now; a real-model source can later.
    """

    def make_pair(
        self, mask: np.ndarray, rng: np.random.Generator, target_dice: float | None = None
    ) -> tuple[np.ndarray, float]:
        ...


@dataclass
class DegradationRecord:
    """A frozen, replayable degradation (single primitive OR a chain).

    ``steps`` is the canonical replay recipe: an ordered list of
    ``{"type", "severity", "seed"}`` applied in sequence via
    :meth:`SyntheticDegrader.apply_record`. For a single degradation ``steps``
    has length 1 and the flat ``degradation_type`` / ``severity`` / ``seed``
    fields mirror that one step, so older single-step records (which predate
    ``steps`` and carry only the flat fields) still replay bit-for-bit.

    ``target_dice`` is the requested Dice (may be None for a random draw) and
    ``target_reached`` records whether the achieved Dice landed within tolerance
    of it — an honest ``False`` when a mask's geometry could not reach the target.
    """

    patient_id: str
    slice_index: int
    degradation_type: str
    severity: float
    seed: int
    true_dice: float
    target_dice: float | None = None
    target_reached: bool = True
    steps: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DegradationRecord":
        return cls(**data)


@dataclass
class DegradationParams:
    """Tunables for the synthetic degrader (mirrors the ``degradation`` config)."""

    target_dice_range: tuple[float, float] = (0.4, 0.99)
    dice_tolerance: float = 0.03
    search_iters: int = 8
    min_valid_dice: float = 0.35
    max_retries: int = 5
    max_chain_steps: int = 3


@dataclass
class _GenResult:
    """Internal return of :meth:`SyntheticDegrader._generate`."""

    name: str                      # single type, or "+"-joined chain
    severity: float                # first step's severity (summary)
    seed: int                      # first step's seed (summary)
    steps: list[dict[str, Any]]    # ordered replay recipe
    degraded: np.ndarray
    achieved: float
    target: float
    target_reached: bool


class SyntheticDegrader:
    """Synthetic implementation of :class:`DegradationSource`.

    Uses inverse Dice sampling (draw target Dice, binary-search severity) and
    guards every output against degenerate masks.
    """

    def __init__(self, params: DegradationParams | None = None) -> None:
        self.params = params or DegradationParams()
        # Per-type reachable Dice ranges, measured on real masks via calibrate().
        # None until calibrated; type selection then falls back to a heuristic.
        self.type_ranges: dict[str, tuple[float, float]] | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SyntheticDegrader":
        section = config.get("degradation", {})
        rng_range = section.get("target_dice_range", (0.4, 0.99))
        return cls(
            DegradationParams(
                target_dice_range=(float(rng_range[0]), float(rng_range[1])),
                dice_tolerance=float(section.get("dice_tolerance", 0.03)),
                search_iters=int(section.get("search_iters", 8)),
                min_valid_dice=float(section.get("min_valid_dice", 0.35)),
                max_retries=int(section.get("max_retries", 5)),
                max_chain_steps=int(section.get("max_chain_steps", 3)),
            )
        )

    # -- calibration --------------------------------------------------------
    def calibrate(
        self, masks: list[np.ndarray], n_severities: int = 12
    ) -> dict[str, tuple[float, float]]:
        """Measure each primitive's reachable ``[dice_min, dice_max]`` on ``masks``.

        Ranges are geometry-dependent, so they must be measured on masks that
        match the evaluation data (real liver slices), not synthetic shapes. A
        fixed seed makes the stochastic primitives (remove_blob/translate)
        reproducible so the reported ranges are stable. The result is stored on
        the instance and used to prefer target-reachable types in ``_choose_type``.
        """
        grid = np.linspace(0.05, 1.0, n_severities)
        ranges: dict[str, tuple[float, float]] = {}
        for name in PRIMITIVES:
            lo, hi = 1.0, 0.0
            for m in masks:
                base = _to_bool(m)
                for s in grid:
                    d = dice(base, self.apply(name, base, float(s), np.random.default_rng(0)))
                    lo, hi = min(lo, d), max(hi, d)
            ranges[name] = (float(lo), float(hi))
        self.type_ranges = ranges
        return ranges

    # -- primitive dispatch -------------------------------------------------
    def apply(
        self, name: str, mask: np.ndarray, severity: float, rng: np.random.Generator
    ) -> np.ndarray:
        return PRIMITIVES[name](mask, severity, rng)

    # -- public interface ---------------------------------------------------
    def make_pair(
        self, mask: np.ndarray, rng: np.random.Generator, target_dice: float | None = None
    ) -> tuple[np.ndarray, float]:
        """mask -> (degraded_mask, true_dice). Returns the *achieved* Dice."""
        result = self._generate(mask, rng, target_dice)
        return _as_input(mask, result.degraded), float(result.achieved)

    def apply_random_degradation(
        self, mask: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Apply one random primitive at random severity (guarded)."""
        binary = _to_bool(mask)
        name = list(PRIMITIVES)[int(rng.integers(len(PRIMITIVES)))]
        severity = float(rng.uniform(0.0, 1.0))
        seed = int(rng.integers(0, 2**31 - 1))
        degraded = self.apply(name, binary, severity, np.random.default_rng(seed))
        _, _, degraded, _ = self._guard(
            name, binary, severity, seed, degraded, dice(binary, degraded)
        )
        return _as_input(mask, degraded)

    def sample_record(
        self,
        mask: np.ndarray,
        rng: np.random.Generator,
        patient_id: str,
        slice_index: int,
        target_dice: float | None = None,
    ) -> DegradationRecord:
        """Produce a frozen, replayable record for an evaluation set."""
        result = self._generate(mask, rng, target_dice)
        return DegradationRecord(
            patient_id=patient_id,
            slice_index=slice_index,
            degradation_type=result.name,
            severity=float(result.severity),
            seed=int(result.seed),
            true_dice=float(result.achieved),
            target_dice=float(result.target),
            target_reached=bool(result.target_reached),
            steps=result.steps,
        )

    def apply_record(
        self, mask: np.ndarray, record: DegradationRecord
    ) -> np.ndarray:
        """Reproduce a recorded degradation exactly (deterministic replay).

        Chains replay their ``steps`` in order; single-step legacy records (no
        ``steps``) fall back to the flat ``degradation_type`` / ``severity`` /
        ``seed`` triple, so old evaluation sets still reproduce bit-for-bit.
        """
        current = _to_bool(mask)
        steps = record.steps or [
            {
                "type": record.degradation_type,
                "severity": record.severity,
                "seed": record.seed,
            }
        ]
        for step in steps:
            current = _to_bool(
                self.apply(
                    step["type"],
                    current,
                    float(step["severity"]),
                    np.random.default_rng(int(step["seed"])),
                )
            )
        return _as_input(mask, current)

    # -- internals ----------------------------------------------------------
    def _choose_type(self, rng: np.random.Generator, target_dice: float) -> str:
        """Pick a primitive that can plausibly reach ``target_dice``.

        With measured ``type_ranges`` (from :meth:`calibrate`) prefer types whose
        reachable band covers the target; if the target sits below every type's
        floor, start from the strongest type (lowest floor) and let chaining push
        the rest of the way. Without calibration, fall back to the old heuristic.
        """
        if self.type_ranges:
            tol = self.params.dice_tolerance
            reachable: list[str] = []
            for name, (lo, hi) in self.type_ranges.items():
                if not (lo - tol <= target_dice <= hi + tol):
                    continue
                # Keep fragmenting types (boundary_noise) light: only eligible in
                # the upper part of their measured range, else they shatter the mask.
                if name in _LIGHT_ONLY_TYPES:
                    safe_floor = lo + _LIGHT_ONLY_FRAC * (hi - lo)
                    if target_dice < safe_floor:
                        continue
                reachable.append(name)
            if reachable:
                return reachable[int(rng.integers(len(reachable)))]
            # Target below every band's floor: pick the strongest coherent type
            # (lowest floor) so chaining can push down — never a light-only type.
            coherent = {
                n: r for n, r in self.type_ranges.items() if n not in _LIGHT_ONLY_TYPES
            }
            return min(coherent, key=lambda n: coherent[n][0])
        pool = _STRONG_TYPES if target_dice < _STRONG_ONLY_BELOW else tuple(PRIMITIVES)
        return pool[int(rng.integers(len(pool)))]

    def _search_severity(
        self,
        name: str,
        base: np.ndarray,
        reference: np.ndarray,
        seed: int,
        target: float,
    ) -> tuple[float, np.ndarray, float]:
        """Binary-search severity of ``name`` applied to ``base`` so that
        ``dice(reference, result)`` lands near ``target``.

        Dice decreases roughly monotonically with severity for a fixed seed, so a
        short bisection over [0, 1] suffices. Returns (severity, result, achieved).
        """
        low_s, high_s = 0.0, 1.0
        best: tuple[float, np.ndarray, float] | None = None
        for _ in range(self.params.search_iters):
            mid = 0.5 * (low_s + high_s)
            cand = self.apply(name, base, mid, np.random.default_rng(seed))
            achieved = dice(reference, cand)
            if best is None or abs(achieved - target) < abs(best[2] - target):
                best = (mid, cand, achieved)
            if achieved > target:
                low_s = mid   # too little corruption -> increase severity
            else:
                high_s = mid
        return best  # type: ignore[return-value]

    def _generate(
        self, mask: np.ndarray, rng: np.random.Generator, target_dice: float | None
    ) -> _GenResult:
        """Choose a type, solve severity, then chain more degradations if the
        single-type floor is still above the target.

        The chain stops (honest failure, ``target_reached=False``) when it reaches
        ``max_chain_steps``, when the mask would go degenerate, or when a step no
        longer meaningfully lowers Dice (``_MIN_CHAIN_GAIN``) — never an unbounded
        loop. Every step's (type, severity, seed) is recorded so the composed
        result replays bit-for-bit.
        """
        binary = _to_bool(mask)
        if _is_empty(binary):
            raise ValueError("cannot degrade an empty mask")
        if target_dice is None:
            low, high = self.params.target_dice_range
            target_dice = float(rng.uniform(low, high))
        tol = self.params.dice_tolerance

        # -- first step: single primitive + degenerate guard ----------------
        name = self._choose_type(rng, target_dice)
        seed = int(rng.integers(0, 2**31 - 1))
        severity, degraded, achieved = self._search_severity(
            name, binary, binary, seed, target_dice
        )
        name, severity, degraded, achieved = self._guard(
            name, binary, severity, seed, degraded, achieved
        )
        steps = [{"type": name, "severity": float(severity), "seed": int(seed)}]

        # -- chain: compose stronger degradations to reach a low target -----
        while achieved - target_dice > tol and len(steps) < self.params.max_chain_steps:
            c_name = _CHAIN_TYPES[int(rng.integers(len(_CHAIN_TYPES)))]
            c_seed = int(rng.integers(0, 2**31 - 1))
            c_sev, cand, c_achieved = self._search_severity(
                c_name, degraded, binary, c_seed, target_dice
            )
            # Stop on degenerate output or when the step barely moves Dice: that
            # type is exhausted on this mask -> return the honest achieved Dice.
            if _is_empty(cand) or _is_full(cand) or c_achieved < self.params.min_valid_dice:
                break
            if achieved - c_achieved < _MIN_CHAIN_GAIN:
                break
            degraded, achieved = cand, c_achieved
            steps.append({"type": c_name, "severity": float(c_sev), "seed": int(c_seed)})

        joined = "+".join(s["type"] for s in steps)
        return _GenResult(
            name=joined,
            severity=steps[0]["severity"],
            seed=steps[0]["seed"],
            steps=steps,
            degraded=degraded,
            achieved=achieved,
            target=target_dice,
            target_reached=abs(achieved - target_dice) <= tol,
        )

    def _is_degenerate(self, mask: np.ndarray, achieved: float) -> bool:
        # min_valid_dice sits just below the target floor (0.4) so results in the
        # tolerance band are not wrongly rejected, while empty/full/collapsed
        # masks (which would become NaN in training) are always caught.
        return _is_empty(mask) or _is_full(mask) or achieved < self.params.min_valid_dice

    def _guard(
        self,
        name: str,
        binary: np.ndarray,
        severity: float,
        seed: int,
        degraded: np.ndarray,
        achieved: float,
    ) -> tuple[str, float, np.ndarray, float]:
        """Protect against degenerate output; keeps the result replayable.

        Backs off severity up to ``max_retries``, then falls back to a gentle
        translate, then to an identity translate (severity 0). The returned
        (name, severity) always reproduce ``degraded`` via a fixed seed.
        """
        tries = 0
        while self._is_degenerate(degraded, achieved) and tries < self.params.max_retries:
            severity *= 0.7
            degraded = self.apply(name, binary, severity, np.random.default_rng(seed))
            achieved = dice(binary, degraded)
            tries += 1

        if self._is_degenerate(degraded, achieved):
            name, severity = "translate", 0.08
            degraded = self.apply(name, binary, severity, np.random.default_rng(seed))
            achieved = dice(binary, degraded)

        if self._is_degenerate(degraded, achieved):
            name, severity = "translate", 0.0  # identity: shift of 0 px
            degraded = self.apply(name, binary, severity, np.random.default_rng(seed))
            achieved = dice(binary, degraded)

        return name, severity, degraded, achieved


# Module-level convenience wrappers over a default synthetic degrader ------
_DEFAULT_DEGRADER = SyntheticDegrader()


def make_degraded_pair(
    mask: np.ndarray, rng: np.random.Generator, target_dice: float | None = None
) -> tuple[np.ndarray, float]:
    """mask -> (degraded_mask, true_dice) using the default synthetic degrader.

    If ``target_dice`` is None a target is drawn from the configured range; the
    returned Dice is the *achieved* value (never the requested target).
    """
    return _DEFAULT_DEGRADER.make_pair(mask, rng, target_dice)


def apply_random_degradation(
    mask: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Apply one random primitive at random severity (guarded)."""
    return _DEFAULT_DEGRADER.apply_random_degradation(mask, rng)
