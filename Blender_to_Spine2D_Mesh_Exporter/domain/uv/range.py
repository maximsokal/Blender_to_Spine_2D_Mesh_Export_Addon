"""Unit-square range inspection for loop-level UV layouts.

Spine mesh attachment UVs are normalized texture coordinates.  This module keeps
range validation Blender-independent and operates on immutable ``MeshSnapshot``
loops before UV seam variants are deduplicated into attachment vertices.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

from ..geometry import LoopId, MeshSnapshot, MeshSnapshotValidator, SourceLoopId
from ..geometry.contracts import (
    require_exact_type,
    require_finite_number,
    require_finite_vector,
    require_identity,
    require_integer,
    require_non_empty_string,
    require_tuple_items,
)


class UvRangePolicy(str, Enum):
    """Control how UV coordinates outside the normalized texture square are handled."""

    REQUIRE_UNIT_SQUARE = "REQUIRE_UNIT_SQUARE"
    WARN_ONLY = "WARN_ONLY"


@dataclass(frozen=True, slots=True)
class UvRangeViolation:
    """One loop whose UV lies beyond the configured unit-square tolerance."""

    loop_id: LoopId
    source_loop_id: SourceLoopId
    coordinate: Tuple[float, float]

    def __post_init__(self) -> None:
        require_exact_type(self.loop_id, LoopId, "loop_id")
        require_exact_type(self.source_loop_id, SourceLoopId, "source_loop_id")
        require_finite_vector(self.coordinate, 2, "coordinate")


@dataclass(frozen=True, slots=True)
class UvRangeReport:
    """Deterministic result of checking one snapshot UV layer."""

    snapshot_id: str
    layer_name: str
    epsilon: float
    loop_count: int
    violations: Tuple[UvRangeViolation, ...]

    def __post_init__(self) -> None:
        require_identity(self.snapshot_id, "snapshot_id")
        require_non_empty_string(self.layer_name, "layer_name")
        epsilon = require_finite_number(self.epsilon, "epsilon")
        if epsilon < 0.0:
            raise ValueError("epsilon cannot be negative")
        require_integer(self.loop_count, "loop_count", minimum=1)
        require_tuple_items(self.violations, UvRangeViolation, "violations")
        loop_ids = tuple(item.loop_id for item in self.violations)
        if len(loop_ids) != len(set(loop_ids)):
            raise ValueError("violations contain duplicate LoopId values")
        if len(self.violations) > self.loop_count:
            raise ValueError("violation count cannot exceed loop_count")

    @property
    def outside_loop_count(self) -> int:
        return len(self.violations)

    @property
    def inside_unit_square(self) -> bool:
        return not self.violations


class UvRangeError(ValueError):
    """Raised when strict range policy rejects one or more UV loops."""

    def __init__(self, report: UvRangeReport):
        require_exact_type(report, UvRangeReport, "report")
        self.report = report
        preview = ", ".join(
            f"loop {item.loop_id.index}={item.coordinate}"
            for item in report.violations[:8]
        )
        if len(report.violations) > 8:
            preview += f", ... +{len(report.violations) - 8} more"
        super().__init__(
            f"UV layer '{report.layer_name}' in snapshot '{report.snapshot_id}' has "
            f"{report.outside_loop_count} loop(s) outside [0, 1] with epsilon "
            f"{report.epsilon}: {preview}"
        )


def inspect_uv_range(
    snapshot: MeshSnapshot,
    layer_name: str,
    *,
    epsilon: float = 1.0e-6,
) -> UvRangeReport:
    """Inspect every loop UV against ``[-epsilon, 1 + epsilon]``.

    The function intentionally does not clamp coordinates.  Clamping would change
    the authored/generated layout and can collapse island margins or hide an
    incorrectly configured unwrap/pack stage.
    """

    require_exact_type(snapshot, MeshSnapshot, "snapshot")
    require_non_empty_string(layer_name, "layer_name")
    resolved_epsilon = require_finite_number(epsilon, "epsilon")
    if resolved_epsilon < 0.0:
        raise ValueError("epsilon cannot be negative")
    MeshSnapshotValidator().validate_or_raise(snapshot)
    if not snapshot.loops:
        raise ValueError("snapshot contains no loops")
    if layer_name not in snapshot.uv_layer_names:
        raise KeyError(f"UV layer '{layer_name}' is not present in snapshot")

    minimum = -resolved_epsilon
    maximum = 1.0 + resolved_epsilon
    violations: list[UvRangeViolation] = []
    for loop in sorted(snapshot.loops, key=lambda item: item.id.index):
        coordinate = loop.uv(layer_name)
        if coordinate is None:
            raise KeyError(
                f"Loop {loop.id.index} does not contain declared UV layer "
                f"'{layer_name}'"
            )
        u = float(coordinate[0])
        v = float(coordinate[1])
        if minimum <= u <= maximum and minimum <= v <= maximum:
            continue
        violations.append(
            UvRangeViolation(
                loop_id=loop.id,
                source_loop_id=loop.source_id,
                coordinate=(u, v),
            )
        )

    return UvRangeReport(
        snapshot_id=snapshot.snapshot_id,
        layer_name=layer_name,
        epsilon=resolved_epsilon,
        loop_count=len(snapshot.loops),
        violations=tuple(violations),
    )


def enforce_uv_range(
    snapshot: MeshSnapshot,
    layer_name: str,
    *,
    policy: UvRangePolicy = UvRangePolicy.REQUIRE_UNIT_SQUARE,
    epsilon: float = 1.0e-6,
) -> UvRangeReport:
    """Inspect one layer and apply the selected range policy."""

    require_exact_type(policy, UvRangePolicy, "policy")
    report = inspect_uv_range(snapshot, layer_name, epsilon=epsilon)
    if report.violations and policy is UvRangePolicy.REQUIRE_UNIT_SQUARE:
        raise UvRangeError(report)
    return report


__all__ = [
    "UvRangeError",
    "UvRangePolicy",
    "UvRangeReport",
    "UvRangeViolation",
    "enforce_uv_range",
    "inspect_uv_range",
]
