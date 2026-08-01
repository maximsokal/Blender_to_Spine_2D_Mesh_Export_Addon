"""Blender-independent 3D-to-2D projection contracts for Normal / UV Segments.

This module owns the canonical ``(U, V, D)`` coordinate system approved in
``docs/tasks/normal_uv_segments_projection_space_and_draw_order.md``.

Slice 1 intentionally implements only the six signed global-axis frames. Active-camera
projection requires evaluated Blender camera data and is resolved by a later Blender
adapter slice; attempting to resolve it here fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Iterable, Sequence, Tuple


Vector3 = Tuple[float, float, float]


class A1ProjectionError(ValueError):
    """Raised when projection input cannot satisfy the canonical contract."""


class A1ProjectionDirection(str, Enum):
    """Stable persisted identifiers for the approved projection directions."""

    POSITIVE_X = "POSITIVE_X"
    NEGATIVE_X = "NEGATIVE_X"
    POSITIVE_Y = "POSITIVE_Y"
    NEGATIVE_Y = "NEGATIVE_Y"
    POSITIVE_Z = "POSITIVE_Z"
    NEGATIVE_Z = "NEGATIVE_Z"
    ACTIVE_CAMERA = "ACTIVE_CAMERA"

    @property
    def label(self) -> str:
        return {
            A1ProjectionDirection.POSITIVE_X: "+X",
            A1ProjectionDirection.NEGATIVE_X: "-X",
            A1ProjectionDirection.POSITIVE_Y: "+Y",
            A1ProjectionDirection.NEGATIVE_Y: "-Y",
            A1ProjectionDirection.POSITIVE_Z: "+Z",
            A1ProjectionDirection.NEGATIVE_Z: "-Z",
            A1ProjectionDirection.ACTIVE_CAMERA: "Active Camera",
        }[self]

    @property
    def axis_aligned(self) -> bool:
        return self is not A1ProjectionDirection.ACTIVE_CAMERA


def resolve_a1_projection_direction(value: object) -> A1ProjectionDirection:
    """Resolve an enum or exact persisted identifier without implicit fallback."""

    if isinstance(value, A1ProjectionDirection):
        return value
    if not isinstance(value, str):
        raise TypeError("projection direction must be A1ProjectionDirection or str")
    normalized = value.strip().upper()
    if not normalized:
        raise ValueError("projection direction cannot be empty")
    try:
        return A1ProjectionDirection(normalized)
    except ValueError as exc:
        supported = tuple(item.value for item in A1ProjectionDirection)
        raise ValueError(
            f"Unsupported projection direction {value!r}; supported={supported}"
        ) from exc


def _finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number")
    resolved = float(value)
    if not isfinite(resolved):
        raise ValueError(f"{field_name} must be finite")
    return 0.0 if resolved == 0.0 else resolved


def _vector3(value: object, field_name: str) -> Vector3:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise TypeError(f"{field_name} must be a three-component tuple or list")
    return tuple(
        _finite_float(component, f"{field_name}[{index}]")
        for index, component in enumerate(value)
    )  # type: ignore[return-value]


def _dot(first: Vector3, second: Vector3) -> float:
    return sum(a * b for a, b in zip(first, second, strict=True))


def _cross(first: Vector3, second: Vector3) -> Vector3:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _length_squared(vector: Vector3) -> float:
    return _dot(vector, vector)


@dataclass(frozen=True, slots=True)
class A1ProjectedPoint:
    """One point in canonical Spine projection space.

    ``u`` maps to Spine X, ``v`` maps to Spine Y, and ``depth`` increases toward the
    observer for every signed-axis frame.
    """

    u: float
    v: float
    depth: float

    def __post_init__(self) -> None:
        for field_name in ("u", "v", "depth"):
            object.__setattr__(
                self,
                field_name,
                _finite_float(getattr(self, field_name), field_name),
            )

    @property
    def canonical_position(self) -> Vector3:
        return (self.u, self.v, self.depth)

    def relative_to(self, origin: "A1ProjectedPoint") -> "A1ProjectedPoint":
        if not isinstance(origin, A1ProjectedPoint):
            raise TypeError("origin must be A1ProjectedPoint")
        return A1ProjectedPoint(
            u=self.u - origin.u,
            v=self.v - origin.v,
            depth=self.depth - origin.depth,
        )


@dataclass(frozen=True, slots=True)
class A1AxisProjectionBasis:
    """Immutable orthonormal signed-axis basis for canonical ``U/V/D``."""

    direction: A1ProjectionDirection
    u_axis: Vector3
    v_axis: Vector3
    depth_axis: Vector3

    def __post_init__(self) -> None:
        resolved_direction = resolve_a1_projection_direction(self.direction)
        if not resolved_direction.axis_aligned:
            raise ValueError("A1AxisProjectionBasis cannot represent ACTIVE_CAMERA")
        object.__setattr__(self, "direction", resolved_direction)

        for field_name in ("u_axis", "v_axis", "depth_axis"):
            object.__setattr__(
                self,
                field_name,
                _vector3(getattr(self, field_name), field_name),
            )

        axes = (self.u_axis, self.v_axis, self.depth_axis)
        if any(_length_squared(axis) != 1.0 for axis in axes):
            raise ValueError("projection basis axes must be unit vectors")
        if any(
            _dot(axes[first], axes[second]) != 0.0
            for first in range(3)
            for second in range(first + 1, 3)
        ):
            raise ValueError("projection basis axes must be mutually orthogonal")
        if _cross(self.u_axis, self.v_axis) != self.depth_axis:
            raise ValueError(
                "projection basis must be right-handed: cross(U, V) must equal D"
            )

    def project_point(self, world_point: object) -> A1ProjectedPoint:
        point = _vector3(world_point, "world_point")
        return A1ProjectedPoint(
            u=_dot(point, self.u_axis),
            v=_dot(point, self.v_axis),
            depth=_dot(point, self.depth_axis),
        )

    def project_vector(self, world_vector: object) -> Vector3:
        vector = _vector3(world_vector, "world_vector")
        projected = self.project_point(vector)
        return projected.canonical_position


_AXIS_BASES: dict[A1ProjectionDirection, A1AxisProjectionBasis] = {
    A1ProjectionDirection.POSITIVE_X: A1AxisProjectionBasis(
        direction=A1ProjectionDirection.POSITIVE_X,
        u_axis=(0.0, 1.0, 0.0),
        v_axis=(0.0, 0.0, 1.0),
        depth_axis=(1.0, 0.0, 0.0),
    ),
    A1ProjectionDirection.NEGATIVE_X: A1AxisProjectionBasis(
        direction=A1ProjectionDirection.NEGATIVE_X,
        u_axis=(0.0, -1.0, 0.0),
        v_axis=(0.0, 0.0, 1.0),
        depth_axis=(-1.0, 0.0, 0.0),
    ),
    A1ProjectionDirection.POSITIVE_Y: A1AxisProjectionBasis(
        direction=A1ProjectionDirection.POSITIVE_Y,
        u_axis=(-1.0, 0.0, 0.0),
        v_axis=(0.0, 0.0, 1.0),
        depth_axis=(0.0, 1.0, 0.0),
    ),
    A1ProjectionDirection.NEGATIVE_Y: A1AxisProjectionBasis(
        direction=A1ProjectionDirection.NEGATIVE_Y,
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(0.0, 0.0, 1.0),
        depth_axis=(0.0, -1.0, 0.0),
    ),
    A1ProjectionDirection.POSITIVE_Z: A1AxisProjectionBasis(
        direction=A1ProjectionDirection.POSITIVE_Z,
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(0.0, 1.0, 0.0),
        depth_axis=(0.0, 0.0, 1.0),
    ),
    A1ProjectionDirection.NEGATIVE_Z: A1AxisProjectionBasis(
        direction=A1ProjectionDirection.NEGATIVE_Z,
        u_axis=(-1.0, 0.0, 0.0),
        v_axis=(0.0, 1.0, 0.0),
        depth_axis=(0.0, 0.0, -1.0),
    ),
}


def resolve_a1_axis_projection_basis(value: object) -> A1AxisProjectionBasis:
    """Return the immutable basis for one signed global axis."""

    direction = resolve_a1_projection_direction(value)
    if direction is A1ProjectionDirection.ACTIVE_CAMERA:
        raise A1ProjectionError(
            "ACTIVE_CAMERA requires an evaluated Blender camera projection frame"
        )
    return _AXIS_BASES[direction]


@dataclass(frozen=True, slots=True)
class A1AxisDepthRange:
    """Nearest and farthest front coordinates for one axis-projected object."""

    nearest: float
    farthest: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "nearest", _finite_float(self.nearest, "nearest"))
        object.__setattr__(self, "farthest", _finite_float(self.farthest, "farthest"))
        if self.nearest < self.farthest:
            raise ValueError("nearest must be greater than or equal to farthest")


def calculate_a1_axis_depth_range(
    projected_points: Iterable[A1ProjectedPoint],
) -> A1AxisDepthRange:
    """Calculate the approved nearest=max(D), farthest=min(D) axis metrics."""

    if isinstance(projected_points, (str, bytes)):
        raise TypeError("projected_points must be an iterable of A1ProjectedPoint")
    try:
        points = tuple(projected_points)
    except TypeError as exc:
        raise TypeError(
            "projected_points must be an iterable of A1ProjectedPoint"
        ) from exc
    if not points:
        raise ValueError("projected_points cannot be empty")
    if not all(isinstance(point, A1ProjectedPoint) for point in points):
        raise TypeError("projected_points must contain A1ProjectedPoint values")
    depths = tuple(point.depth for point in points)
    return A1AxisDepthRange(nearest=max(depths), farthest=min(depths))


def project_a1_axis_points(
    world_points: Iterable[object],
    direction: object,
) -> Tuple[A1ProjectedPoint, ...]:
    """Project a non-empty sequence of world points through one signed-axis frame."""

    if isinstance(world_points, (str, bytes)):
        raise TypeError("world_points must be an iterable of three-component points")
    try:
        points = tuple(world_points)
    except TypeError as exc:
        raise TypeError(
            "world_points must be an iterable of three-component points"
        ) from exc
    if not points:
        raise ValueError("world_points cannot be empty")
    basis = resolve_a1_axis_projection_basis(direction)
    return tuple(basis.project_point(point) for point in points)


__all__ = [
    "A1AxisDepthRange",
    "A1AxisProjectionBasis",
    "A1ProjectedPoint",
    "A1ProjectionDirection",
    "A1ProjectionError",
    "calculate_a1_axis_depth_range",
    "project_a1_axis_points",
    "resolve_a1_axis_projection_basis",
    "resolve_a1_projection_direction",
]
