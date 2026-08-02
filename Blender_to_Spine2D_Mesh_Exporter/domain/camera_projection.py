"""Pure camera-space projection contracts for Normal / UV Segments.

The Blender adapter resolves one evaluated active-camera view and projection matrix.
This module then performs deterministic world-to-camera projection without importing
``bpy`` or ``mathutils``. Screen coordinates are expressed in pixels relative to the
centre of the configured export texture. Canonical depth is Blender camera-local Z:
objects in front of the camera have negative values and nearer points have larger values.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite, sqrt
from typing import Tuple

from .projection import A1ProjectedPoint, A1ProjectionError


Vector3 = Tuple[float, float, float]
Vector4 = Tuple[float, float, float, float]
Matrix4x4 = Tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]


class A1CameraProjectionKind(str, Enum):
    """Supported active-camera projection models."""

    PERSPECTIVE = "PERSPECTIVE"
    ORTHOGRAPHIC = "ORTHOGRAPHIC"


def _finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number")
    resolved = float(value)
    if not isfinite(resolved):
        raise ValueError(f"{field_name} must be finite")
    return 0.0 if resolved == 0.0 else resolved


def _positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _vector3(value: object, field_name: str) -> Vector3:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise TypeError(f"{field_name} must contain three values")
    return tuple(
        _finite_float(component, f"{field_name}[{index}]")
        for index, component in enumerate(value)
    )  # type: ignore[return-value]


def _matrix4x4(value: object, field_name: str) -> Matrix4x4:
    if not isinstance(value, tuple) or len(value) != 16:
        raise TypeError(f"{field_name} must be a 16-value tuple")
    return tuple(
        _finite_float(component, f"{field_name}[{index}]")
        for index, component in enumerate(value)
    )  # type: ignore[return-value]


def _multiply_matrix_vector(matrix: Matrix4x4, vector: Vector4) -> Vector4:
    return tuple(
        sum(matrix[row * 4 + column] * vector[column] for column in range(4))
        for row in range(4)
    )  # type: ignore[return-value]


def _dot(first: Vector3, second: Vector3) -> float:
    return sum(a * b for a, b in zip(first, second, strict=True))


def _cross(first: Vector3, second: Vector3) -> Vector3:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _normalized(vector: Vector3, field_name: str) -> Vector3:
    length_squared = _dot(vector, vector)
    if not isfinite(length_squared) or length_squared <= 1.0e-30:
        raise A1ProjectionError(f"{field_name} collapses in camera space")
    inverse_length = 1.0 / sqrt(length_squared)
    result = tuple(component * inverse_length for component in vector)
    if not all(isfinite(component) for component in result):
        raise A1ProjectionError(f"{field_name} became non-finite in camera space")
    return (float(result[0]), float(result[1]), float(result[2]))


_VIEW_MATRIX_AFFINE_TOLERANCE = 1.0e-8

# Blender Matrix and Quaternion components use float32 storage. A valid rotated
# camera therefore commonly returns row-length, orthogonality, and handedness
# residuals in the 1e-8 to 1e-7 range after decomposition and inversion.
# The rotation tolerance remains far below meaningful scale or shear while
# accepting Blender's real numeric representation.
_VIEW_MATRIX_ROTATION_TOLERANCE = 1.0e-6


def _validate_view_matrix(matrix: Matrix4x4) -> None:
    if (
        abs(matrix[12]) > _VIEW_MATRIX_AFFINE_TOLERANCE
        or abs(matrix[13]) > _VIEW_MATRIX_AFFINE_TOLERANCE
        or abs(matrix[14]) > _VIEW_MATRIX_AFFINE_TOLERANCE
        or abs(matrix[15] - 1.0) > _VIEW_MATRIX_AFFINE_TOLERANCE
    ):
        raise ValueError("view_matrix must be affine with final row (0, 0, 0, 1)")

    rows = (
        (matrix[0], matrix[1], matrix[2]),
        (matrix[4], matrix[5], matrix[6]),
        (matrix[8], matrix[9], matrix[10]),
    )
    for index, row in enumerate(rows):
        if abs(_dot(row, row) - 1.0) > _VIEW_MATRIX_ROTATION_TOLERANCE:
            raise ValueError(f"view_matrix rotation row {index} must be unit length")
    if any(
        abs(_dot(rows[first], rows[second])) > _VIEW_MATRIX_ROTATION_TOLERANCE
        for first in range(3)
        for second in range(first + 1, 3)
    ):
        raise ValueError("view_matrix rotation rows must be orthogonal")
    if any(
        abs(actual - expected) > _VIEW_MATRIX_ROTATION_TOLERANCE
        for actual, expected in zip(_cross(rows[0], rows[1]), rows[2], strict=True)
    ):
        raise ValueError("view_matrix rotation must be right-handed")


@dataclass(frozen=True, slots=True)
class A1CameraProjectionFrame:
    """One evaluated active-camera projection for a fixed export texture canvas."""

    camera_id: str
    kind: A1CameraProjectionKind
    texture_width: int
    texture_height: int
    clip_start: float
    clip_end: float
    view_matrix: Matrix4x4
    projection_matrix: Matrix4x4

    def __post_init__(self) -> None:
        if not isinstance(self.camera_id, str) or not self.camera_id.strip():
            raise ValueError("camera_id must be a non-empty string")
        if not isinstance(self.kind, A1CameraProjectionKind):
            raise TypeError("kind must be A1CameraProjectionKind")
        object.__setattr__(
            self,
            "texture_width",
            _positive_integer(self.texture_width, "texture_width"),
        )
        object.__setattr__(
            self,
            "texture_height",
            _positive_integer(self.texture_height, "texture_height"),
        )
        clip_start = _finite_float(self.clip_start, "clip_start")
        clip_end = _finite_float(self.clip_end, "clip_end")
        if clip_start <= 0.0:
            raise ValueError("clip_start must be positive")
        if clip_end <= clip_start:
            raise ValueError("clip_end must be greater than clip_start")
        object.__setattr__(self, "clip_start", clip_start)
        object.__setattr__(self, "clip_end", clip_end)

        view_matrix = _matrix4x4(self.view_matrix, "view_matrix")
        projection_matrix = _matrix4x4(
            self.projection_matrix,
            "projection_matrix",
        )
        _validate_view_matrix(view_matrix)
        object.__setattr__(self, "view_matrix", view_matrix)
        object.__setattr__(self, "projection_matrix", projection_matrix)

    @property
    def aspect_ratio(self) -> float:
        return float(self.texture_width) / float(self.texture_height)

    def world_to_camera_point(
        self,
        world_point: object,
        *,
        field_name: str = "world_point",
    ) -> Vector3:
        point = _vector3(world_point, field_name)
        camera = _multiply_matrix_vector(
            self.view_matrix,
            (point[0], point[1], point[2], 1.0),
        )
        if abs(camera[3] - 1.0) > 1.0e-8:
            raise A1ProjectionError(
                f"{field_name} produced invalid affine camera W={camera[3]}"
            )
        return (camera[0], camera[1], camera[2])

    def project_world_point(
        self,
        world_point: object,
        *,
        field_name: str = "world_point",
    ) -> A1ProjectedPoint:
        """Project one world point to centred texture pixels and camera-local depth."""

        camera_x, camera_y, camera_z = self.world_to_camera_point(
            world_point,
            field_name=field_name,
        )
        forward_depth = -camera_z
        near_tolerance = max(1.0e-9, self.clip_start * 1.0e-9)
        if forward_depth <= self.clip_start + near_tolerance:
            raise A1ProjectionError(
                f"{field_name} lies on or behind the active camera near plane; "
                f"forward_depth={forward_depth}, clip_start={self.clip_start}"
            )

        clip = _multiply_matrix_vector(
            self.projection_matrix,
            (camera_x, camera_y, camera_z, 1.0),
        )
        if abs(clip[3]) <= 1.0e-12:
            raise A1ProjectionError(
                f"{field_name} produced zero homogeneous projection W"
            )
        ndc_x = clip[0] / clip[3]
        ndc_y = clip[1] / clip[3]
        if not isfinite(ndc_x) or not isfinite(ndc_y):
            raise A1ProjectionError(
                f"{field_name} produced non-finite camera projection coordinates"
            )

        pixel_x = (ndc_x * 0.5 + 0.5) * float(self.texture_width)
        pixel_y = (ndc_y * 0.5 + 0.5) * float(self.texture_height)
        return A1ProjectedPoint(
            u=pixel_x - float(self.texture_width) / 2.0,
            v=pixel_y - float(self.texture_height) / 2.0,
            depth=camera_z,
        )

    def transform_world_direction(
        self,
        world_direction: object,
        *,
        field_name: str = "world_direction",
    ) -> Vector3:
        """Rotate one already world-space direction into camera-local XYZ."""

        direction = _vector3(world_direction, field_name)
        transformed = (
            self.view_matrix[0] * direction[0]
            + self.view_matrix[1] * direction[1]
            + self.view_matrix[2] * direction[2],
            self.view_matrix[4] * direction[0]
            + self.view_matrix[5] * direction[1]
            + self.view_matrix[6] * direction[2],
            self.view_matrix[8] * direction[0]
            + self.view_matrix[9] * direction[1]
            + self.view_matrix[10] * direction[2],
        )
        return _normalized(transformed, field_name)


__all__ = [
    "A1CameraProjectionFrame",
    "A1CameraProjectionKind",
]
