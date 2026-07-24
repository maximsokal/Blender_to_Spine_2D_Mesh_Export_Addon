"""Bake a Blender object's linear world transform into immutable mesh geometry.

Rewrite geometry algorithms operate on snapshot-local positions, while Blender renders
the source object through its complete ``matrix_world``. To keep segmentation, UV unwrap,
bake targets, and Spine vertex bones in one physical space, rotation/scale/shear are
applied to positions and oriented normals exactly once. The returned snapshot retains
only world translation, preserving the invariant::

    original_world_matrix @ original_local_position
    == normalized_translation_matrix @ normalized_local_position

The source Blender object is not mutated. Runtime source-object validation continues to
use its original full matrix independently from this geometry normalization.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite, sqrt
from typing import Tuple

from .model import Matrix4x4, MeshSnapshot, Vector3
from .validator import MeshSnapshotValidator


Matrix3x3 = Tuple[
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


class MeshWorldTransformError(ValueError):
    """Raised when an affine world transform cannot be baked safely."""


@dataclass(frozen=True, slots=True)
class MeshWorldTransformResult:
    snapshot: MeshSnapshot
    linear_matrix: Matrix3x3
    translation: Vector3
    determinant: float
    mirrored: bool
    changed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, MeshSnapshot):
            raise TypeError("snapshot must be MeshSnapshot")
        if not isinstance(self.linear_matrix, tuple) or len(self.linear_matrix) != 9:
            raise ValueError("linear_matrix must contain nine values")
        if not isinstance(self.translation, tuple) or len(self.translation) != 3:
            raise ValueError("translation must contain three values")
        if not all(
            isinstance(value, float) and isfinite(value)
            for value in self.linear_matrix
        ):
            raise ValueError("linear_matrix must contain finite floats")
        if not all(
            isinstance(value, float) and isfinite(value)
            for value in self.translation
        ):
            raise ValueError("translation must contain finite floats")
        if not isinstance(self.determinant, float) or not isfinite(self.determinant):
            raise ValueError("determinant must be a finite float")
        if not isinstance(self.mirrored, bool) or not isinstance(self.changed, bool):
            raise TypeError("mirrored and changed must be bool")
        if self.mirrored != (self.determinant < 0.0):
            raise ValueError("mirrored must match the determinant sign")


def _matrix_parts(matrix: Matrix4x4) -> tuple[Matrix3x3, Vector3]:
    if not isinstance(matrix, tuple) or len(matrix) != 16:
        raise TypeError("world_matrix must be a 16-value tuple")
    values = tuple(float(value) for value in matrix)
    if not all(isfinite(value) for value in values):
        raise MeshWorldTransformError("world_matrix contains non-finite values")

    affine_tolerance = 1.0e-10
    if (
        abs(values[12]) > affine_tolerance
        or abs(values[13]) > affine_tolerance
        or abs(values[14]) > affine_tolerance
        or abs(values[15] - 1.0) > affine_tolerance
    ):
        raise MeshWorldTransformError(
            "world_matrix must be affine with final row (0, 0, 0, 1)"
        )
    return (
        (
            values[0],
            values[1],
            values[2],
            values[4],
            values[5],
            values[6],
            values[8],
            values[9],
            values[10],
        ),
        (values[3], values[7], values[11]),
    )


def _determinant(matrix: Matrix3x3) -> float:
    a, b, c, d, e, f, g, h, i = matrix
    return (
        a * (e * i - f * h)
        - b * (d * i - f * g)
        + c * (d * h - e * g)
    )


def _cofactor_matrix(matrix: Matrix3x3) -> Matrix3x3:
    """Return the oriented normal transform ``det(A) * inverse(A).T``."""

    a, b, c, d, e, f, g, h, i = matrix
    return (
        e * i - f * h,
        f * g - d * i,
        d * h - e * g,
        c * h - b * i,
        a * i - c * g,
        b * g - a * h,
        b * f - c * e,
        c * d - a * f,
        a * e - b * d,
    )


def _multiply_vector(matrix: Matrix3x3, value: Vector3) -> Vector3:
    a, b, c, d, e, f, g, h, i = matrix
    x, y, z = value
    return (
        a * x + b * y + c * z,
        d * x + e * y + f * z,
        g * x + h * y + i * z,
    )


def _normalized(value: Vector3, *, field_name: str) -> Vector3:
    length_squared = sum(component * component for component in value)
    if not isfinite(length_squared) or length_squared <= 1.0e-30:
        raise MeshWorldTransformError(
            f"{field_name} collapses under the object world transform"
        )
    inverse_length = 1.0 / sqrt(length_squared)
    result = tuple(component * inverse_length for component in value)
    if not all(isfinite(component) for component in result):
        raise MeshWorldTransformError(
            f"{field_name} became non-finite after normalization"
        )
    return (float(result[0]), float(result[1]), float(result[2]))


def _translation_matrix(translation: Vector3) -> Matrix4x4:
    x, y, z = translation
    return (
        1.0,
        0.0,
        0.0,
        x,
        0.0,
        1.0,
        0.0,
        y,
        0.0,
        0.0,
        1.0,
        z,
        0.0,
        0.0,
        0.0,
        1.0,
    )


def _linear_is_identity(matrix: Matrix3x3, *, tolerance: float) -> bool:
    identity: Matrix3x3 = (
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    return all(
        abs(actual - expected) <= tolerance
        for actual, expected in zip(matrix, identity, strict=True)
    )


def normalize_mesh_snapshot_world_transform(
    snapshot: MeshSnapshot,
    *,
    identity_tolerance: float = 1.0e-10,
    singular_tolerance: float = 1.0e-12,
) -> MeshWorldTransformResult:
    """Bake rotation/scale/shear into positions and leave translation on the Object.

    Negative determinants are supported and preserve actual mirrored winding by using
    the cofactor normal transform. Singular transforms are rejected because they collapse
    at least one geometric dimension and cannot produce a stable Spine mesh.
    """

    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("snapshot must be MeshSnapshot")
    for field_name, value in (
        ("identity_tolerance", identity_tolerance),
        ("singular_tolerance", singular_tolerance),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"{field_name} must be a finite non-negative number")

    linear, translation = _matrix_parts(snapshot.world_matrix)
    determinant = float(_determinant(linear))
    coefficient_scale = max(1.0, *(abs(value) for value in linear))
    determinant_limit = float(singular_tolerance) * coefficient_scale**3
    if not isfinite(determinant_limit) or abs(determinant) <= determinant_limit:
        raise MeshWorldTransformError(
            "Object world transform is singular or numerically unstable; "
            f"determinant={determinant}, threshold={determinant_limit}"
        )

    translation_only = _translation_matrix(translation)
    changed = not _linear_is_identity(
        linear,
        tolerance=float(identity_tolerance),
    )
    if not changed and snapshot.world_matrix == translation_only:
        return MeshWorldTransformResult(
            snapshot=snapshot,
            linear_matrix=linear,
            translation=translation,
            determinant=determinant,
            mirrored=determinant < 0.0,
            changed=False,
        )

    cofactor = _cofactor_matrix(linear)
    vertices = tuple(
        replace(
            vertex,
            position=_multiply_vector(linear, vertex.position),
            normal=_normalized(
                _multiply_vector(cofactor, vertex.normal),
                field_name=f"vertex[{vertex.id.index}].normal",
            ),
        )
        for vertex in snapshot.vertices
    )
    faces = tuple(
        replace(
            face,
            normal=_normalized(
                _multiply_vector(cofactor, face.normal),
                field_name=f"face[{face.id.index}].normal",
            ),
        )
        for face in snapshot.faces
    )
    normalized_snapshot = replace(
        snapshot,
        vertices=vertices,
        faces=faces,
        world_matrix=translation_only,
    )
    MeshSnapshotValidator().validate_or_raise(normalized_snapshot)
    return MeshWorldTransformResult(
        snapshot=normalized_snapshot,
        linear_matrix=linear,
        translation=translation,
        determinant=determinant,
        mirrored=determinant < 0.0,
        changed=changed,
    )


__all__ = [
    "MeshWorldTransformError",
    "MeshWorldTransformResult",
    "normalize_mesh_snapshot_world_transform",
]
