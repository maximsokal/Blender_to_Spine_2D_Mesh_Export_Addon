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

# Blender stores object transforms with single-precision components internally. Parent
# inverses, linked collection instances, and long hierarchy evaluation can therefore
# leave a small numerical residue in the homogeneous row of an otherwise affine object
# matrix. Values within this bound are canonicalized by the normalized output matrix;
# larger projective components remain unsupported and fail closed.
_AFFINE_ROW_FLOAT32_TOLERANCE = 1.0e-5


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
    """Return affine linear/translation parts and reject real projective transforms.

    The domain tuple is row-major. Small float32 residue in the homogeneous row is
    accepted because Blender Object transforms are semantically affine; the normalized
    snapshot later writes an exact ``(0, 0, 0, 1)`` row. A larger residue is treated as a
    genuine unsupported projective transform and remains a hard error.
    """

    if not isinstance(matrix, tuple) or len(matrix) != 16:
        raise TypeError("world_matrix must be a 16-value tuple")
    values = tuple(float(value) for value in matrix)
    if not all(isfinite(value) for value in values):
        raise MeshWorldTransformError("world_matrix contains non-finite values")

    final_row = (values[12], values[13], values[14], values[15])
    expected_row = (0.0, 0.0, 0.0, 1.0)
    if any(
        abs(actual - expected) > _AFFINE_ROW_FLOAT32_TOLERANCE
        for actual, expected in zip(final_row, expected_row, strict=True)
    ):
        raise MeshWorldTransformError(
            "world_matrix must be affine with final row (0, 0, 0, 1); "
            f"final_row={final_row!r}, "
            f"tolerance={_AFFINE_ROW_FLOAT32_TOLERANCE}"
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


def _relative_determinant(matrix: Matrix3x3, determinant: float) -> float:
    """Normalize determinant by axis magnitudes to measure linear dependence."""

    a, b, c, d, e, f, g, h, i = matrix
    first_length = sqrt(a * a + d * d + g * g)
    second_length = sqrt(b * b + e * e + h * h)
    third_length = sqrt(c * c + f * f + i * i)
    scale_product = first_length * second_length * third_length
    if not isfinite(scale_product) or scale_product <= 0.0:
        return 0.0
    result = abs(determinant) / scale_product
    return result if isfinite(result) else 0.0


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


def normalize_mesh_snapshot_world_transform(
    snapshot: MeshSnapshot,
    *,
    singular_tolerance: float = 1.0e-12,
) -> MeshWorldTransformResult:
    """Bake rotation/scale/shear into positions and leave translation on the Object.

    Negative determinants are supported and preserve actual mirrored winding by using
    the cofactor normal transform. Singular transforms are rejected because they collapse
    at least one geometric dimension and cannot produce a stable Spine mesh.
    """

    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("snapshot must be MeshSnapshot")
    if (
        isinstance(singular_tolerance, bool)
        or not isinstance(singular_tolerance, (int, float))
        or not isfinite(float(singular_tolerance))
        or float(singular_tolerance) < 0.0
    ):
        raise ValueError(
            "singular_tolerance must be a finite non-negative number"
        )

    linear, translation = _matrix_parts(snapshot.world_matrix)
    determinant = float(_determinant(linear))
    relative_determinant = _relative_determinant(linear, determinant)
    if relative_determinant <= float(singular_tolerance):
        raise MeshWorldTransformError(
            "Object world transform is singular or numerically unstable; "
            f"determinant={determinant}, relative_determinant={relative_determinant}, "
            f"threshold={float(singular_tolerance)}"
        )

    translation_only = _translation_matrix(translation)
    if snapshot.world_matrix == translation_only:
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
        changed=True,
    )


__all__ = [
    "MeshWorldTransformError",
    "MeshWorldTransformResult",
    "normalize_mesh_snapshot_world_transform",
]
