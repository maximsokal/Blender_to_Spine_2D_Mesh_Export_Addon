"""Static setup-pose safety checks required by the Spine 4.1 runtime.

Spine 4.1 world-space transform constraints call ``Bone.updateAppliedTransform`` after
modifying each constrained bone. That operation inverts the constrained bone's parent
setup matrix. A generated rig whose parent matrix is singular can therefore produce
non-finite applied transforms even when the JSON schema itself is valid.

This module validates the typed canonical document before any Spine 4.1 JSON can be
considered runtime-safe. It never rewrites scales, hierarchy, or constraint payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, isfinite, pi, sin
from typing import Mapping, Tuple

from .model import Bone, SpineDocument, TransformConstraint


_DEGREES_TO_RADIANS = pi / 180.0
_SUPPORTED_INHERIT_VALUES = frozenset(
    {
        "normal",
        "onlyTranslation",
        "noRotationOrReflection",
        "noScale",
        "noScaleOrReflection",
    }
)


class Spine41RigSafetyError(ValueError):
    """Raised when a canonical rig cannot be evaluated safely by Spine 4.1."""


@dataclass(frozen=True, slots=True)
class Spine41SetupMatrix:
    """Immutable 2D setup matrix for one canonical Spine bone."""

    a: float
    b: float
    c: float
    d: float

    def __post_init__(self) -> None:
        values = (self.a, self.b, self.c, self.d)
        if not all(isinstance(value, float) and isfinite(value) for value in values):
            raise ValueError("Spine41SetupMatrix values must be finite floats")

    @property
    def determinant(self) -> float:
        return float(self.a * self.d - self.b * self.c)


@dataclass(frozen=True, slots=True)
class Spine41UnsafeWorldConstraint:
    """One world constraint whose parent setup matrix cannot be inverted."""

    constraint_name: str
    bone_name: str
    parent_name: str
    parent_determinant: float

    def __post_init__(self) -> None:
        for field_name in ("constraint_name", "bone_name", "parent_name"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(self.parent_determinant, float) or not isfinite(
            self.parent_determinant
        ):
            raise ValueError("parent_determinant must be a finite float")


def _number(value: object, default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Spine setup values must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError("Spine setup values must be finite")
    return result


def _inherit_value(bone: Bone) -> str:
    raw = bone.extras.get("inherit", "normal")
    if not isinstance(raw, str) or not raw:
        raise TypeError(f"Bone '{bone.name}' inherit must be a non-empty string")
    if raw not in _SUPPORTED_INHERIT_VALUES:
        raise Spine41RigSafetyError(
            f"Bone '{bone.name}' uses unsupported inherit mode {raw!r}"
        )
    return raw


def _local_matrix(bone: Bone) -> Spine41SetupMatrix:
    rotation = _number(bone.rotation, 0.0)
    scale_x = _number(bone.scale_x, 1.0)
    scale_y = _number(bone.scale_y, 1.0)
    shear_x = _number(bone.extras.get("shearX"), 0.0)
    shear_y = _number(bone.extras.get("shearY"), 0.0)

    rotation_x = (rotation + shear_x) * _DEGREES_TO_RADIANS
    rotation_y = (rotation + 90.0 + shear_y) * _DEGREES_TO_RADIANS
    return Spine41SetupMatrix(
        a=float(cos(rotation_x) * scale_x),
        b=float(cos(rotation_y) * scale_y),
        c=float(sin(rotation_x) * scale_x),
        d=float(sin(rotation_y) * scale_y),
    )


def _multiply(
    parent: Spine41SetupMatrix,
    local: Spine41SetupMatrix,
) -> Spine41SetupMatrix:
    return Spine41SetupMatrix(
        a=float(parent.a * local.a + parent.b * local.c),
        b=float(parent.a * local.b + parent.b * local.d),
        c=float(parent.c * local.a + parent.d * local.c),
        d=float(parent.c * local.b + parent.d * local.d),
    )


def _no_rotation_or_reflection_matrix(
    parent: Spine41SetupMatrix,
    bone: Bone,
) -> Spine41SetupMatrix:
    pa, pb, pc, pd = parent.a, parent.b, parent.c, parent.d
    parent_axis_length_squared = pa * pa + pc * pc
    parent_rotation = 0.0
    if parent_axis_length_squared > 0.0001:
        parent_scale = abs(parent.determinant) / parent_axis_length_squared
        pb = pc * parent_scale
        pd = pa * parent_scale
        parent_rotation = float(atan2(pc, pa) / _DEGREES_TO_RADIANS)
    else:
        pa = 0.0
        pc = 0.0
        parent_rotation = float(90.0 - atan2(pd, pb) / _DEGREES_TO_RADIANS)

    rotation = _number(bone.rotation, 0.0)
    scale_x = _number(bone.scale_x, 1.0)
    scale_y = _number(bone.scale_y, 1.0)
    shear_x = _number(bone.extras.get("shearX"), 0.0)
    shear_y = _number(bone.extras.get("shearY"), 0.0)
    rotation_x = (rotation + shear_x - parent_rotation) * _DEGREES_TO_RADIANS
    rotation_y = (
        rotation + shear_y - parent_rotation + 90.0
    ) * _DEGREES_TO_RADIANS
    local = Spine41SetupMatrix(
        a=float(cos(rotation_x) * scale_x),
        b=float(cos(rotation_y) * scale_y),
        c=float(sin(rotation_x) * scale_x),
        d=float(sin(rotation_y) * scale_y),
    )
    return Spine41SetupMatrix(
        a=float(pa * local.a - pb * local.c),
        b=float(pa * local.b - pb * local.d),
        c=float(pc * local.a + pd * local.c),
        d=float(pc * local.b + pd * local.d),
    )


def _no_scale_matrix(
    parent: Spine41SetupMatrix,
    bone: Bone,
    *,
    preserve_reflection: bool,
) -> Spine41SetupMatrix:
    rotation = _number(bone.rotation, 0.0) * _DEGREES_TO_RADIANS
    cosine = cos(rotation)
    sine = sin(rotation)
    axis_a = parent.a * cosine + parent.b * sine
    axis_c = parent.c * cosine + parent.d * sine
    axis_length = (axis_a * axis_a + axis_c * axis_c) ** 0.5
    if axis_length > 0.00001:
        axis_a /= axis_length
        axis_c /= axis_length

    perpendicular_scale = (axis_a * axis_a + axis_c * axis_c) ** 0.5
    if preserve_reflection and parent.determinant < 0.0:
        perpendicular_scale = -perpendicular_scale
    perpendicular_rotation = pi / 2.0 + atan2(axis_c, axis_a)
    axis_b = cos(perpendicular_rotation) * perpendicular_scale
    axis_d = sin(perpendicular_rotation) * perpendicular_scale

    scale_x = _number(bone.scale_x, 1.0)
    scale_y = _number(bone.scale_y, 1.0)
    shear_x = _number(bone.extras.get("shearX"), 0.0) * _DEGREES_TO_RADIANS
    shear_y = (
        90.0 + _number(bone.extras.get("shearY"), 0.0)
    ) * _DEGREES_TO_RADIANS
    local_a = cos(shear_x) * scale_x
    local_b = cos(shear_y) * scale_y
    local_c = sin(shear_x) * scale_x
    local_d = sin(shear_y) * scale_y
    return Spine41SetupMatrix(
        a=float(axis_a * local_a + axis_b * local_c),
        b=float(axis_a * local_b + axis_b * local_d),
        c=float(axis_c * local_a + axis_d * local_c),
        d=float(axis_c * local_b + axis_d * local_d),
    )


def calculate_spine41_setup_matrices(
    bones: Tuple[Bone, ...],
) -> Mapping[str, Spine41SetupMatrix]:
    """Calculate deterministic Spine 4.1 setup matrices in bone-array order."""

    if not isinstance(bones, tuple) or not bones:
        raise ValueError("bones must be a non-empty tuple")
    if not all(isinstance(bone, Bone) for bone in bones):
        raise TypeError("bones must contain Bone values")

    result: dict[str, Spine41SetupMatrix] = {}
    for bone in bones:
        if bone.name in result:
            raise ValueError(f"Duplicate bone name: {bone.name}")
        local = _local_matrix(bone)
        if bone.parent is None:
            result[bone.name] = local
            continue

        parent = result.get(bone.parent)
        if parent is None:
            raise Spine41RigSafetyError(
                f"Bone '{bone.name}' parent '{bone.parent}' must appear earlier"
            )
        inherit = _inherit_value(bone)
        if inherit == "normal":
            matrix = _multiply(parent, local)
        elif inherit == "onlyTranslation":
            matrix = local
        elif inherit == "noRotationOrReflection":
            matrix = _no_rotation_or_reflection_matrix(parent, bone)
        elif inherit == "noScale":
            matrix = _no_scale_matrix(
                parent,
                bone,
                preserve_reflection=True,
            )
        elif inherit == "noScaleOrReflection":
            matrix = _no_scale_matrix(
                parent,
                bone,
                preserve_reflection=False,
            )
        else:  # pragma: no cover - guarded by _inherit_value.
            raise AssertionError(f"Unhandled inherit mode: {inherit}")
        result[bone.name] = matrix

    return result


def _constraint_changes_world_transform(constraint: TransformConstraint) -> bool:
    if bool(constraint.extras.get("local", False)):
        return False

    mixes = (
        constraint.extras.get("mixRotate", 1),
        constraint.extras.get("mixX", 1),
        constraint.extras.get(
            "mixY",
            constraint.extras.get("mixX", 1),
        ),
        constraint.extras.get("mixScaleX", 1),
        constraint.extras.get(
            "mixScaleY",
            constraint.extras.get("mixScaleX", 1),
        ),
        constraint.extras.get("mixShearY", 1),
    )
    return any(abs(_number(value, 0.0)) > 0.0 for value in mixes)


def find_spine41_unsafe_world_constraints(
    document: SpineDocument,
    *,
    singular_tolerance: float = 1.0e-12,
) -> Tuple[Spine41UnsafeWorldConstraint, ...]:
    """Return every world constraint that must invert a singular parent matrix."""

    if not isinstance(document, SpineDocument):
        raise TypeError("document must be SpineDocument")
    tolerance = _number(singular_tolerance, 0.0)
    if tolerance < 0.0:
        raise ValueError("singular_tolerance must be non-negative")

    matrices = calculate_spine41_setup_matrices(document.bones)
    bone_by_name = {bone.name: bone for bone in document.bones}
    unsafe: list[Spine41UnsafeWorldConstraint] = []

    for constraint in document.transform:
        if not _constraint_changes_world_transform(constraint):
            continue
        for bone_name in constraint.bones:
            bone = bone_by_name.get(bone_name)
            if bone is None:
                raise Spine41RigSafetyError(
                    f"Constraint '{constraint.name}' references missing bone "
                    f"'{bone_name}'"
                )
            if bone.parent is None:
                continue
            parent_matrix = matrices[bone.parent]
            determinant = parent_matrix.determinant
            if abs(determinant) <= tolerance:
                unsafe.append(
                    Spine41UnsafeWorldConstraint(
                        constraint_name=constraint.name,
                        bone_name=bone.name,
                        parent_name=bone.parent,
                        parent_determinant=float(determinant),
                    )
                )

    return tuple(unsafe)


def validate_spine41_setup_safety(
    document: SpineDocument,
    *,
    singular_tolerance: float = 1.0e-12,
) -> None:
    """Fail closed when Spine 4.1 would invert a singular parent setup matrix."""

    unsafe = find_spine41_unsafe_world_constraints(
        document,
        singular_tolerance=singular_tolerance,
    )
    if not unsafe:
        return

    details = "; ".join(
        f"constraint={item.constraint_name!r}, bone={item.bone_name!r}, "
        f"parent={item.parent_name!r}, determinant={item.parent_determinant}"
        for item in unsafe
    )
    raise Spine41RigSafetyError(
        "Spine 4.1 cannot safely evaluate world-space transform constraints whose "
        "constrained bone has a singular parent setup matrix: "
        + details
    )


__all__ = [
    "Spine41RigSafetyError",
    "Spine41SetupMatrix",
    "Spine41UnsafeWorldConstraint",
    "calculate_spine41_setup_matrices",
    "find_spine41_unsafe_world_constraints",
    "validate_spine41_setup_safety",
]
