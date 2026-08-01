"""Evaluate Spine setup-pose bone transforms without importing Blender.

The connected/mixed acceptance workers need the final setup position of object main
bones. Connected two-axis rigs intentionally contain local +90/-90 degree wrapper bones,
zero helper scale, and ``onlyTranslation`` inheritance. Summing translations or rejecting
local rotations cannot represent that hierarchy.

This module implements the subset of the Spine setup transform contract used by generated
A1 rigs. Unsupported inheritance or shear data fails closed instead of silently producing
an incomplete oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, radians, sin
from typing import Mapping

from Blender_to_Spine2D_Mesh_Exporter.domain.spine.model import Bone, SpineDocument


_NORMAL_INHERIT = "normal"
_ONLY_TRANSLATION_INHERIT = "onlyTranslation"
_SUPPORTED_INHERIT = frozenset({_NORMAL_INHERIT, _ONLY_TRANSLATION_INHERIT})


class SpineSetupTransformError(ValueError):
    """Raised when a setup transform cannot be evaluated without guessing."""


@dataclass(frozen=True, slots=True)
class SpineSetupAffine2D:
    """One Spine-style 2D affine transform.

    Points are transformed as::

        world_x = a * local_x + b * local_y + x
        world_y = c * local_x + d * local_y + y
    """

    a: float
    b: float
    c: float
    d: float
    x: float
    y: float

    def __post_init__(self) -> None:
        for field_name in ("a", "b", "c", "d", "x", "y"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be a finite number")
            resolved = float(value)
            if not isfinite(resolved):
                raise ValueError(f"{field_name} must be finite")
            object.__setattr__(
                self,
                field_name,
                0.0 if resolved == 0.0 else resolved,
            )

    def transform_point(self, point: tuple[float, float]) -> tuple[float, float]:
        if not isinstance(point, tuple) or len(point) != 2:
            raise TypeError("point must contain two numeric values")
        local_x = _finite_number(point[0], "point[0]")
        local_y = _finite_number(point[1], "point[1]")
        return (
            self.a * local_x + self.b * local_y + self.x,
            self.c * local_x + self.d * local_y + self.y,
        )


@dataclass(frozen=True, slots=True)
class SpineSetupBoneResult:
    bone_name: str
    transform: SpineSetupAffine2D
    inherit_mode: str

    def __post_init__(self) -> None:
        if not isinstance(self.bone_name, str) or not self.bone_name.strip():
            raise ValueError("bone_name must be a non-empty string")
        if not isinstance(self.transform, SpineSetupAffine2D):
            raise TypeError("transform must be SpineSetupAffine2D")
        if self.inherit_mode not in _SUPPORTED_INHERIT:
            raise ValueError(f"Unsupported inherit_mode: {self.inherit_mode!r}")

    @property
    def position(self) -> tuple[float, float]:
        return self.transform.x, self.transform.y


def _finite_number(value: object, field_name: str, *, default: float | None = None) -> float:
    if value is None:
        if default is None:
            raise TypeError(f"{field_name} cannot be None")
        value = default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number")
    resolved = float(value)
    if not isfinite(resolved):
        raise ValueError(f"{field_name} must be finite")
    return 0.0 if resolved == 0.0 else resolved


def _inherit_mode(bone: Bone) -> str:
    extras = bone.extras
    if not isinstance(extras, Mapping):
        raise TypeError(f"Bone {bone.name!r} extras must be a mapping")

    inherit_value = extras.get("inherit")
    transform_value = extras.get("transform")
    if inherit_value is not None and transform_value is not None:
        if inherit_value != transform_value:
            raise SpineSetupTransformError(
                f"Bone {bone.name!r} defines conflicting inherit/transform modes: "
                f"{inherit_value!r} versus {transform_value!r}"
            )
    raw_mode = inherit_value if inherit_value is not None else transform_value
    mode = _NORMAL_INHERIT if raw_mode is None else raw_mode
    if not isinstance(mode, str):
        raise TypeError(f"Bone {bone.name!r} inherit mode must be str")
    if mode not in _SUPPORTED_INHERIT:
        raise SpineSetupTransformError(
            f"Bone {bone.name!r} uses unsupported inherit mode {mode!r}; "
            f"supported={tuple(sorted(_SUPPORTED_INHERIT))}"
        )

    for field_name in ("shear", "shearX", "shearY"):
        if field_name not in extras:
            continue
        value = _finite_number(extras[field_name], f"{bone.name}.{field_name}")
        if abs(value) > 1.0e-12:
            raise SpineSetupTransformError(
                f"Bone {bone.name!r} uses unsupported non-zero {field_name}={value}"
            )
    return mode


def _local_basis(bone: Bone) -> tuple[float, float, float, float]:
    rotation = radians(_finite_number(bone.rotation, f"{bone.name}.rotation", default=0.0))
    scale_x = _finite_number(bone.scale_x, f"{bone.name}.scale_x", default=1.0)
    scale_y = _finite_number(bone.scale_y, f"{bone.name}.scale_y", default=1.0)
    cosine = cos(rotation)
    sine = sin(rotation)
    return (
        cosine * scale_x,
        -sine * scale_y,
        sine * scale_x,
        cosine * scale_y,
    )


def _compose_normal(
    parent: SpineSetupAffine2D,
    *,
    local_x: float,
    local_y: float,
    local_basis: tuple[float, float, float, float],
) -> SpineSetupAffine2D:
    local_a, local_b, local_c, local_d = local_basis
    world_x, world_y = parent.transform_point((local_x, local_y))
    return SpineSetupAffine2D(
        a=parent.a * local_a + parent.b * local_c,
        b=parent.a * local_b + parent.b * local_d,
        c=parent.c * local_a + parent.d * local_c,
        d=parent.c * local_b + parent.d * local_d,
        x=world_x,
        y=world_y,
    )


def _compose_only_translation(
    parent: SpineSetupAffine2D,
    *,
    local_x: float,
    local_y: float,
    local_basis: tuple[float, float, float, float],
) -> SpineSetupAffine2D:
    """Match Spine ``onlyTranslation`` setup inheritance.

    The bone origin is still its parent-space local X/Y transformed by the complete
    parent matrix. Only the child basis stops inheriting parent rotation and scale.
    """

    world_x, world_y = parent.transform_point((local_x, local_y))
    local_a, local_b, local_c, local_d = local_basis
    return SpineSetupAffine2D(
        a=local_a,
        b=local_b,
        c=local_c,
        d=local_d,
        x=world_x,
        y=world_y,
    )


def evaluate_spine_setup_bone(
    document: SpineDocument,
    bone_name: str,
) -> SpineSetupBoneResult:
    """Evaluate one bone's setup affine transform through its complete parent chain."""

    if not isinstance(document, SpineDocument):
        raise TypeError("document must be SpineDocument")
    if not isinstance(bone_name, str) or not bone_name.strip():
        raise ValueError("bone_name must be a non-empty string")

    by_name: dict[str, Bone] = {}
    for bone in document.bones:
        if not isinstance(bone, Bone):
            raise TypeError("document.bones must contain Bone values")
        if bone.name in by_name:
            raise SpineSetupTransformError(
                f"Connected document contains duplicate bone name {bone.name!r}"
            )
        by_name[bone.name] = bone
    if bone_name not in by_name:
        raise SpineSetupTransformError(f"Bone {bone_name!r} is absent from document")

    cache: dict[str, SpineSetupBoneResult] = {}
    active: set[str] = set()

    def resolve(current_name: str) -> SpineSetupBoneResult:
        cached = cache.get(current_name)
        if cached is not None:
            return cached
        if current_name in active:
            raise SpineSetupTransformError(f"Bone parent cycle detected at {current_name!r}")
        try:
            bone = by_name[current_name]
        except KeyError as exc:
            raise SpineSetupTransformError(
                f"Missing parent bone {current_name!r}"
            ) from exc

        active.add(current_name)
        try:
            local_x = _finite_number(bone.x, f"{bone.name}.x", default=0.0)
            local_y = _finite_number(bone.y, f"{bone.name}.y", default=0.0)
            local_basis = _local_basis(bone)
            inherit_mode = _inherit_mode(bone)

            if bone.parent is None:
                local_a, local_b, local_c, local_d = local_basis
                transform = SpineSetupAffine2D(
                    a=local_a,
                    b=local_b,
                    c=local_c,
                    d=local_d,
                    x=local_x,
                    y=local_y,
                )
            else:
                parent = resolve(bone.parent).transform
                if inherit_mode == _NORMAL_INHERIT:
                    transform = _compose_normal(
                        parent,
                        local_x=local_x,
                        local_y=local_y,
                        local_basis=local_basis,
                    )
                elif inherit_mode == _ONLY_TRANSLATION_INHERIT:
                    transform = _compose_only_translation(
                        parent,
                        local_x=local_x,
                        local_y=local_y,
                        local_basis=local_basis,
                    )
                else:
                    raise AssertionError(f"Unhandled inherit mode: {inherit_mode}")

            result = SpineSetupBoneResult(
                bone_name=bone.name,
                transform=transform,
                inherit_mode=inherit_mode,
            )
            cache[current_name] = result
            return result
        finally:
            active.remove(current_name)

    return resolve(bone_name)


def evaluate_spine_setup_bone_position(
    document: SpineDocument,
    bone_name: str,
) -> tuple[float, float]:
    """Return the evaluated setup world origin of one bone."""

    return evaluate_spine_setup_bone(document, bone_name).position


__all__ = [
    "SpineSetupAffine2D",
    "SpineSetupBoneResult",
    "SpineSetupTransformError",
    "evaluate_spine_setup_bone",
    "evaluate_spine_setup_bone_position",
]
