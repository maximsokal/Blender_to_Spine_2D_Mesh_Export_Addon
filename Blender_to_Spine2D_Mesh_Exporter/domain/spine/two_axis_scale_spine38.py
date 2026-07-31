"""Spine 3.8-specific two-axis constraint scheduling.

For every two-axis object Spine 3.8 must evaluate constraints in this exact order:

1. ``<prefix>_scale_spine38_position``
2. ``<prefix>_rotation_X_constraint``
3. ``<prefix>_IK``
4. ``<prefix>_scale_rotate_X_constraint``
5. ``<prefix>_rotation_Y``
6. ``<prefix>_scale``

The first internal phase scales the collapse hierarchy before Rotation X. The public
Scale constraint remains last and applies the same control to final layer geometry.
Other Spine targets do not use this adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping, Sequence, TypeVar

from .model import Bone, IKConstraint, SpineDocument, TransformConstraint
from .spine41_setup_safety import validate_spine41_setup_safety
from .two_axis_scale_profile import TwoAxisScaleRigProfile
from .two_axis_scale_spine41 import adapt_two_axis_document_for_spine41_with_report
from .validator import SpineValidator


_ConstraintT = TypeVar("_ConstraintT", IKConstraint, TransformConstraint)


@dataclass(frozen=True, slots=True)
class Spine38TwoAxisDocumentAdaptation:
    document: SpineDocument
    old_to_new_bone_indices: Mapping[int, int]
    bridge_bone_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.document, SpineDocument):
            raise TypeError("document must be SpineDocument")
        if not isinstance(self.old_to_new_bone_indices, Mapping):
            raise TypeError("old_to_new_bone_indices must be a mapping")
        index_map: dict[int, int] = {}
        for old_index, new_index in self.old_to_new_bone_indices.items():
            if (
                isinstance(old_index, bool)
                or not isinstance(old_index, int)
                or old_index < 0
                or isinstance(new_index, bool)
                or not isinstance(new_index, int)
                or new_index < 0
            ):
                raise ValueError("bone index remapping must contain non-negative ints")
            index_map[old_index] = new_index
        if not isinstance(self.bridge_bone_names, tuple) or not all(
            isinstance(name, str) and name.strip() for name in self.bridge_bone_names
        ):
            raise ValueError("bridge_bone_names must contain non-empty strings")
        object.__setattr__(
            self,
            "old_to_new_bone_indices",
            MappingProxyType(index_map),
        )


def _named(
    constraints: Sequence[_ConstraintT],
    name: str,
    expected_type: type[_ConstraintT],
) -> _ConstraintT:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("constraint name must be a non-empty string")
    matches = tuple(item for item in constraints if item.name == name)
    if len(matches) != 1:
        raise ValueError(f"Expected one constraint {name!r}, found {len(matches)}")
    value = matches[0]
    if not isinstance(value, expected_type):
        raise TypeError(
            f"Constraint {name!r} must be {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )
    return value


def _optional_transform(
    constraints: Sequence[TransformConstraint],
    name: str,
) -> TransformConstraint | None:
    matches = tuple(item for item in constraints if item.name == name)
    if len(matches) > 1:
        raise ValueError(f"Duplicate transform constraint: {name!r}")
    return matches[0] if matches else None


def _position_name(prefix: str) -> str:
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("prefix must be a non-empty string")
    return f"{prefix}_scale_spine38_position"


def _identity_index_map(bones: tuple[Bone, ...]) -> Mapping[int, int]:
    if not isinstance(bones, tuple) or not bones:
        raise ValueError("bones must be a non-empty tuple")
    if not all(isinstance(bone, Bone) for bone in bones):
        raise TypeError("bones must contain Bone values")
    return MappingProxyType({index: index for index in range(len(bones))})


def _validate_scale_payload(constraint: TransformConstraint) -> None:
    if not isinstance(constraint, TransformConstraint):
        raise TypeError("constraint must be TransformConstraint")
    extras = dict(constraint.extras)
    if extras.get("relative") is not True or extras.get("local") not in {None, False}:
        raise ValueError(
            f"Scale constraint {constraint.name!r} must be relative world-space"
        )
    for field_name in ("mixRotate", "mixX", "mixShearY"):
        if extras.get(field_name) != 0:
            raise ValueError(
                f"Scale constraint {constraint.name!r} requires {field_name}=0"
            )


def _layer_topology(
    document: SpineDocument,
    depth: TransformConstraint,
    *,
    expected_bridge_parent: str,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return validated bridge, wrapper, and final-layer names in depth order."""

    bones = {bone.name: bone for bone in document.bones}
    if len(bones) != len(document.bones):
        raise ValueError("document contains duplicate bone names")
    children: dict[str, list[str]] = {}
    for bone in document.bones:
        if bone.parent is not None:
            children.setdefault(bone.parent, []).append(bone.name)

    bridges: list[str] = []
    wrappers: list[str] = []
    layers: list[str] = []
    for wrapper_name in depth.bones:
        wrapper = bones.get(wrapper_name)
        bridge_name = f"{wrapper_name}_spine41_bridge"
        bridge = bones.get(bridge_name)
        if wrapper is None or bridge is None:
            raise ValueError(
                f"Depth wrapper/bridge topology is incomplete for {wrapper_name!r}"
            )
        if wrapper.parent != bridge_name:
            raise ValueError(f"Wrapper {wrapper_name!r} has invalid bridge parent")
        if bridge.parent != expected_bridge_parent:
            raise ValueError(
                f"Bridge {bridge_name!r} must be parented to "
                f"{expected_bridge_parent!r}"
            )
        if wrapper.extras.get("inherit") != "onlyTranslation":
            raise ValueError(f"Wrapper {wrapper_name!r} must inherit only translation")
        if bridge.extras.get("inherit") != "onlyTranslation":
            raise ValueError(f"Bridge {bridge_name!r} must inherit only translation")
        layer_children = tuple(children.get(wrapper_name, ()))
        if len(layer_children) != 1:
            raise ValueError(
                f"Wrapper {wrapper_name!r} must have exactly one layer child"
            )
        bridges.append(bridge_name)
        wrappers.append(wrapper_name)
        layers.append(layer_children[0])

    if not wrappers:
        raise ValueError("depth constraint must constrain at least one wrapper")
    if len(set(wrappers)) != len(wrappers) or len(set(layers)) != len(layers):
        raise ValueError("wrapper/layer topology must be unique")
    return tuple(bridges), tuple(wrappers), tuple(layers)


def _validate_canonical_orders(
    rotation_x: TransformConstraint,
    ik: IKConstraint,
    scale: TransformConstraint,
    depth: TransformConstraint,
    rotation_y: TransformConstraint,
) -> int:
    base = rotation_x.order
    actual = (rotation_x.order, ik.order, scale.order, depth.order, rotation_y.order)
    expected = tuple(range(base, base + 5))
    if actual != expected:
        raise ValueError(
            "Spine 3.8 two-axis constraints must form canonical "
            f"X/IK/Scale/Depth/Y; expected={expected}, actual={actual}"
        )
    return base


def _validate_adapted_orders(
    position: TransformConstraint,
    rotation_x: TransformConstraint,
    ik: IKConstraint,
    depth: TransformConstraint,
    rotation_y: TransformConstraint,
    scale: TransformConstraint,
) -> int:
    base = position.order
    actual = (
        position.order,
        rotation_x.order,
        ik.order,
        depth.order,
        rotation_y.order,
        scale.order,
    )
    expected = tuple(range(base, base + 6))
    if actual != expected:
        raise ValueError(
            "Spine 3.8 two-axis constraints must form "
            "ScalePosition/X/IK/Depth/Y/ScaleGeometry; "
            f"expected={expected}, actual={actual}"
        )
    return base


def _replace_ik(
    constraints: tuple[IKConstraint, ...],
    replacement: IKConstraint,
) -> tuple[IKConstraint, ...]:
    matches = sum(item.name == replacement.name for item in constraints)
    if matches != 1:
        raise ValueError(
            f"Expected one IK constraint {replacement.name!r}, found {matches}"
        )
    return tuple(
        replacement if item.name == replacement.name else item
        for item in constraints
    )


def adapt_two_axis_document_for_spine38_with_report(
    document: SpineDocument,
    *,
    profile: TwoAxisScaleRigProfile,
    prefix: str,
) -> Spine38TwoAxisDocumentAdaptation:
    """Return an immutable Spine 3.8 document with position Scale first."""

    if not isinstance(document, SpineDocument):
        raise TypeError("document must be SpineDocument")
    if not isinstance(profile, TwoAxisScaleRigProfile):
        raise TypeError("profile must be TwoAxisScaleRigProfile")
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("prefix must be a non-empty string")

    position_name = _position_name(prefix)
    preexisting_position = _optional_transform(document.transform, position_name)
    if preexisting_position is None:
        legacy = adapt_two_axis_document_for_spine41_with_report(
            document,
            profile=profile,
            prefix=prefix,
        )
        adapted = legacy.document
        index_map = legacy.old_to_new_bone_indices
        reported_bridges = legacy.bridge_bone_names
    else:
        adapted = document
        index_map = _identity_index_map(document.bones)
        reported_bridges = ()

    rotation_x = _named(
        adapted.transform,
        profile.rotation_x_constraint(prefix),
        TransformConstraint,
    )
    ik = _named(adapted.ik, profile.scale_ik_constraint(prefix), IKConstraint)
    scale = _named(
        adapted.transform,
        profile.scale_constraint(prefix),
        TransformConstraint,
    )
    depth = _named(
        adapted.transform,
        profile.scale_depth_constraint(prefix),
        TransformConstraint,
    )
    rotation_y = _named(
        adapted.transform,
        profile.rotation_y_constraint(prefix),
        TransformConstraint,
    )
    position = _optional_transform(adapted.transform, position_name)

    bridges, wrappers, layers = _layer_topology(
        adapted,
        depth,
        expected_bridge_parent=profile.rotate_x_bone(prefix),
    )
    if reported_bridges and reported_bridges != bridges:
        raise ValueError("reported bridge names differ from document topology")
    if depth.bones != wrappers:
        raise ValueError("depth constraint must preserve wrapper order")
    if len(rotation_y.bones) != len(layers) or set(rotation_y.bones) != set(layers):
        raise ValueError("Rotation Y must constrain every final layer exactly once")

    _validate_scale_payload(scale)
    collapse = profile.scale_rotate_x_bone(prefix)
    if position is None:
        base = _validate_canonical_orders(rotation_x, ik, scale, depth, rotation_y)
        if scale.bones.count(collapse) != 1:
            raise ValueError("canonical Scale must contain exactly one collapse bone")
        layer_set = set(layers)
        scale_layers = tuple(name for name in scale.bones if name in layer_set)
        if (
            len(scale.bones) != len(layers) + 1
            or len(scale_layers) != len(layers)
            or set(scale_layers) != layer_set
        ):
            raise ValueError("canonical Scale must contain collapse and every layer")
        position = replace(
            scale,
            name=position_name,
            order=base,
            bones=(collapse,),
        )
        rotation_x = replace(rotation_x, order=base + 1)
        ik = replace(ik, order=base + 2)
        scale = replace(scale, order=base + 5, bones=scale_layers)
        created_position = True
    else:
        _validate_adapted_orders(
            position,
            rotation_x,
            ik,
            depth,
            rotation_y,
            scale,
        )
        _validate_scale_payload(position)
        if position.target != scale.target or dict(position.extras) != dict(scale.extras):
            raise ValueError("split Scale constraints must share target and payload")
        if position.bones != (collapse,):
            raise ValueError("position Scale must constrain only the collapse bone")
        if len(scale.bones) != len(layers) or set(scale.bones) != set(layers):
            raise ValueError("public Scale must constrain every final layer")
        created_position = False

    replacements = {
        position.name: position,
        rotation_x.name: rotation_x,
        scale.name: scale,
    }
    transform_values: list[TransformConstraint] = []
    inserted = not created_position
    for constraint in adapted.transform:
        if created_position and constraint.name == rotation_x.name:
            transform_values.append(position)
            inserted = True
        transform_values.append(replacements.get(constraint.name, constraint))
    if not inserted:
        raise RuntimeError("unable to insert position Scale before Rotation X")

    final_document = replace(
        adapted,
        ik=_replace_ik(adapted.ik, ik),
        transform=tuple(transform_values),
    )
    SpineValidator().validate_or_raise(final_document)
    validate_spine41_setup_safety(final_document)
    return Spine38TwoAxisDocumentAdaptation(
        document=final_document,
        old_to_new_bone_indices=index_map,
        bridge_bone_names=bridges,
    )


def adapt_two_axis_document_for_spine38(
    document: SpineDocument,
    *,
    profile: TwoAxisScaleRigProfile,
    prefix: str,
) -> SpineDocument:
    return adapt_two_axis_document_for_spine38_with_report(
        document,
        profile=profile,
        prefix=prefix,
    ).document


__all__ = [
    "Spine38TwoAxisDocumentAdaptation",
    "adapt_two_axis_document_for_spine38",
    "adapt_two_axis_document_for_spine38_with_report",
]
