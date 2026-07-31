"""Build Spine 3.8-safe two-axis documents for the legacy update cache.

Spine 3.8 and Spine 4.0 evaluate the same JSON constraint graph differently when a
local transform constraint targets a bone that already appeared in the update cache.
Spine 3.8 does not reinsert that child after a later world constraint changes its parent.
The canonical five-phase two-axis graph therefore lets Rotation Y decompose stale
``*_1``/``*_2`` world matrices, producing shear when Rotation X or Scale is edited.

The target-specific solution keeps the verified legacy bridge topology and splits the
single Scale control into two transform constraints driven by the same control bone:

1. an internal ``<prefix>_scale_spine38_position`` constraint scales only
   ``<prefix>_scale_rotate_X`` before depth evaluation;
2. depth-scale rebuilds the depth wrappers after that parent scale;
3. Rotation Y becomes the first update-cache owner of the final layer children;
4. the public ``<prefix>_scale`` constraint remains the final geometry-scale phase and
   uniformly scales the final layer matrices after all local rotation work completes.

Keeping the public Scale name on the geometry phase preserves diagnostics and runtime
probes that disable the user-facing scale constraint. The internal position phase keeps
layer distances responsive to the same control. No epsilon scales, serialized JSON
repair, or fixture-specific names are used.
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
    """Complete immutable result of one Spine 3.8 two-axis adaptation."""

    document: SpineDocument
    old_to_new_bone_indices: Mapping[int, int]
    bridge_bone_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.document, SpineDocument):
            raise TypeError("document must be SpineDocument")
        if not isinstance(self.old_to_new_bone_indices, Mapping):
            raise TypeError("old_to_new_bone_indices must be a mapping")
        normalized: dict[int, int] = {}
        for old_index, new_index in self.old_to_new_bone_indices.items():
            if (
                isinstance(old_index, bool)
                or not isinstance(old_index, int)
                or old_index < 0
            ):
                raise ValueError("old bone indices must be non-negative integers")
            if (
                isinstance(new_index, bool)
                or not isinstance(new_index, int)
                or new_index < 0
            ):
                raise ValueError("new bone indices must be non-negative integers")
            normalized[old_index] = new_index
        if not isinstance(self.bridge_bone_names, tuple) or not all(
            isinstance(name, str) and name.strip()
            for name in self.bridge_bone_names
        ):
            raise ValueError("bridge_bone_names must contain non-empty strings")
        object.__setattr__(
            self,
            "old_to_new_bone_indices",
            MappingProxyType(normalized),
        )


def _constraint_by_name(
    constraints: Sequence[_ConstraintT],
    name: str,
    *,
    expected_type: type[_ConstraintT],
) -> _ConstraintT:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("constraint name must be a non-empty string")
    matches = tuple(item for item in constraints if item.name == name)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one constraint named {name!r}, found {len(matches)}"
        )
    resolved = matches[0]
    if not isinstance(resolved, expected_type):
        raise TypeError(
            f"Constraint {name!r} must be {expected_type.__name__}, "
            f"got {type(resolved).__name__}"
        )
    return resolved


def _optional_transform_by_name(
    constraints: Sequence[TransformConstraint],
    name: str,
) -> TransformConstraint | None:
    matches = tuple(item for item in constraints if item.name == name)
    if len(matches) > 1:
        raise ValueError(
            f"Expected at most one transform constraint named {name!r}, "
            f"found {len(matches)}"
        )
    return matches[0] if matches else None


def _position_scale_constraint_name(prefix: str) -> str:
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("prefix must be a non-empty string")
    return f"{prefix}_scale_spine38_position"


def _expected_bridge_name(wrapper_name: str) -> str:
    if not isinstance(wrapper_name, str) or not wrapper_name.strip():
        raise ValueError("wrapper_name must be a non-empty string")
    return f"{wrapper_name}_spine41_bridge"


def _wrapper_layer_pairs(
    document: SpineDocument,
    depth_constraint: TransformConstraint,
    *,
    expected_parent_name: str,
) -> tuple[tuple[str, str, str], ...]:
    """Resolve and validate every bridge, depth wrapper, and final layer child."""

    if not isinstance(document, SpineDocument):
        raise TypeError("document must be SpineDocument")
    if not isinstance(depth_constraint, TransformConstraint):
        raise TypeError("depth_constraint must be TransformConstraint")
    if not isinstance(expected_parent_name, str) or not expected_parent_name.strip():
        raise ValueError("expected_parent_name must be a non-empty string")

    bone_by_name = {bone.name: bone for bone in document.bones}
    if len(bone_by_name) != len(document.bones):
        raise ValueError("Spine document contains duplicate bone names")

    children_by_parent: dict[str, list[str]] = {}
    for bone in document.bones:
        if bone.parent is not None:
            children_by_parent.setdefault(bone.parent, []).append(bone.name)

    pairs: list[tuple[str, str, str]] = []
    for wrapper_name in depth_constraint.bones:
        wrapper = bone_by_name.get(wrapper_name)
        if wrapper is None:
            raise ValueError(
                f"Depth constraint {depth_constraint.name!r} references missing "
                f"wrapper {wrapper_name!r}"
            )
        if wrapper.extras.get("inherit") != "onlyTranslation":
            raise ValueError(
                f"Depth wrapper {wrapper_name!r} must use inherit=onlyTranslation"
            )

        bridge_name = _expected_bridge_name(wrapper_name)
        bridge = bone_by_name.get(bridge_name)
        if bridge is None:
            raise ValueError(
                f"Depth wrapper {wrapper_name!r} is missing bridge {bridge_name!r}"
            )
        if wrapper.parent != bridge_name:
            raise ValueError(
                f"Depth wrapper {wrapper_name!r} must be parented to {bridge_name!r}"
            )
        if bridge.parent != expected_parent_name:
            raise ValueError(
                f"Depth bridge {bridge_name!r} must be parented to "
                f"{expected_parent_name!r}; actual={bridge.parent!r}"
            )
        if bridge.extras.get("inherit") != "onlyTranslation":
            raise ValueError(
                f"Depth bridge {bridge_name!r} must use inherit=onlyTranslation"
            )

        children = tuple(children_by_parent.get(wrapper_name, ()))
        if len(children) != 1:
            raise ValueError(
                f"Depth wrapper {wrapper_name!r} must have exactly one direct layer "
                f"child, found {len(children)}"
            )
        pairs.append((bridge_name, wrapper_name, children[0]))

    if not pairs:
        raise ValueError(
            f"Depth constraint {depth_constraint.name!r} must constrain wrappers"
        )
    if len({wrapper for _bridge, wrapper, _layer in pairs}) != len(pairs):
        raise ValueError("Depth wrapper names must be unique")
    if len({layer for _bridge, _wrapper, layer in pairs}) != len(pairs):
        raise ValueError("Final layer names must be unique")
    return tuple(pairs)


def _validate_scale_only_constraint(constraint: TransformConstraint) -> None:
    if not isinstance(constraint, TransformConstraint):
        raise TypeError("constraint must be TransformConstraint")
    extras = dict(constraint.extras)
    if extras.get("relative") is not True:
        raise ValueError(
            f"Uniform scale constraint {constraint.name!r} must be relative"
        )
    if extras.get("local") not in {None, False}:
        raise ValueError(
            f"Uniform scale constraint {constraint.name!r} must remain world-space"
        )
    for field_name in ("mixRotate", "mixX", "mixShearY"):
        if extras.get(field_name) != 0:
            raise ValueError(
                f"Uniform scale constraint {constraint.name!r} requires "
                f"{field_name}=0"
            )


def _canonical_scale_layers(
    constraint: TransformConstraint,
    *,
    collapse_bone: str,
    layers: tuple[str, ...],
) -> tuple[str, ...]:
    """Validate canonical collapse-plus-layers ownership and preserve layer order."""

    _validate_scale_only_constraint(constraint)
    if not isinstance(collapse_bone, str) or not collapse_bone.strip():
        raise ValueError("collapse_bone must be a non-empty string")
    if not isinstance(layers, tuple) or not layers:
        raise ValueError("layers must be a non-empty tuple")
    if len(layers) != len(set(layers)):
        raise ValueError("layers must be unique")

    if constraint.bones.count(collapse_bone) != 1:
        raise ValueError(
            f"Uniform scale constraint {constraint.name!r} must contain exactly one "
            f"collapse bone {collapse_bone!r}; actual={constraint.bones}"
        )
    layer_set = set(layers)
    source_layers = tuple(name for name in constraint.bones if name in layer_set)
    if (
        len(constraint.bones) != len(layers) + 1
        or len(source_layers) != len(layers)
        or set(source_layers) != layer_set
    ):
        raise ValueError(
            f"Uniform scale constraint {constraint.name!r} must contain the collapse "
            f"bone and every final layer exactly once; collapse={collapse_bone!r}, "
            f"layers={layers}, actual={constraint.bones}"
        )
    return source_layers


def _validate_canonical_orders(
    *,
    rotation_x: TransformConstraint,
    scale_ik: IKConstraint,
    scale: TransformConstraint,
    depth_scale: TransformConstraint,
    rotation_y: TransformConstraint,
) -> int:
    base_order = rotation_x.order
    actual = (
        rotation_x.order,
        scale_ik.order,
        scale.order,
        depth_scale.order,
        rotation_y.order,
    )
    expected = tuple(range(base_order, base_order + 5))
    if actual != expected:
        raise ValueError(
            "Spine 3.8 two-axis constraints must form the canonical "
            "X/IK/Scale/Depth/Y block before target adaptation; "
            f"expected={expected}, actual={actual}"
        )
    return base_order


def _validate_adapted_orders(
    *,
    rotation_x: TransformConstraint,
    scale_ik: IKConstraint,
    position_scale: TransformConstraint,
    depth_scale: TransformConstraint,
    rotation_y: TransformConstraint,
    public_scale: TransformConstraint,
) -> int:
    base_order = rotation_x.order
    actual = (
        rotation_x.order,
        scale_ik.order,
        position_scale.order,
        depth_scale.order,
        rotation_y.order,
        public_scale.order,
    )
    expected = tuple(range(base_order, base_order + 6))
    if actual != expected:
        raise ValueError(
            "Spine 3.8 two-axis constraints must form the adapted "
            "X/IK/ScalePosition/Depth/Y/ScaleGeometry block; "
            f"expected={expected}, actual={actual}"
        )
    return base_order


def _build_scale_phases(
    public_scale: TransformConstraint,
    *,
    existing_position_scale: TransformConstraint | None,
    collapse_bone: str,
    layers: tuple[str, ...],
    position_scale_name: str,
    base_order: int,
) -> tuple[TransformConstraint, TransformConstraint, bool]:
    """Return internal position-scale and public geometry-scale constraints."""

    if not isinstance(public_scale, TransformConstraint):
        raise TypeError("public_scale must be TransformConstraint")
    if not isinstance(position_scale_name, str) or not position_scale_name.strip():
        raise ValueError("position_scale_name must be a non-empty string")
    if isinstance(base_order, bool) or not isinstance(base_order, int) or base_order < 0:
        raise ValueError("base_order must be a non-negative integer")

    if existing_position_scale is None:
        source_layers = _canonical_scale_layers(
            public_scale,
            collapse_bone=collapse_bone,
            layers=layers,
        )
        position_scale = replace(
            public_scale,
            name=position_scale_name,
            order=base_order + 2,
            bones=(collapse_bone,),
        )
        adapted_public_scale = replace(
            public_scale,
            order=base_order + 5,
            bones=source_layers,
        )
        return position_scale, adapted_public_scale, True

    _validate_scale_only_constraint(existing_position_scale)
    _validate_scale_only_constraint(public_scale)
    if existing_position_scale.target != public_scale.target:
        raise ValueError(
            f"Split Scale constraints {existing_position_scale.name!r} and "
            f"{public_scale.name!r} must use the same target"
        )
    if dict(existing_position_scale.extras) != dict(public_scale.extras):
        raise ValueError("Split Spine 3.8 Scale constraints must preserve one payload")
    if existing_position_scale.bones != (collapse_bone,):
        raise ValueError(
            f"Position Scale constraint {existing_position_scale.name!r} must constrain "
            f"only {collapse_bone!r}; actual={existing_position_scale.bones}"
        )
    if existing_position_scale.order != base_order + 2:
        raise ValueError(
            f"Position Scale constraint {existing_position_scale.name!r} must use "
            f"order {base_order + 2}; actual={existing_position_scale.order}"
        )
    if (
        public_scale.order != base_order + 5
        or len(public_scale.bones) != len(layers)
        or set(public_scale.bones) != set(layers)
    ):
        raise ValueError(
            f"Public Scale constraint {public_scale.name!r} must own every final layer "
            f"at order {base_order + 5}; expected={layers}, "
            f"actual_order={public_scale.order}, actual_bones={public_scale.bones}"
        )
    return existing_position_scale, public_scale, False


def _identity_index_map(bones: tuple[Bone, ...]) -> Mapping[int, int]:
    if not isinstance(bones, tuple) or not bones:
        raise ValueError("bones must be a non-empty tuple")
    if not all(isinstance(bone, Bone) for bone in bones):
        raise TypeError("bones must contain Bone values")
    return MappingProxyType({index: index for index in range(len(bones))})


def adapt_two_axis_document_for_spine38_with_report(
    document: SpineDocument,
    *,
    profile: TwoAxisScaleRigProfile,
    prefix: str,
) -> Spine38TwoAxisDocumentAdaptation:
    """Return a Spine 3.8 cache-safe document plus exact bone-index remapping."""

    if not isinstance(document, SpineDocument):
        raise TypeError("document must be SpineDocument")
    if not isinstance(profile, TwoAxisScaleRigProfile):
        raise TypeError("profile must be TwoAxisScaleRigProfile")
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("prefix must be a non-empty string")

    position_scale_name = _position_scale_constraint_name(prefix)
    preexisting_position_scale = _optional_transform_by_name(
        document.transform,
        position_scale_name,
    )
    if preexisting_position_scale is None:
        legacy = adapt_two_axis_document_for_spine41_with_report(
            document,
            profile=profile,
            prefix=prefix,
        )
        adapted_document = legacy.document
        old_to_new_bone_indices = legacy.old_to_new_bone_indices
        reported_bridge_names = legacy.bridge_bone_names
    else:
        adapted_document = document
        old_to_new_bone_indices = _identity_index_map(document.bones)
        reported_bridge_names = ()

    rotation_x = _constraint_by_name(
        adapted_document.transform,
        profile.rotation_x_constraint(prefix),
        expected_type=TransformConstraint,
    )
    scale_ik = _constraint_by_name(
        adapted_document.ik,
        profile.scale_ik_constraint(prefix),
        expected_type=IKConstraint,
    )
    public_scale = _constraint_by_name(
        adapted_document.transform,
        profile.scale_constraint(prefix),
        expected_type=TransformConstraint,
    )
    depth_scale = _constraint_by_name(
        adapted_document.transform,
        profile.scale_depth_constraint(prefix),
        expected_type=TransformConstraint,
    )
    rotation_y = _constraint_by_name(
        adapted_document.transform,
        profile.rotation_y_constraint(prefix),
        expected_type=TransformConstraint,
    )
    existing_position_scale = _optional_transform_by_name(
        adapted_document.transform,
        position_scale_name,
    )

    bridge_wrapper_layer = _wrapper_layer_pairs(
        adapted_document,
        depth_scale,
        expected_parent_name=profile.rotate_x_bone(prefix),
    )
    bridge_names = tuple(item[0] for item in bridge_wrapper_layer)
    wrappers = tuple(item[1] for item in bridge_wrapper_layer)
    layers = tuple(item[2] for item in bridge_wrapper_layer)
    if reported_bridge_names and reported_bridge_names != bridge_names:
        raise ValueError(
            "Spine 3.8 bridge report differs from validated document topology: "
            f"reported={reported_bridge_names}, actual={bridge_names}"
        )
    if depth_scale.bones != wrappers:
        raise ValueError(
            f"Depth constraint {depth_scale.name!r} must preserve wrapper order; "
            f"expected={wrappers}, actual={depth_scale.bones}"
        )
    if len(rotation_y.bones) != len(layers) or set(rotation_y.bones) != set(layers):
        raise ValueError(
            f"Rotation Y constraint {rotation_y.name!r} must constrain every final "
            f"layer exactly once; expected={layers}, actual={rotation_y.bones}"
        )

    if existing_position_scale is None:
        base_order = _validate_canonical_orders(
            rotation_x=rotation_x,
            scale_ik=scale_ik,
            scale=public_scale,
            depth_scale=depth_scale,
            rotation_y=rotation_y,
        )
    else:
        base_order = _validate_adapted_orders(
            rotation_x=rotation_x,
            scale_ik=scale_ik,
            position_scale=existing_position_scale,
            depth_scale=depth_scale,
            rotation_y=rotation_y,
            public_scale=public_scale,
        )

    position_scale, public_scale, created_position_scale = _build_scale_phases(
        public_scale,
        existing_position_scale=existing_position_scale,
        collapse_bone=profile.scale_rotate_x_bone(prefix),
        layers=layers,
        position_scale_name=position_scale_name,
        base_order=base_order,
    )

    transformed_by_name = {
        constraint.name: constraint for constraint in adapted_document.transform
    }
    transformed_by_name[position_scale.name] = position_scale
    transformed_by_name[public_scale.name] = public_scale
    if created_position_scale:
        transform_values: list[TransformConstraint] = []
        for constraint in adapted_document.transform:
            if constraint.name == public_scale.name:
                transform_values.append(position_scale)
                transform_values.append(public_scale)
            else:
                transform_values.append(constraint)
        transform = tuple(transform_values)
    else:
        transform = tuple(
            transformed_by_name[constraint.name]
            for constraint in adapted_document.transform
        )

    final_document = replace(adapted_document, transform=transform)
    SpineValidator().validate_or_raise(final_document)
    validate_spine41_setup_safety(final_document)
    return Spine38TwoAxisDocumentAdaptation(
        document=final_document,
        old_to_new_bone_indices=old_to_new_bone_indices,
        bridge_bone_names=bridge_names,
    )


def adapt_two_axis_document_for_spine38(
    document: SpineDocument,
    *,
    profile: TwoAxisScaleRigProfile,
    prefix: str,
) -> SpineDocument:
    """Compatibility wrapper returning only the adapted immutable document."""

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
