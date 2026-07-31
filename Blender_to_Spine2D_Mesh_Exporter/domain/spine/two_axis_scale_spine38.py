"""Build Spine 3.8-safe two-axis documents for the legacy update cache.

Spine 3.8 and Spine 4.0 evaluate the same JSON constraint graph differently when a
local transform constraint targets a bone that already appeared in the update cache.
Spine 3.8 does not reinsert that child after a later world constraint changes its parent.
The canonical five-phase two-axis graph therefore lets Rotation Y decompose stale
``*_1``/``*_2`` world matrices, producing shear when Rotation X or Scale is edited.

The target-specific solution keeps the verified legacy bridge topology and splits the
single public Scale operation into two constraints driven by the same Scale control:

1. the public ``<prefix>_scale`` constraint scales only ``*_scale_rotate_X``;
2. depth-scale then rebuilds the depth wrappers after that parent scale;
3. Rotation Y becomes the first update-cache owner of the final layer children;
4. an internal ``<prefix>_scale_spine38_layers`` constraint applies the same uniform
   scale to final layer matrices after all local rotation work is complete.

This preserves both parts of uniform object scaling: layer positions are recomputed from
the scaled collapse hierarchy, while attachment geometry receives the same uniform
factor without any later applied-transform decomposition. No epsilon scales, serialized
JSON repair, or fixture-specific names are used.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping, Sequence, TypeVar

from .model import IKConstraint, SpineDocument, TransformConstraint
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


def _layer_scale_constraint_name(prefix: str) -> str:
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("prefix must be a non-empty string")
    return f"{prefix}_scale_spine38_layers"


def _wrapper_layer_pairs(
    document: SpineDocument,
    depth_constraint: TransformConstraint,
) -> tuple[tuple[str, str], ...]:
    """Resolve every depth wrapper and its one direct final-layer child."""

    if not isinstance(document, SpineDocument):
        raise TypeError("document must be SpineDocument")
    if not isinstance(depth_constraint, TransformConstraint):
        raise TypeError("depth_constraint must be TransformConstraint")

    bone_by_name = {bone.name: bone for bone in document.bones}
    if len(bone_by_name) != len(document.bones):
        raise ValueError("Spine document contains duplicate bone names")

    children_by_parent: dict[str, list[str]] = {}
    for bone in document.bones:
        if bone.parent is not None:
            children_by_parent.setdefault(bone.parent, []).append(bone.name)

    pairs: list[tuple[str, str]] = []
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
        children = tuple(children_by_parent.get(wrapper_name, ()))
        if len(children) != 1:
            raise ValueError(
                f"Depth wrapper {wrapper_name!r} must have exactly one direct layer "
                f"child, found {len(children)}"
            )
        pairs.append((wrapper_name, children[0]))

    if not pairs:
        raise ValueError(
            f"Depth constraint {depth_constraint.name!r} must constrain wrappers"
        )
    if len({wrapper for wrapper, _layer in pairs}) != len(pairs):
        raise ValueError("Depth wrapper names must be unique")
    if len({layer for _wrapper, layer in pairs}) != len(pairs):
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


def _split_uniform_scale_constraint(
    source: TransformConstraint,
    *,
    existing_layer_scale: TransformConstraint | None,
    collapse_bone: str,
    layers: tuple[str, ...],
    layer_scale_name: str,
    layer_scale_order: int,
) -> tuple[TransformConstraint, TransformConstraint, bool]:
    """Return public position-scale and internal geometry-scale constraints."""

    if not isinstance(source, TransformConstraint):
        raise TypeError("source must be TransformConstraint")
    if not isinstance(collapse_bone, str) or not collapse_bone.strip():
        raise ValueError("collapse_bone must be a non-empty string")
    if not isinstance(layers, tuple) or not layers:
        raise ValueError("layers must be a non-empty tuple")
    if len(layers) != len(set(layers)):
        raise ValueError("layers must be unique")
    if not isinstance(layer_scale_name, str) or not layer_scale_name.strip():
        raise ValueError("layer_scale_name must be a non-empty string")
    if (
        isinstance(layer_scale_order, bool)
        or not isinstance(layer_scale_order, int)
        or layer_scale_order < 0
    ):
        raise ValueError("layer_scale_order must be a non-negative integer")

    _validate_scale_only_constraint(source)
    layer_set = set(layers)

    if existing_layer_scale is None:
        if source.bones.count(collapse_bone) != 1:
            raise ValueError(
                f"Uniform scale constraint {source.name!r} must contain exactly one "
                f"collapse bone {collapse_bone!r}; actual={source.bones}"
            )
        source_layers = tuple(name for name in source.bones if name in layer_set)
        if (
            len(source.bones) != len(layers) + 1
            or len(source_layers) != len(layers)
            or set(source_layers) != layer_set
        ):
            raise ValueError(
                f"Uniform scale constraint {source.name!r} must contain the collapse "
                f"bone and every final layer exactly once before Spine 3.8 splitting; "
                f"collapse={collapse_bone!r}, layers={layers}, actual={source.bones}"
            )
        public_scale = replace(source, bones=(collapse_bone,))
        layer_scale = replace(
            source,
            name=layer_scale_name,
            order=layer_scale_order,
            bones=source_layers,
        )
        return public_scale, layer_scale, True

    _validate_scale_only_constraint(existing_layer_scale)
    if source.bones != (collapse_bone,):
        raise ValueError(
            f"Adapted public Scale constraint {source.name!r} must constrain only "
            f"{collapse_bone!r}; actual={source.bones}"
        )
    if existing_layer_scale.target != source.target:
        raise ValueError(
            f"Internal Scale constraint {existing_layer_scale.name!r} must use target "
            f"{source.target!r}; actual={existing_layer_scale.target!r}"
        )
    if dict(existing_layer_scale.extras) != dict(source.extras):
        raise ValueError(
            f"Internal Scale constraint {existing_layer_scale.name!r} must preserve "
            "the public Scale payload"
        )
    if (
        existing_layer_scale.order != layer_scale_order
        or len(existing_layer_scale.bones) != len(layers)
        or set(existing_layer_scale.bones) != layer_set
    ):
        raise ValueError(
            f"Internal Scale constraint {existing_layer_scale.name!r} must own every "
            f"final layer at order {layer_scale_order}; expected={layers}, "
            f"actual_order={existing_layer_scale.order}, "
            f"actual_bones={existing_layer_scale.bones}"
        )
    return source, existing_layer_scale, False


def _validate_runtime_orders(
    *,
    rotation_x: TransformConstraint,
    scale_ik: IKConstraint,
    public_scale: TransformConstraint,
    depth_scale: TransformConstraint,
    rotation_y: TransformConstraint,
    layer_scale: TransformConstraint | None,
) -> tuple[int, bool]:
    """Validate either canonical five-phase or adapted six-phase order."""

    base_order = rotation_x.order
    canonical_orders = tuple(range(base_order, base_order + 5))
    authored_orders = (
        rotation_x.order,
        scale_ik.order,
        public_scale.order,
        depth_scale.order,
        rotation_y.order,
    )

    if layer_scale is None:
        if authored_orders != canonical_orders:
            raise ValueError(
                "Spine 3.8 two-axis constraints must form the canonical "
                "X/IK/Scale/Depth/Y block before target adaptation; "
                f"expected={canonical_orders}, actual={authored_orders}"
            )
        return base_order + 5, True

    adapted_orders = authored_orders + (layer_scale.order,)
    expected_adapted = tuple(range(base_order, base_order + 6))
    if adapted_orders != expected_adapted:
        raise ValueError(
            "Spine 3.8 two-axis constraints must form the adapted "
            "X/IK/ScalePosition/Depth/Y/ScaleGeometry block; "
            f"expected={expected_adapted}, actual={adapted_orders}"
        )
    return base_order + 5, False


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

    legacy = adapt_two_axis_document_for_spine41_with_report(
        document,
        profile=profile,
        prefix=prefix,
    )
    adapted_document = legacy.document

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
    layer_scale_name = _layer_scale_constraint_name(prefix)
    existing_layer_scale = _optional_transform_by_name(
        adapted_document.transform,
        layer_scale_name,
    )

    wrapper_layer_pairs = _wrapper_layer_pairs(adapted_document, depth_scale)
    wrappers = tuple(wrapper for wrapper, _layer in wrapper_layer_pairs)
    layers = tuple(layer for _wrapper, layer in wrapper_layer_pairs)
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

    layer_scale_order, should_create_layer_scale = _validate_runtime_orders(
        rotation_x=rotation_x,
        scale_ik=scale_ik,
        public_scale=public_scale,
        depth_scale=depth_scale,
        rotation_y=rotation_y,
        layer_scale=existing_layer_scale,
    )
    public_scale, layer_scale, created_layer_scale = _split_uniform_scale_constraint(
        public_scale,
        existing_layer_scale=existing_layer_scale,
        collapse_bone=profile.scale_rotate_x_bone(prefix),
        layers=layers,
        layer_scale_name=layer_scale_name,
        layer_scale_order=layer_scale_order,
    )
    if created_layer_scale is not should_create_layer_scale:
        raise RuntimeError("Spine 3.8 Scale split state is inconsistent")

    transformed_by_name = {
        constraint.name: constraint for constraint in adapted_document.transform
    }
    transformed_by_name[public_scale.name] = public_scale
    transformed_by_name[layer_scale.name] = layer_scale
    transform = tuple(
        transformed_by_name[constraint.name]
        for constraint in adapted_document.transform
    )
    if created_layer_scale:
        transform += (layer_scale,)

    final_document = replace(adapted_document, transform=transform)
    SpineValidator().validate_or_raise(final_document)
    validate_spine41_setup_safety(final_document)
    return Spine38TwoAxisDocumentAdaptation(
        document=final_document,
        old_to_new_bone_indices=legacy.old_to_new_bone_indices,
        bridge_bone_names=legacy.bridge_bone_names,
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
