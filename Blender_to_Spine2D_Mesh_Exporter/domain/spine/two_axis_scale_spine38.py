"""Build Spine 3.8-safe two-axis documents for the legacy update cache.

Spine 3.8 and Spine 4.0 evaluate the same JSON constraint schema differently when a
local transform constraint targets a bone that was already inserted into the runtime
update cache. Spine 3.8 does not reinsert that child after a later world constraint
changes its parent. The canonical two-axis schedule therefore leaves ``*_1``/``*_2``
with a stale world matrix before the local Rotation Y constraint and the runtime derives
an unintended skewed applied transform.

The target-specific solution keeps the verified Spine 4.1 bridge topology, but changes
only the two phases that participate in the stale-child dependency:

- depth-scale evaluates before uniform Scale;
- uniform Scale constrains the invertible depth wrappers instead of their final layer
  children;
- Rotation Y remains last and becomes the first update-cache owner of the final layer
  children.

No epsilon scales, JSON post-processing, or fixture-specific names are used. The typed
document remains immutable and weighted attachment indices keep the exact remapping
reported by the shared legacy bridge adapter.
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


def _adapt_uniform_scale_bones(
    constraint: TransformConstraint,
    *,
    collapse_bone: str,
    wrapper_layer_pairs: tuple[tuple[str, str], ...],
) -> TransformConstraint:
    """Move uniform scale ownership from final layers to their wrappers."""

    if not isinstance(constraint, TransformConstraint):
        raise TypeError("constraint must be TransformConstraint")
    if not isinstance(collapse_bone, str) or not collapse_bone.strip():
        raise ValueError("collapse_bone must be a non-empty string")
    if not isinstance(wrapper_layer_pairs, tuple) or not wrapper_layer_pairs:
        raise ValueError("wrapper_layer_pairs must be a non-empty tuple")

    wrappers = tuple(wrapper for wrapper, _layer in wrapper_layer_pairs)
    layers = tuple(layer for _wrapper, layer in wrapper_layer_pairs)
    wrapper_set = set(wrappers)
    layer_to_wrapper = {
        layer: wrapper for wrapper, layer in wrapper_layer_pairs
    }

    if constraint.bones.count(collapse_bone) != 1:
        raise ValueError(
            f"Uniform scale constraint {constraint.name!r} must contain exactly one "
            f"collapse bone {collapse_bone!r}; actual={constraint.bones}"
        )

    adapted_bones: list[str] = []
    for bone_name in constraint.bones:
        if bone_name == collapse_bone or bone_name in wrapper_set:
            adapted_bones.append(bone_name)
            continue
        wrapper_name = layer_to_wrapper.get(bone_name)
        if wrapper_name is None:
            raise ValueError(
                f"Uniform scale constraint {constraint.name!r} contains unsupported "
                f"bone {bone_name!r}; expected collapse={collapse_bone!r}, "
                f"wrappers={wrappers}, or layers={layers}"
            )
        adapted_bones.append(wrapper_name)

    expected = {collapse_bone, *wrappers}
    if len(adapted_bones) != len(expected) or set(adapted_bones) != expected:
        raise ValueError(
            f"Uniform scale constraint {constraint.name!r} must resolve to exactly "
            f"the collapse bone and every depth wrapper; actual={tuple(adapted_bones)}"
        )
    if len(adapted_bones) != len(set(adapted_bones)):
        raise ValueError(
            f"Uniform scale constraint {constraint.name!r} repeats constrained bones"
        )
    if any(layer in adapted_bones for layer in layers):
        raise ValueError(
            f"Uniform scale constraint {constraint.name!r} cannot constrain final "
            "layer children in Spine 3.8"
        )

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

    return replace(constraint, bones=tuple(adapted_bones))


def _adapt_runtime_orders(
    *,
    rotation_x: TransformConstraint,
    scale_ik: IKConstraint,
    uniform_scale: TransformConstraint,
    depth_scale: TransformConstraint,
    rotation_y: TransformConstraint,
) -> tuple[TransformConstraint, TransformConstraint]:
    """Place depth before uniform scale while preserving one dense five-phase block."""

    constraints = (
        rotation_x,
        scale_ik,
        uniform_scale,
        depth_scale,
        rotation_y,
    )
    if not all(
        isinstance(item, (IKConstraint, TransformConstraint))
        for item in constraints
    ):
        raise TypeError("runtime schedule contains invalid constraint values")

    base_order = rotation_x.order
    canonical_orders = (
        base_order,
        base_order + 1,
        base_order + 2,
        base_order + 3,
        base_order + 4,
    )
    current_orders = tuple(item.order for item in constraints)
    adapted_orders = (
        base_order,
        base_order + 1,
        base_order + 3,
        base_order + 2,
        base_order + 4,
    )

    if current_orders == canonical_orders:
        return (
            replace(uniform_scale, order=base_order + 3),
            replace(depth_scale, order=base_order + 2),
        )
    if current_orders == adapted_orders:
        return uniform_scale, depth_scale
    raise ValueError(
        "Spine 3.8 two-axis constraints must form either the canonical "
        "X/IK/Scale/Depth/Y block or the adapted X/IK/Depth/Scale/Y block; "
        f"actual={current_orders}"
    )


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
    uniform_scale = _constraint_by_name(
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

    uniform_scale = _adapt_uniform_scale_bones(
        uniform_scale,
        collapse_bone=profile.scale_rotate_x_bone(prefix),
        wrapper_layer_pairs=wrapper_layer_pairs,
    )
    uniform_scale, depth_scale = _adapt_runtime_orders(
        rotation_x=rotation_x,
        scale_ik=scale_ik,
        uniform_scale=uniform_scale,
        depth_scale=depth_scale,
        rotation_y=rotation_y,
    )

    transformed_by_name = {
        constraint.name: constraint for constraint in adapted_document.transform
    }
    transformed_by_name[uniform_scale.name] = uniform_scale
    transformed_by_name[depth_scale.name] = depth_scale
    final_document = replace(
        adapted_document,
        transform=tuple(
            transformed_by_name[constraint.name]
            for constraint in adapted_document.transform
        ),
    )

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
