"""Adapt rigid camera-relative two-axis documents for Spine 3.8-4.1.

The generic legacy adapters operate on the historical model-space rig where one Scale
constraint has two responsibilities: it scales the singular X-collapse hierarchy and
all final depth layers. Camera-relative rigs intentionally use a different contract:
Scale targets only the object ``base`` below the orbital X/Y layers, so resizing the
object cannot alter its distance from camera zero.

This module reuses the proven generic bridge/index-remap adapters through a temporary
model-space-compatible constraint view, then restores the camera-relative Scale target
and Spine 3.8 evaluation schedule on the immutable adapted document. No serialized JSON
patching or Blender runtime state is involved.
"""

from __future__ import annotations

from dataclasses import replace

from .model import Bone, IKConstraint, SpineDocument, TransformConstraint
from .spine41_setup_safety import validate_spine41_setup_safety
from .two_axis_scale_profile import TwoAxisScaleRigProfile
from .two_axis_scale_spine38 import (
    Spine38TwoAxisDocumentAdaptation,
    adapt_two_axis_document_for_spine38_with_report,
)
from .two_axis_scale_spine41 import (
    Spine41TwoAxisDocumentAdaptation,
    adapt_two_axis_document_for_spine41_with_report,
)
from .validator import SpineValidator


def _named_transform(
    document: SpineDocument,
    name: str,
) -> TransformConstraint:
    matches = tuple(item for item in document.transform if item.name == name)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one transform constraint {name!r}, found {len(matches)}"
        )
    return matches[0]


def _named_ik(
    document: SpineDocument,
    name: str,
) -> IKConstraint:
    matches = tuple(item for item in document.ik if item.name == name)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one IK constraint {name!r}, found {len(matches)}"
        )
    return matches[0]


def _bone_by_name(document: SpineDocument) -> dict[str, Bone]:
    result = {bone.name: bone for bone in document.bones}
    if len(result) != len(document.bones):
        raise ValueError("Camera-relative document contains duplicate bone names")
    return result


def _replace_transform_constraints(
    document: SpineDocument,
    replacements: dict[str, TransformConstraint],
    *,
    remove_names: frozenset[str] = frozenset(),
) -> SpineDocument:
    if not isinstance(document, SpineDocument):
        raise TypeError("document must be SpineDocument")
    if not isinstance(replacements, dict):
        raise TypeError("replacements must be dict")
    if not isinstance(remove_names, frozenset):
        raise TypeError("remove_names must be frozenset")

    source_names = {constraint.name for constraint in document.transform}
    unknown_replacements = tuple(sorted(set(replacements) - source_names))
    if unknown_replacements:
        raise ValueError(
            "Transform replacements reference unknown constraints: "
            f"{unknown_replacements}"
        )
    unknown_removals = tuple(sorted(set(remove_names) - source_names))
    if unknown_removals:
        raise ValueError(
            f"Transform removals reference unknown constraints: {unknown_removals}"
        )

    return replace(
        document,
        transform=tuple(
            replacements.get(constraint.name, constraint)
            for constraint in document.transform
            if constraint.name not in remove_names
        ),
    )


def _camera_relative_topology(
    document: SpineDocument,
    *,
    profile: TwoAxisScaleRigProfile,
    prefix: str,
) -> tuple[str, str, str]:
    """Return and validate ``base``, depth wrapper, and rigid layer names."""

    if not isinstance(document, SpineDocument):
        raise TypeError("document must be SpineDocument")
    if not isinstance(profile, TwoAxisScaleRigProfile):
        raise TypeError("profile must be TwoAxisScaleRigProfile")
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("prefix must be a non-empty string")

    bones = _bone_by_name(document)
    children: dict[str, list[str]] = {}
    for bone in document.bones:
        if bone.parent is not None:
            children.setdefault(bone.parent, []).append(bone.name)

    depth = _named_transform(
        document,
        profile.scale_depth_constraint(prefix),
    )
    if len(depth.bones) != 1:
        raise ValueError(
            "Camera-relative legacy finalization requires exactly one depth wrapper"
        )
    wrapper_name = depth.bones[0]
    wrapper = bones.get(wrapper_name)
    if wrapper is None:
        raise ValueError(f"Camera depth wrapper is missing: {wrapper_name!r}")
    if wrapper.extras.get("inherit") != "onlyTranslation":
        raise ValueError(
            f"Camera depth wrapper {wrapper_name!r} must inherit only translation"
        )

    rotate_x_name = profile.rotate_x_bone(prefix)
    bridge_name = f"{wrapper_name}_spine41_bridge"
    if wrapper.parent == rotate_x_name:
        pass
    elif wrapper.parent == bridge_name:
        bridge = bones.get(bridge_name)
        if bridge is None:
            raise ValueError(
                f"Camera depth wrapper references missing bridge {bridge_name!r}"
            )
        if bridge.parent != rotate_x_name:
            raise ValueError(
                f"Camera depth bridge {bridge_name!r} must be parented to "
                f"{rotate_x_name!r}"
            )
        if bridge.extras.get("inherit") != "onlyTranslation":
            raise ValueError(
                f"Camera depth bridge {bridge_name!r} must inherit only translation"
            )
    else:
        raise ValueError(
            f"Camera depth wrapper {wrapper_name!r} has unexpected parent "
            f"{wrapper.parent!r}"
        )

    layer_children = tuple(children.get(wrapper_name, ()))
    if len(layer_children) != 1:
        raise ValueError(
            f"Camera depth wrapper {wrapper_name!r} must have exactly one layer child"
        )
    layer_name = layer_children[0]

    base_name = profile.base_bone(prefix)
    base = bones.get(base_name)
    if base is None:
        raise ValueError(f"Camera-relative object base is missing: {base_name!r}")
    if base.parent != layer_name:
        raise ValueError(
            f"Camera-relative object base {base_name!r} must be parented to rigid "
            f"layer {layer_name!r}; actual={base.parent!r}"
        )

    scale = _named_transform(document, profile.scale_constraint(prefix))
    if scale.bones != (base_name,):
        raise ValueError(
            "Camera-relative Scale must constrain only the object base; "
            f"expected={(base_name,)}, actual={scale.bones}"
        )
    return base_name, wrapper_name, layer_name


def _temporary_scale_document(
    document: SpineDocument,
    *,
    profile: TwoAxisScaleRigProfile,
    prefix: str,
    include_layer: bool,
) -> tuple[SpineDocument, str, str]:
    """Build a temporary constraint view accepted by the generic bridge adapters."""

    base_name, _wrapper_name, layer_name = _camera_relative_topology(
        document,
        profile=profile,
        prefix=prefix,
    )
    scale_name = profile.scale_constraint(prefix)
    source_scale = _named_transform(document, scale_name)
    temporary_bones = (
        (profile.rotate_x_bone(prefix), layer_name)
        if include_layer
        else (profile.rotate_x_bone(prefix),)
    )
    temporary_scale = replace(source_scale, bones=temporary_bones)
    temporary = _replace_transform_constraints(
        document,
        {scale_name: temporary_scale},
    )
    return temporary, base_name, layer_name


def _normalize_spine38_temporary_orders(
    document: SpineDocument,
    *,
    profile: TwoAxisScaleRigProfile,
    prefix: str,
) -> SpineDocument:
    """Restore the five-phase canonical order expected by the generic 3.8 adapter."""

    rotation_x = _named_transform(
        document,
        profile.rotation_x_constraint(prefix),
    )
    scale = _named_transform(document, profile.scale_constraint(prefix))
    depth = _named_transform(
        document,
        profile.scale_depth_constraint(prefix),
    )
    rotation_y = _named_transform(
        document,
        profile.rotation_y_constraint(prefix),
    )
    ik = _named_ik(document, profile.scale_ik_constraint(prefix))

    orders = (
        rotation_x.order,
        ik.order,
        scale.order,
        depth.order,
        rotation_y.order,
    )
    base_order = min(orders)
    if set(orders) != set(range(base_order, base_order + 5)):
        raise ValueError(
            "Camera-relative Spine 3.8 constraints must occupy one contiguous "
            f"five-phase range; actual={orders}"
        )

    transform_replacements = {
        rotation_x.name: replace(rotation_x, order=base_order),
        scale.name: replace(scale, order=base_order + 2),
        depth.name: replace(depth, order=base_order + 3),
        rotation_y.name: replace(rotation_y, order=base_order + 4),
    }
    normalized = _replace_transform_constraints(
        document,
        transform_replacements,
    )
    normalized_ik = tuple(
        replace(item, order=base_order + 1) if item.name == ik.name else item
        for item in normalized.ik
    )
    return replace(normalized, ik=normalized_ik)


def _restore_camera_relative_scale(
    document: SpineDocument,
    *,
    profile: TwoAxisScaleRigProfile,
    prefix: str,
    base_name: str,
) -> SpineDocument:
    scale_name = profile.scale_constraint(prefix)
    scale = _named_transform(document, scale_name)
    restored = _replace_transform_constraints(
        document,
        {scale_name: replace(scale, bones=(base_name,))},
    )
    _camera_relative_topology(
        restored,
        profile=profile,
        prefix=prefix,
    )
    SpineValidator().validate_or_raise(restored)
    validate_spine41_setup_safety(restored)
    return restored


def adapt_camera_relative_two_axis_document_for_spine41_with_report(
    document: SpineDocument,
    *,
    profile: TwoAxisScaleRigProfile,
    prefix: str,
) -> Spine41TwoAxisDocumentAdaptation:
    """Add Spine 4.1 bridges while keeping Scale on the rigid object base."""

    temporary, base_name, _layer_name = _temporary_scale_document(
        document,
        profile=profile,
        prefix=prefix,
        include_layer=False,
    )
    adapted = adapt_two_axis_document_for_spine41_with_report(
        temporary,
        profile=profile,
        prefix=prefix,
    )
    restored_document = _restore_camera_relative_scale(
        adapted.document,
        profile=profile,
        prefix=prefix,
        base_name=base_name,
    )
    return replace(adapted, document=restored_document)


def adapt_camera_relative_two_axis_document_for_spine38_with_report(
    document: SpineDocument,
    *,
    profile: TwoAxisScaleRigProfile,
    prefix: str,
) -> Spine38TwoAxisDocumentAdaptation:
    """Build the Spine 3.8 X/IK/Depth/Y/object-Scale camera schedule."""

    temporary, base_name, _layer_name = _temporary_scale_document(
        document,
        profile=profile,
        prefix=prefix,
        include_layer=True,
    )
    temporary = _normalize_spine38_temporary_orders(
        temporary,
        profile=profile,
        prefix=prefix,
    )
    adapted = adapt_two_axis_document_for_spine38_with_report(
        temporary,
        profile=profile,
        prefix=prefix,
    )

    position_name = f"{prefix}_scale_spine38_position"
    position = _named_transform(adapted.document, position_name)
    rotation_x = _named_transform(
        adapted.document,
        profile.rotation_x_constraint(prefix),
    )
    depth = _named_transform(
        adapted.document,
        profile.scale_depth_constraint(prefix),
    )
    rotation_y = _named_transform(
        adapted.document,
        profile.rotation_y_constraint(prefix),
    )
    scale = _named_transform(
        adapted.document,
        profile.scale_constraint(prefix),
    )
    ik = _named_ik(
        adapted.document,
        profile.scale_ik_constraint(prefix),
    )
    base_order = position.order

    transform_replacements = {
        rotation_x.name: replace(rotation_x, order=base_order),
        depth.name: replace(depth, order=base_order + 2),
        rotation_y.name: replace(rotation_y, order=base_order + 3),
        scale.name: replace(
            scale,
            order=base_order + 4,
            bones=(base_name,),
        ),
    }
    restored = _replace_transform_constraints(
        adapted.document,
        transform_replacements,
        remove_names=frozenset({position_name}),
    )
    restored_ik = tuple(
        replace(item, order=base_order + 1) if item.name == ik.name else item
        for item in restored.ik
    )
    restored = replace(restored, ik=restored_ik)

    _camera_relative_topology(
        restored,
        profile=profile,
        prefix=prefix,
    )
    SpineValidator().validate_or_raise(restored)
    validate_spine41_setup_safety(restored)
    return replace(adapted, document=restored)


__all__ = [
    "adapt_camera_relative_two_axis_document_for_spine38_with_report",
    "adapt_camera_relative_two_axis_document_for_spine41_with_report",
]
