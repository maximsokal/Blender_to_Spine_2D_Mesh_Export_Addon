"""Correct connected wrapper setup transforms before final Spine validation.

Connected composition wraps already valid per-object rigs. The generated global rig may
reuse the same internal mechanics as an object rig, but its setup pose must be an
identity transform. This module owns the final profile-aware correction so object
placement, control icons, UVs, attachments, and weighted vertices remain independent.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Mapping, Tuple

from .connected_group_contracts import (
    ConnectedObjectDocument,
    ConnectedObjectPlacement,
    ConnectedPlacementSpace,
    ConnectedZLayer,
)
from .legacy_profile import LegacyRigProfile
from .model import Bone, SpineDocument, TransformConstraint
from .rig_profiles import A1RigProfile, resolve_a1_rig_profile
from .two_axis_scale_profile import TwoAxisScaleRigProfile


def _number(value: float | None) -> float:
    return 0.0 if value is None else float(value)


def _bone_by_name(document: SpineDocument, name: str) -> Bone:
    matches = tuple(bone for bone in document.bones if bone.name == name)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one bone named {name!r}, found {len(matches)}"
        )
    return matches[0]


def _constraint_by_name(
    constraints: Tuple[TransformConstraint, ...],
    name: str,
) -> TransformConstraint:
    matches = tuple(item for item in constraints if item.name == name)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one transform constraint named {name!r}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _replace_extras(
    constraint: TransformConstraint,
    *,
    remove: Iterable[str] = (),
    update: Mapping[str, object] | None = None,
) -> TransformConstraint:
    extras = dict(constraint.extras)
    for key in remove:
        extras.pop(key, None)
    if update is not None:
        extras.update(update)
    return replace(constraint, extras=extras)


def _rotation_only_relative_local(
    constraint: TransformConstraint,
) -> TransformConstraint:
    """Make one wrapper rotation constraint an identity in setup pose.

    Spine 4.2 relative-local constraints add ``target local + offset`` to each
    constrained bone. A connected wrapper must therefore expose only the target's
    rotation delta. Per-object calibration offsets must not be applied a second time.
    """

    return _replace_extras(
        constraint,
        remove=("rotation", "x", "y", "scaleX", "scaleY", "shearY"),
        update={
            "local": True,
            "relative": True,
            "mixX": 0,
            "mixY": 0,
            "mixScaleX": 0,
            "mixScaleY": 0,
            "mixShearY": 0,
        },
    )


def _correct_global_constraints(
    document: SpineDocument,
    profile: LegacyRigProfile,
    group_prefix: str,
) -> SpineDocument:
    profile_id = resolve_a1_rig_profile(profile.profile_id)
    transform_by_name = {item.name: item for item in document.transform}

    rotation_x_name = profile.rotation_x_constraint(group_prefix)
    rotation_y_name = profile.rotation_y_constraint(group_prefix)
    transform_by_name[rotation_x_name] = _rotation_only_relative_local(
        _constraint_by_name(document.transform, rotation_x_name)
    )
    transform_by_name[rotation_y_name] = _rotation_only_relative_local(
        _constraint_by_name(document.transform, rotation_y_name)
    )

    if profile_id is A1RigProfile.TWO_AXIS_ROTATION_SCALE:
        if not isinstance(profile, TwoAxisScaleRigProfile):
            raise TypeError(
                "TWO_AXIS_ROTATION_SCALE setup correction requires "
                "TwoAxisScaleRigProfile"
            )
        scale_name = profile.scale_constraint(group_prefix)
        transform_by_name[scale_name] = _replace_extras(
            _constraint_by_name(document.transform, scale_name),
            update={"mixRotate": 0, "mixX": 0, "mixY": 0, "mixShearY": 0},
        )
    elif profile_id is A1RigProfile.THREE_AXIS_ROTATION:
        rotation_z_name = profile.rotation_z_constraint(group_prefix)
        transform_by_name[rotation_z_name] = _replace_extras(
            _constraint_by_name(document.transform, rotation_z_name),
            remove=("rotation", "x", "y", "scaleX", "scaleY", "shearY"),
            update={
                "local": True,
                "relative": True,
                "mixX": 0,
                "mixY": 0,
                "mixScaleX": 0,
                "mixScaleY": 0,
                "mixShearY": 0,
            },
        )

        scale_name = profile.scale_constraint(group_prefix)
        transform_by_name[scale_name] = _replace_extras(
            _constraint_by_name(document.transform, scale_name),
            update={
                "mixRotate": 0,
                "mixX": 0,
                "mixY": 0,
                "mixScaleX": 1,
                "mixScaleY": 1,
                "mixShearY": 0,
            },
        )
    else:
        raise AssertionError(f"Unhandled connected rig profile: {profile_id}")

    return replace(
        document,
        transform=tuple(transform_by_name[item.name] for item in document.transform),
    )


def _correct_object_main_placements(
    document: SpineDocument,
    objects: Tuple[ConnectedObjectDocument, ...],
    layers: Tuple[ConnectedZLayer, ...],
    placements: Tuple[ConnectedObjectPlacement, ...],
    uniform_scale: float,
) -> SpineDocument:
    """Compensate the visible XY placement already stored by each Z wrapper.

    ``<group>_<layer>_scale.y`` stores the layer's relative Blender Z in pixels. The
    child layer therefore already contributes that value to setup world Y. Object main
    bones must subtract the same setup offset, otherwise Z is added to visible Y a
    second time before any control is moved.
    """

    object_by_component = {item.component_id: item for item in objects}
    layer_by_index = {item.layer_index: item for item in layers}
    expected_by_main: dict[str, tuple[str, float]] = {}

    for placement in placements:
        try:
            source = object_by_component[placement.component_id]
            layer = layer_by_index[placement.layer_index]
        except KeyError as exc:
            raise ValueError(
                f"Connected placement references unknown component or layer: {exc.args[0]}"
            ) from exc

        source_main = _bone_by_name(source.document, placement.main_bone_name)
        layer_setup_y = round(
            float(layer.representative_relative_z) * float(uniform_scale),
            2,
        )
        visible_y = _number(source_main.y)
        if placement.placement_space is ConnectedPlacementSpace.ANCHOR_RELATIVE_WORLD:
            visible_y += float(placement.relative_y) * float(uniform_scale)
        elif placement.placement_space is not ConnectedPlacementSpace.PRESERVE_DOCUMENT:
            raise TypeError(
                f"Unsupported connected placement space: {placement.placement_space!r}"
            )

        expected_by_main[placement.main_bone_name] = (
            placement.parent_layer_bone_name,
            round(visible_y - layer_setup_y, 2),
        )

    found: set[str] = set()
    bones: list[Bone] = []
    for bone in document.bones:
        expected = expected_by_main.get(bone.name)
        if expected is None:
            bones.append(bone)
            continue
        parent_name, local_y = expected
        found.add(bone.name)
        bones.append(replace(bone, parent=parent_name, y=local_y))

    missing = set(expected_by_main) - found
    if missing:
        raise ValueError(
            "Connected setup correction cannot find object main bones: "
            f"{tuple(sorted(missing))}"
        )
    return replace(document, bones=tuple(bones))


def correct_connected_setup_pose(
    document: SpineDocument,
    objects: Tuple[ConnectedObjectDocument, ...],
    layers: Tuple[ConnectedZLayer, ...],
    placements: Tuple[ConnectedObjectPlacement, ...],
    profile: LegacyRigProfile,
    group_prefix: str,
    uniform_scale: float,
) -> SpineDocument:
    """Return a connected document whose global wrapper is setup-pose neutral."""

    if not isinstance(document, SpineDocument):
        raise TypeError("document must be SpineDocument")
    if not isinstance(objects, tuple) or not objects:
        raise ValueError("objects must be a non-empty tuple")
    if not isinstance(layers, tuple) or not layers:
        raise ValueError("layers must be a non-empty tuple")
    if not isinstance(placements, tuple) or not placements:
        raise ValueError("placements must be a non-empty tuple")
    if not isinstance(profile, LegacyRigProfile):
        raise TypeError("profile must be LegacyRigProfile")
    if not isinstance(group_prefix, str) or not group_prefix.strip():
        raise ValueError("group_prefix must be a non-empty string")
    if isinstance(uniform_scale, bool) or not isinstance(uniform_scale, (int, float)):
        raise TypeError("uniform_scale must be numeric")

    placed = _correct_object_main_placements(
        document,
        objects,
        layers,
        placements,
        float(uniform_scale),
    )
    return _correct_global_constraints(placed, profile, group_prefix)


__all__ = ["correct_connected_setup_pose"]
