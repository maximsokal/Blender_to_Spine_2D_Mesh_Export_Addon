"""Build global bones and constraints for connected A1 rig profiles."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Mapping, Tuple

from .connected_group_contracts import (
    ConnectedConstraintSchedule,
    ConnectedGroupSettings,
    ConnectedObjectDocument,
    ConnectedZLayer,
)
from .legacy_profile import LegacyRigProfile
from .legacy_rig_assembly import build_legacy_rig
from .legacy_rig_contracts import LegacyRigBuildRequest, LegacyZGroup
from .model import Bone, IKConstraint, SpineDocument, TransformConstraint
from .rig_profiles import A1RigProfile, A1RigSetupPoseMode, resolve_a1_rig_profile
from .two_axis_scale_profile import TwoAxisScaleRigProfile
from .two_axis_scale_rig_assembly import build_two_axis_scale_rig
from .validator import SpineValidator


def _connected_z_groups(
    layers: Tuple[ConnectedZLayer, ...],
    uniform_scale: float,
) -> Tuple[LegacyZGroup, ...]:
    if not isinstance(layers, tuple) or not layers:
        raise ValueError("layers must be a non-empty tuple")
    return tuple(
        LegacyZGroup(
            z_value=float(layer.representative_relative_z),
            height_real_pixels=round(
                float(layer.representative_relative_z) * float(uniform_scale),
                2,
            ),
        )
        for layer in layers
    )


def _remap_connected_layers(
    bones: Tuple[Bone, ...],
    ik: Tuple[IKConstraint, ...],
    transform: Tuple[TransformConstraint, ...],
    rig_z_groups,
    layers: Tuple[ConnectedZLayer, ...],
) -> tuple[Tuple[Bone, ...], Tuple[IKConstraint, ...], Tuple[TransformConstraint, ...]]:
    """Rename builder-owned Z groups to stable connected layer identities."""

    layer_by_z = {
        float(layer.representative_relative_z): layer for layer in layers
    }
    if len(layer_by_z) != len(layers):
        raise ValueError("connected layers cannot repeat representative_relative_z")

    name_map: dict[str, str] = {}
    for group in rig_z_groups:
        try:
            layer = layer_by_z[float(group.z_value)]
        except KeyError as exc:
            raise ValueError(
                "connected rig produced an unknown Z group: "
                f"{group.z_value}"
            ) from exc
        name_map[group.scale_bone_name] = layer.scale_bone_name
        name_map[group.bone_name] = layer.layer_bone_name

    if len(name_map) != len(layers) * 2:
        raise ValueError("connected Z-layer remap is incomplete")

    def remap_name(name: str | None) -> str | None:
        if name is None:
            return None
        return name_map.get(name, name)

    return (
        tuple(
            replace(
                bone,
                name=remap_name(bone.name),
                parent=remap_name(bone.parent),
            )
            for bone in bones
        ),
        tuple(
            replace(
                constraint,
                bones=tuple(remap_name(name) for name in constraint.bones),
                target=remap_name(constraint.target),
            )
            for constraint in ik
        ),
        tuple(
            replace(
                constraint,
                bones=tuple(remap_name(name) for name in constraint.bones),
                target=remap_name(constraint.target),
            )
            for constraint in transform
        ),
    )


def _build_connected_profile_parts(
    layers: Tuple[ConnectedZLayer, ...],
    settings: ConnectedGroupSettings,
    profile: LegacyRigProfile,
    uniform_scale: float,
) -> tuple[Tuple[Bone, ...], Tuple[IKConstraint, ...], Tuple[TransformConstraint, ...]]:
    """Build one complete global wrapper through the selected validated rig owner."""

    if not isinstance(settings, ConnectedGroupSettings):
        raise TypeError("settings must be ConnectedGroupSettings")
    if not isinstance(profile, LegacyRigProfile):
        raise TypeError("profile must be LegacyRigProfile")

    request = LegacyRigBuildRequest(
        prefix=settings.group_prefix,
        texture_width=settings.texture_width,
        texture_height=settings.texture_height,
        z_groups=_connected_z_groups(layers, uniform_scale),
        main_position_pixels=(0.0, 0.0),
        scale_mode=settings.scale_mode,
        setup_pose_mode=A1RigSetupPoseMode.NORMALIZED_SINGLE,
    )
    profile_id = resolve_a1_rig_profile(profile.profile_id)
    if profile_id is A1RigProfile.TWO_AXIS_ROTATION_SCALE:
        if not isinstance(profile, TwoAxisScaleRigProfile):
            raise TypeError(
                "TWO_AXIS_ROTATION_SCALE global rig requires TwoAxisScaleRigProfile"
            )
        rig = build_two_axis_scale_rig(request, profile=profile)
    elif profile_id is A1RigProfile.THREE_AXIS_ROTATION:
        rig = build_legacy_rig(request, profile=profile)
    else:
        raise AssertionError(f"Unhandled connected rig profile: {profile_id}")

    if abs(float(rig.info.uniform_scale) - float(uniform_scale)) > 1e-9:
        raise ValueError(
            "connected rig scale differs from the resolved connected scale"
        )
    return _remap_connected_layers(
        rig.bones,
        rig.ik,
        rig.transform,
        rig.info.z_groups,
        layers,
    )


def build_global_bones_document(
    source_skeleton: Mapping[str, object],
    layers: Tuple[ConnectedZLayer, ...],
    settings: ConnectedGroupSettings,
    profile: LegacyRigProfile,
    uniform_scale: float,
) -> SpineDocument:
    """Build and validate the bones-only global connected control component."""

    bones, _ik, _transform = _build_connected_profile_parts(
        layers,
        settings,
        profile,
        uniform_scale,
    )
    document = SpineDocument(
        skeleton=deepcopy(dict(source_skeleton)),
        bones=bones,
        slots=(),
        skins=(),
    )
    SpineValidator().validate_or_raise(document)
    return document


def _global_order_by_name(
    schedule: ConnectedConstraintSchedule,
    settings: ConnectedGroupSettings,
    profile: LegacyRigProfile,
) -> dict[str, int]:
    prefix = settings.group_prefix
    profile_id = resolve_a1_rig_profile(profile.profile_id)

    if profile_id is A1RigProfile.TWO_AXIS_ROTATION_SCALE:
        if not isinstance(profile, TwoAxisScaleRigProfile):
            raise TypeError(
                "TWO_AXIS_ROTATION_SCALE global constraints require "
                "TwoAxisScaleRigProfile"
            )
        if schedule.global_scale_depth is None:
            raise ValueError("two-axis connected schedule has no global scale-depth order")
        return {
            profile.rotation_x_constraint(prefix): schedule.global_rotation_x,
            profile.scale_ik_constraint(prefix): schedule.global_scale_ik,
            profile.scale_constraint(prefix): schedule.global_scale,
            profile.scale_depth_constraint(prefix): schedule.global_scale_depth,
            profile.rotation_y_constraint(prefix): schedule.global_rotation_y,
        }

    if profile_id is A1RigProfile.THREE_AXIS_ROTATION:
        if schedule.global_rotation_z is None:
            raise ValueError("three-axis connected schedule has no global Rotation Z order")
        return {
            profile.rotation_x_constraint(prefix): schedule.global_rotation_x,
            profile.rotation_y_constraint(prefix): schedule.global_rotation_y,
            profile.rotation_z_constraint(prefix): schedule.global_rotation_z,
            profile.scale_ik_constraint(prefix): schedule.global_scale_ik,
            profile.scale_constraint(prefix): schedule.global_scale,
        }

    raise AssertionError(f"Unhandled connected rig profile: {profile_id}")


def build_global_constraints(
    objects: Tuple[ConnectedObjectDocument, ...],
    layers: Tuple[ConnectedZLayer, ...],
    schedule: ConnectedConstraintSchedule,
    settings: ConnectedGroupSettings,
    profile: LegacyRigProfile,
    uniform_scale: float,
) -> tuple[Tuple[IKConstraint, ...], Tuple[TransformConstraint, ...]]:
    """Build global constraints and assign the profile-specific connected schedule."""

    if not isinstance(objects, tuple) or not objects:
        raise ValueError("objects must be a non-empty tuple")
    if not isinstance(profile, LegacyRigProfile):
        raise TypeError("profile must be LegacyRigProfile")
    if resolve_a1_rig_profile(schedule.profile_id) is not resolve_a1_rig_profile(
        profile.profile_id
    ):
        raise ValueError("connected constraint schedule profile does not match rig profile")

    _bones, ik, transform = _build_connected_profile_parts(
        layers,
        settings,
        profile,
        uniform_scale,
    )
    order_by_name = _global_order_by_name(schedule, settings, profile)
    generated = {constraint.name for constraint in (*ik, *transform)}
    missing = set(order_by_name) - generated
    if missing:
        raise ValueError(
            "generated connected global constraints are incomplete: "
            f"{tuple(sorted(missing))}"
        )

    selected_ik = tuple(
        replace(constraint, order=order_by_name[constraint.name])
        for constraint in ik
        if constraint.name in order_by_name
    )
    selected_transform = tuple(
        replace(constraint, order=order_by_name[constraint.name])
        for constraint in transform
        if constraint.name in order_by_name
    )
    if {item.name for item in (*selected_ik, *selected_transform)} != set(order_by_name):
        raise ValueError("connected global constraint selection is incomplete")
    return selected_ik, selected_transform


__all__ = ["build_global_bones_document", "build_global_constraints"]
