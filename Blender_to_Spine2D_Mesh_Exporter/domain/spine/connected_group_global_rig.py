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
from .legacy_rig_contracts import LegacyRigBuildRequest, LegacyZGroup
from .model import Bone, IKConstraint, SpineDocument, TransformConstraint
from .rig_profiles import A1RigProfile, A1RigSetupPoseMode, resolve_a1_rig_profile
from .two_axis_scale_profile import TwoAxisScaleRigProfile
from .two_axis_scale_rig_assembly import build_two_axis_scale_rig
from .validator import SpineValidator


def _build_two_axis_connected_parts(
    layers: Tuple[ConnectedZLayer, ...],
    settings: ConnectedGroupSettings,
    profile: TwoAxisScaleRigProfile,
    uniform_scale: float,
) -> tuple[Tuple[Bone, ...], Tuple[IKConstraint, ...], Tuple[TransformConstraint, ...]]:
    """Build a neutral global two-axis wrapper and rename connected Z-layer bones.

    A connected global rig is a newly generated wrapper around already valid object
    rigs. It must therefore have a neutral setup pose. Reusing the per-object
    ``PRESERVE_COMPOSITION`` setup angles would immediately rotate the whole group a
    second time before the user touches a control.
    """

    if not isinstance(profile, TwoAxisScaleRigProfile):
        raise TypeError("profile must be TwoAxisScaleRigProfile")
    if not layers:
        raise ValueError("layers cannot be empty")

    z_groups = tuple(
        LegacyZGroup(
            z_value=float(layer.representative_relative_z),
            height_real_pixels=round(
                float(layer.representative_relative_z) * float(uniform_scale),
                2,
            ),
        )
        for layer in layers
    )
    rig = build_two_axis_scale_rig(
        LegacyRigBuildRequest(
            prefix=settings.group_prefix,
            texture_width=settings.texture_width,
            texture_height=settings.texture_height,
            z_groups=z_groups,
            main_position_pixels=(0.0, 0.0),
            scale_mode=settings.scale_mode,
            setup_pose_mode=A1RigSetupPoseMode.NORMALIZED_SINGLE,
        ),
        profile=profile,
    )
    if abs(float(rig.info.uniform_scale) - float(uniform_scale)) > 1e-9:
        raise ValueError(
            "connected two-axis rig scale differs from the resolved connected scale"
        )

    layer_by_z = {
        float(layer.representative_relative_z): layer for layer in layers
    }
    if len(layer_by_z) != len(layers):
        raise ValueError("connected layers cannot repeat representative_relative_z")

    name_map: dict[str, str] = {}
    for group in rig.info.z_groups:
        try:
            layer = layer_by_z[float(group.z_value)]
        except KeyError as exc:
            raise ValueError(
                "two-axis connected rig produced an unknown Z group: "
                f"{group.z_value}"
            ) from exc
        name_map[group.scale_bone_name] = layer.scale_bone_name
        name_map[group.bone_name] = layer.layer_bone_name

    if len(name_map) != len(layers) * 2:
        raise ValueError("two-axis connected Z-layer remap is incomplete")

    def remap_name(name: str | None) -> str | None:
        if name is None:
            return None
        return name_map.get(name, name)

    bones = tuple(
        replace(
            bone,
            name=remap_name(bone.name),
            parent=remap_name(bone.parent),
        )
        for bone in rig.bones
    )
    ik = tuple(
        replace(
            constraint,
            bones=tuple(remap_name(name) for name in constraint.bones),
            target=remap_name(constraint.target),
        )
        for constraint in rig.ik
    )
    transform = tuple(
        replace(
            constraint,
            bones=tuple(remap_name(name) for name in constraint.bones),
            target=remap_name(constraint.target),
        )
        for constraint in rig.transform
    )
    return bones, ik, transform


def _build_legacy_global_bones(
    layers: Tuple[ConnectedZLayer, ...],
    settings: ConnectedGroupSettings,
    profile: LegacyRigProfile,
    uniform_scale: float,
) -> Tuple[Bone, ...]:
    """Build the three-axis global connected hierarchy."""

    half_scale = uniform_scale / 2.0
    prefix = settings.group_prefix
    root = profile.root_bone()
    main = profile.main_bone(prefix)
    base = profile.base_bone(prefix)
    scale = profile.scale_rotate_x_bone(prefix)
    rotate = profile.rotate_x_bone(prefix)
    control_x, control_y, control_z = profile.control_bones(prefix)
    constraint, constraint_scale, constraint_rotate, constraint_ik = (
        profile.ik_chain_bones(prefix)
    )

    bones: list[Bone] = [
        Bone(name=root),
        Bone(name=main, parent=root, length=half_scale),
        Bone(name=base, parent=main),
        Bone(name=scale, parent=base, length=half_scale),
        Bone(
            name=control_x,
            parent=root,
            length=uniform_scale,
            x=uniform_scale,
            y=half_scale,
            color="ff0000ff",
        ),
        Bone(
            name=control_y,
            parent=root,
            length=uniform_scale,
            x=uniform_scale,
            color="00ff18ff",
        ),
        Bone(
            name=control_z,
            parent=root,
            length=uniform_scale,
            x=uniform_scale,
            y=-half_scale,
            color="002cffff",
        ),
        Bone(name=rotate, parent=scale, length=half_scale * 0.1),
        Bone(
            name=constraint,
            parent=base,
            length=half_scale,
            rotation=-90.0,
            color="abe323ff",
        ),
        Bone(name=constraint_scale, parent=base, rotation=-90.0),
        Bone(name=constraint_rotate, parent=constraint_scale),
        Bone(
            name=constraint_ik,
            parent=constraint_rotate,
            rotation=90.0,
            color="ff3f00ff",
            icon="ik",
        ),
    ]
    for layer in layers:
        depth_pixels = round(
            float(layer.representative_relative_z) * uniform_scale,
            2,
        )
        bones.extend(
            (
                Bone(
                    name=layer.scale_bone_name,
                    parent=rotate,
                    length=half_scale * 0.1,
                    rotation=90.0,
                    y=depth_pixels,
                    extras={"inherit": "onlyTranslation"},
                ),
                Bone(
                    name=layer.layer_bone_name,
                    parent=layer.scale_bone_name,
                    length=half_scale * 0.1,
                    rotation=-90.0,
                ),
            )
        )
    return tuple(bones)


def build_global_bones_document(
    source_skeleton: Mapping[str, object],
    layers: Tuple[ConnectedZLayer, ...],
    settings: ConnectedGroupSettings,
    profile: LegacyRigProfile,
    uniform_scale: float,
) -> SpineDocument:
    """Build and validate the bones-only global connected control component."""

    if not isinstance(profile, LegacyRigProfile):
        raise TypeError("profile must be LegacyRigProfile")
    profile_id = resolve_a1_rig_profile(profile.profile_id)
    if profile_id is A1RigProfile.TWO_AXIS_ROTATION_SCALE:
        if not isinstance(profile, TwoAxisScaleRigProfile):
            raise TypeError(
                "TWO_AXIS_ROTATION_SCALE global rig requires TwoAxisScaleRigProfile"
            )
        bones, _ik, _transform = _build_two_axis_connected_parts(
            layers,
            settings,
            profile,
            uniform_scale,
        )
    elif profile_id is A1RigProfile.THREE_AXIS_ROTATION:
        bones = _build_legacy_global_bones(
            layers,
            settings,
            profile,
            uniform_scale,
        )
    else:
        raise AssertionError(f"Unhandled connected rig profile: {profile_id}")

    document = SpineDocument(
        skeleton=deepcopy(dict(source_skeleton)),
        bones=bones,
        slots=(),
        skins=(),
    )
    SpineValidator().validate_or_raise(document)
    return document


def _build_legacy_global_constraints(
    objects: Tuple[ConnectedObjectDocument, ...],
    layers: Tuple[ConnectedZLayer, ...],
    schedule: ConnectedConstraintSchedule,
    settings: ConnectedGroupSettings,
    profile: LegacyRigProfile,
    uniform_scale: float,
) -> tuple[Tuple[IKConstraint, ...], Tuple[TransformConstraint, ...]]:
    """Build a setup-safe three-axis global connected constraint chain."""

    prefix = settings.group_prefix
    half_scale = uniform_scale / 2.0
    base = profile.base_bone(prefix)
    rotate = profile.rotate_x_bone(prefix)
    control_x, control_y, control_z = profile.control_bones(prefix)
    constraint, _, constraint_rotate, constraint_ik = profile.ik_chain_bones(prefix)
    scale_bones = tuple(layer.scale_bone_name for layer in layers)
    layer_bones = tuple(layer.layer_bone_name for layer in layers)

    ik = (
        IKConstraint(
            name=profile.scale_ik_constraint(prefix),
            order=schedule.global_scale_ik,
            bones=(constraint,),
            target=constraint_ik,
            extras={"compress": True, "stretch": True},
        ),
    )
    transform = (
        TransformConstraint(
            name=profile.rotation_x_constraint(prefix),
            order=schedule.global_rotation_x,
            bones=scale_bones + (base,),
            target=control_x,
            extras={
                "rotation": 90,
                "local": True,
                "relative": True,
                "x": -(uniform_scale * 2.0),
                "y": -half_scale,
                "scaleX": -1,
                "mixX": 0,
                "mixScaleX": 0,
                # Spine 4.2 defaults omitted scale mixes to 1. A historical
                # scaleY=-1 offset therefore collapsed the connected hierarchy to
                # zero height in setup pose. The global X controller must never own
                # Y scale, so disable that channel explicitly.
                "mixScaleY": 0,
                "mixShearY": 0,
            },
        ),
        TransformConstraint(
            name=profile.rotation_y_constraint(prefix),
            order=schedule.global_rotation_y,
            bones=(rotate, constraint_rotate),
            target=control_y,
            extras={
                "local": True,
                "relative": True,
                "x": uniform_scale,
                "scaleX": -1,
                "mixX": 0,
                "mixScaleX": 0,
                "mixShearY": 0,
            },
        ),
        TransformConstraint(
            name=profile.rotation_z_constraint(prefix),
            order=schedule.global_rotation_z,
            # Operate on generated wrapper layers, not object base bones. Local
            # object X/Y/Z constraints remain below the wrapper and cannot overwrite
            # or be overwritten by the group Z rotation.
            bones=layer_bones,
            target=control_z,
            extras={
                "local": True,
                "mixX": 0,
                "mixScaleX": 0,
                "mixShearY": 0,
            },
        ),
        TransformConstraint(
            name=profile.scale_constraint(prefix),
            order=schedule.global_scale,
            bones=scale_bones,
            target=constraint,
            extras={
                "scaleX": -1,
                "mixX": 0,
                "mixScaleX": 0,
                "mixShearY": 0,
            },
        ),
    )
    return ik, transform


def _build_two_axis_global_constraints(
    layers: Tuple[ConnectedZLayer, ...],
    schedule: ConnectedConstraintSchedule,
    settings: ConnectedGroupSettings,
    profile: TwoAxisScaleRigProfile,
    uniform_scale: float,
) -> tuple[Tuple[IKConstraint, ...], Tuple[TransformConstraint, ...]]:
    """Build global X/Y/Scale constraints from the validated five-phase rig."""

    _bones, ik, transform = _build_two_axis_connected_parts(
        layers,
        settings,
        profile,
        uniform_scale,
    )
    prefix = settings.group_prefix
    if schedule.global_scale_depth is None:
        raise ValueError("two-axis connected schedule has no global scale-depth order")
    order_by_name = {
        profile.rotation_x_constraint(prefix): schedule.global_rotation_x,
        profile.scale_ik_constraint(prefix): schedule.global_scale_ik,
        profile.scale_constraint(prefix): schedule.global_scale,
        profile.scale_depth_constraint(prefix): schedule.global_scale_depth,
        profile.rotation_y_constraint(prefix): schedule.global_rotation_y,
    }
    actual_names = {constraint.name for constraint in (*ik, *transform)}
    if actual_names != set(order_by_name):
        raise ValueError(
            "generated connected two-axis global constraints do not match the "
            f"profile contract: expected={tuple(order_by_name)}, "
            f"actual={tuple(sorted(actual_names))}"
        )
    return (
        tuple(
            replace(constraint, order=order_by_name[constraint.name])
            for constraint in ik
        ),
        tuple(
            replace(constraint, order=order_by_name[constraint.name])
            for constraint in transform
        ),
    )


def build_global_constraints(
    objects: Tuple[ConnectedObjectDocument, ...],
    layers: Tuple[ConnectedZLayer, ...],
    schedule: ConnectedConstraintSchedule,
    settings: ConnectedGroupSettings,
    profile: LegacyRigProfile,
    uniform_scale: float,
) -> tuple[Tuple[IKConstraint, ...], Tuple[TransformConstraint, ...]]:
    """Build global connected constraints for the selected rig profile."""

    if not isinstance(profile, LegacyRigProfile):
        raise TypeError("profile must be LegacyRigProfile")
    profile_id = resolve_a1_rig_profile(profile.profile_id)
    if resolve_a1_rig_profile(schedule.profile_id) is not profile_id:
        raise ValueError("connected constraint schedule profile does not match rig profile")

    if profile_id is A1RigProfile.THREE_AXIS_ROTATION:
        return _build_legacy_global_constraints(
            objects,
            layers,
            schedule,
            settings,
            profile,
            uniform_scale,
        )
    if profile_id is A1RigProfile.TWO_AXIS_ROTATION_SCALE:
        if not isinstance(profile, TwoAxisScaleRigProfile):
            raise TypeError(
                "TWO_AXIS_ROTATION_SCALE global constraints require "
                "TwoAxisScaleRigProfile"
            )
        return _build_two_axis_global_constraints(
            layers,
            schedule,
            settings,
            profile,
            uniform_scale,
        )
    raise AssertionError(f"Unhandled connected rig profile: {profile_id}")


__all__ = ["build_global_bones_document", "build_global_constraints"]
