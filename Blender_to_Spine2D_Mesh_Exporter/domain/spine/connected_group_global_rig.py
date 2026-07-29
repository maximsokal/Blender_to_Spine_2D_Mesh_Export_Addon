"""Build profile-aware global bones and constraints for connected A1 rigs.

The three-axis path reproduces the dedicated connected wrapper from the historical
``main`` branch. It must not be replaced by a normal per-object rig: the wrapper has a
different hierarchy, explicit zero-valued helper fields, exact constraint targets, and a
Z-layer-based order model.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Mapping, Tuple

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


def _require_inputs(
    layers: Tuple[ConnectedZLayer, ...],
    settings: ConnectedGroupSettings,
    profile: LegacyRigProfile,
    uniform_scale: float,
) -> None:
    if not isinstance(layers, tuple) or not layers:
        raise ValueError("layers must be a non-empty tuple")
    if not all(isinstance(item, ConnectedZLayer) for item in layers):
        raise TypeError("layers must contain ConnectedZLayer values")
    if not isinstance(settings, ConnectedGroupSettings):
        raise TypeError("settings must be ConnectedGroupSettings")
    if not isinstance(profile, LegacyRigProfile):
        raise TypeError("profile must be LegacyRigProfile")
    if isinstance(uniform_scale, bool) or not isinstance(uniform_scale, (int, float)):
        raise TypeError("uniform_scale must be numeric")
    if float(uniform_scale) <= 0.0:
        raise ValueError("uniform_scale must be positive")


def _legacy_bone(
    name: str,
    parent: str | None,
    *,
    length: float = 0.0,
    x: float = 0.0,
    y: float = 0.0,
    rotation: float | None = None,
    color: str | None = None,
    icon: str | None = None,
) -> Bone:
    """Typed equivalent of historical ``main._mk_bone``.

    The old helper always serialized ``length``, ``x``, and ``y`` for generated bones,
    including zero values. Keeping those fields makes review against Legacy JSON exact
    instead of relying on Spine defaults.
    """

    return Bone(
        name=name,
        parent=parent,
        length=float(length),
        x=round(float(x), 2),
        y=round(float(y), 2),
        rotation=rotation,
        color=color,
        icon=icon,
    )


def _legacy_connected_bones(
    layers: Tuple[ConnectedZLayer, ...],
    settings: ConnectedGroupSettings,
    profile: LegacyRigProfile,
    uniform_scale: float,
) -> Tuple[Bone, ...]:
    """Reproduce ``main._build_global_rig`` as immutable Spine model values."""

    prefix = settings.group_prefix
    scale = float(uniform_scale)
    half = scale / 2.0
    base = profile.base_bone(prefix)
    scale_rotate_x = profile.scale_rotate_x_bone(prefix)
    rotate_x = profile.rotate_x_bone(prefix)
    constraint_bone = profile.rotate_x_constraint_bone(prefix)
    constraint_scale_ik = profile.rotate_x_constraint_scale_ik_bone(prefix)
    constraint_rotate_ik = profile.rotate_x_constraint_rotate_ik_bone(prefix)
    constraint_ik = profile.rotate_x_constraint_ik_bone(prefix)

    generated = [
        _legacy_bone(
            profile.main_bone(prefix),
            profile.root_bone(),
            length=half,
        ),
        _legacy_bone(base, profile.main_bone(prefix)),
        _legacy_bone(scale_rotate_x, base, length=half),
        _legacy_bone(
            profile.control_x_bone(prefix),
            profile.root_bone(),
            length=scale,
            x=scale,
            y=half,
            color="ff0000ff",
        ),
        _legacy_bone(
            profile.control_y_bone(prefix),
            profile.root_bone(),
            length=scale,
            x=scale,
            color="00ff18ff",
        ),
        _legacy_bone(
            profile.control_z_bone(prefix),
            profile.root_bone(),
            length=scale,
            x=scale,
            y=-half,
            color="002cffff",
        ),
        _legacy_bone(rotate_x, scale_rotate_x, length=half * 0.1),
        _legacy_bone(
            constraint_bone,
            base,
            length=half,
            rotation=-90.0,
            color="abe323ff",
        ),
        _legacy_bone(
            constraint_scale_ik,
            base,
            rotation=-90.0,
        ),
        _legacy_bone(constraint_rotate_ik, constraint_scale_ik),
        _legacy_bone(
            constraint_ik,
            constraint_rotate_ik,
            rotation=90.0,
            color="ff3f00ff",
            icon="ik",
        ),
    ]
    for layer in layers:
        generated.extend(
            (
                _legacy_bone(
                    layer.scale_bone_name,
                    rotate_x,
                    length=half * 0.1,
                ),
                _legacy_bone(
                    layer.layer_bone_name,
                    layer.scale_bone_name,
                    length=half * 0.1,
                ),
            )
        )

    return (Bone(name=profile.root_bone()), *generated)


def _connected_z_groups(
    layers: Tuple[ConnectedZLayer, ...],
    uniform_scale: float,
) -> Tuple[LegacyZGroup, ...]:
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


def _remap_two_axis_layers(
    bones: Tuple[Bone, ...],
    rig_z_groups,
    layers: Tuple[ConnectedZLayer, ...],
) -> Tuple[Bone, ...]:
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
                f"two-axis connected rig produced unknown Z value {group.z_value}"
            ) from exc
        name_map[group.scale_bone_name] = layer.scale_bone_name
        name_map[group.bone_name] = layer.layer_bone_name

    if len(name_map) != len(layers) * 2:
        raise ValueError("two-axis connected layer remap is incomplete")

    return tuple(
        replace(
            bone,
            name=name_map.get(bone.name, bone.name),
            parent=(
                None
                if bone.parent is None
                else name_map.get(bone.parent, bone.parent)
            ),
        )
        for bone in bones
    )


def _build_two_axis_source_rig(
    layers: Tuple[ConnectedZLayer, ...],
    settings: ConnectedGroupSettings,
    profile: TwoAxisScaleRigProfile,
    uniform_scale: float,
):
    rig = build_two_axis_scale_rig(
        LegacyRigBuildRequest(
            prefix=settings.group_prefix,
            texture_width=settings.texture_width,
            texture_height=settings.texture_height,
            z_groups=_connected_z_groups(layers, uniform_scale),
            main_position_pixels=(0.0, 0.0),
            scale_mode=settings.scale_mode,
            setup_pose_mode=A1RigSetupPoseMode.NORMALIZED_SINGLE,
        ),
        profile=profile,
    )
    if abs(float(rig.info.uniform_scale) - float(uniform_scale)) > 1e-9:
        raise ValueError("connected two-axis rig scale differs from group scale")
    return rig


def build_global_bones_document(
    source_skeleton: Mapping[str, object],
    layers: Tuple[ConnectedZLayer, ...],
    settings: ConnectedGroupSettings,
    profile: LegacyRigProfile,
    uniform_scale: float,
) -> SpineDocument:
    """Build the bones-only global connected control component."""

    _require_inputs(layers, settings, profile, uniform_scale)
    profile_id = resolve_a1_rig_profile(profile.profile_id)
    if profile_id is A1RigProfile.THREE_AXIS_ROTATION:
        bones = _legacy_connected_bones(layers, settings, profile, uniform_scale)
    elif profile_id is A1RigProfile.TWO_AXIS_ROTATION_SCALE:
        if not isinstance(profile, TwoAxisScaleRigProfile):
            raise TypeError(
                "TWO_AXIS_ROTATION_SCALE global rig requires TwoAxisScaleRigProfile"
            )
        rig = _build_two_axis_source_rig(layers, settings, profile, uniform_scale)
        bones = _remap_two_axis_layers(rig.bones, rig.info.z_groups, layers)
    else:
        raise AssertionError(f"Unhandled connected rig profile: {profile_id}")

    document = SpineDocument(
        skeleton=dict(source_skeleton),
        bones=bones,
        slots=(),
        skins=(),
    )
    SpineValidator().validate_or_raise(document)
    return document


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


def _legacy_global_constraints(
    objects: Tuple[ConnectedObjectDocument, ...],
    layers: Tuple[ConnectedZLayer, ...],
    schedule: ConnectedConstraintSchedule,
    settings: ConnectedGroupSettings,
    profile: LegacyRigProfile,
    uniform_scale: float,
) -> tuple[Tuple[IKConstraint, ...], Tuple[TransformConstraint, ...]]:
    """Reproduce ``main._build_global_constraints`` without semantic changes."""

    if schedule.global_rotation_z is None:
        raise ValueError("legacy connected schedule has no global Rotation Z order")

    prefix = settings.group_prefix
    scale = float(uniform_scale)
    half = scale / 2.0
    scale_bones = tuple(layer.scale_bone_name for layer in layers)
    object_base_bones = tuple(profile.base_bone(item.prefix) for item in objects)

    ik = (
        IKConstraint(
            name=profile.scale_ik_constraint(prefix),
            order=schedule.global_scale_ik,
            bones=(profile.rotate_x_constraint_bone(prefix),),
            target=profile.rotate_x_constraint_ik_bone(prefix),
            extras={"compress": True, "stretch": True},
        ),
    )
    transform = (
        TransformConstraint(
            name=profile.rotation_x_constraint(prefix),
            order=schedule.global_rotation_x,
            bones=(*scale_bones, profile.base_bone(prefix)),
            target=profile.control_x_bone(prefix),
            extras={
                "rotation": 90,
                "local": True,
                "relative": True,
                "x": -(scale * 2.0),
                "y": -half,
                "scaleX": -1,
                "scaleY": -1,
                "mixX": 0,
                "mixScaleX": 0,
                "mixShearY": 0,
            },
        ),
        TransformConstraint(
            name=profile.rotation_y_constraint(prefix),
            order=schedule.global_rotation_y,
            bones=(
                profile.rotate_x_bone(prefix),
                profile.rotate_x_constraint_rotate_ik_bone(prefix),
            ),
            target=profile.control_y_bone(prefix),
            extras={
                "local": True,
                "relative": True,
                "x": scale,
                "scaleX": -1,
                "mixX": 0,
                "mixScaleX": 0,
                "mixShearY": 0,
            },
        ),
        TransformConstraint(
            name=profile.rotation_z_constraint(prefix),
            order=schedule.global_rotation_z,
            bones=object_base_bones,
            target=profile.control_z_bone(prefix),
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
            target=profile.rotate_x_constraint_bone(prefix),
            extras={
                "scaleX": -1,
                "mixX": 0,
                "mixScaleX": 0,
                "mixShearY": 0,
            },
        ),
    )
    return ik, transform


def _constraint_by_name(constraints, name: str):
    matches = tuple(item for item in constraints if item.name == name)
    if len(matches) != 1:
        raise ValueError(
            f"Expected one generated constraint {name!r}, found {len(matches)}"
        )
    return matches[0]


def _two_axis_global_constraints(
    layers: Tuple[ConnectedZLayer, ...],
    schedule: ConnectedConstraintSchedule,
    settings: ConnectedGroupSettings,
    profile: TwoAxisScaleRigProfile,
    uniform_scale: float,
) -> tuple[Tuple[IKConstraint, ...], Tuple[TransformConstraint, ...]]:
    if schedule.global_scale_depth is None:
        raise ValueError("two-axis connected schedule has no global depth order")

    prefix = settings.group_prefix
    rig = _build_two_axis_source_rig(layers, settings, profile, uniform_scale)
    generated = (*rig.ik, *rig.transform)
    layer_bones = tuple(layer.layer_bone_name for layer in layers)
    scale_bones = tuple(layer.scale_bone_name for layer in layers)

    source_ik = _constraint_by_name(generated, profile.scale_ik_constraint(prefix))
    source_x = _constraint_by_name(generated, profile.rotation_x_constraint(prefix))
    source_scale = _constraint_by_name(generated, profile.scale_constraint(prefix))
    source_depth = _constraint_by_name(
        generated,
        profile.scale_depth_constraint(prefix),
    )
    source_y = _constraint_by_name(generated, profile.rotation_y_constraint(prefix))

    ik = (
        replace(
            source_ik,
            order=schedule.global_scale_ik,
            bones=(profile.rotate_x_constraint_bone(prefix),),
            target=profile.rotate_x_constraint_ik_bone(prefix),
        ),
    )
    transform = (
        _replace_extras(
            replace(
                source_x,
                order=schedule.global_rotation_x,
                bones=(
                    profile.rotate_x_constraint_rotate_ik_bone(prefix),
                    profile.rotate_x_bone(prefix),
                ),
                target=profile.control_x_bone(prefix),
            ),
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
        ),
        _replace_extras(
            replace(
                source_scale,
                order=schedule.global_scale,
                bones=(profile.rotate_x_bone(prefix), *layer_bones),
                target=profile.scale_control_bone(prefix),
            ),
            update={
                "relative": True,
                "mixRotate": 0,
                "mixX": 0,
                "mixY": 0,
                "mixShearY": 0,
            },
        ),
        replace(
            source_depth,
            order=schedule.global_scale_depth,
            bones=scale_bones,
            target=profile.rotate_x_constraint_bone(prefix),
        ),
        _replace_extras(
            replace(
                source_y,
                order=schedule.global_rotation_y,
                bones=layer_bones,
                target=profile.control_y_bone(prefix),
            ),
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
        ),
    )
    return ik, transform


def build_global_constraints(
    objects: Tuple[ConnectedObjectDocument, ...],
    layers: Tuple[ConnectedZLayer, ...],
    schedule: ConnectedConstraintSchedule,
    settings: ConnectedGroupSettings,
    profile: LegacyRigProfile,
    uniform_scale: float,
) -> tuple[Tuple[IKConstraint, ...], Tuple[TransformConstraint, ...]]:
    """Build explicit connected constraints for one selected profile."""

    _require_inputs(layers, settings, profile, uniform_scale)
    if not isinstance(objects, tuple) or not objects:
        raise ValueError("objects must be a non-empty tuple")
    if not isinstance(schedule, ConnectedConstraintSchedule):
        raise TypeError("schedule must be ConnectedConstraintSchedule")
    if resolve_a1_rig_profile(schedule.profile_id) is not resolve_a1_rig_profile(
        profile.profile_id
    ):
        raise ValueError("connected schedule profile does not match rig profile")

    profile_id = resolve_a1_rig_profile(profile.profile_id)
    if profile_id is A1RigProfile.THREE_AXIS_ROTATION:
        return _legacy_global_constraints(
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
        return _two_axis_global_constraints(
            layers,
            schedule,
            settings,
            profile,
            uniform_scale,
        )
    raise AssertionError(f"Unhandled connected rig profile: {profile_id}")


__all__ = ["build_global_bones_document", "build_global_constraints"]
