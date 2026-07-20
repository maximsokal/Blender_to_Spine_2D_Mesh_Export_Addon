"""Build the global legacy ``all_objects`` bones and constraints."""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping, Tuple

from .connected_group_contracts import (
    ConnectedConstraintSchedule,
    ConnectedGroupSettings,
    ConnectedObjectDocument,
    ConnectedZLayer,
)
from .legacy_profile import LegacyRigProfile
from .model import Bone, IKConstraint, SpineDocument, TransformConstraint
from .validator import SpineValidator


def build_global_bones_document(
    source_skeleton: Mapping[str, object],
    layers: Tuple[ConnectedZLayer, ...],
    settings: ConnectedGroupSettings,
    profile: LegacyRigProfile,
    uniform_scale: float,
) -> SpineDocument:
    """Build and validate the bones-only global connected control component."""

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
        bones.extend(
            (
                Bone(
                    name=layer.scale_bone_name,
                    parent=rotate,
                    length=half_scale * 0.1,
                ),
                Bone(
                    name=layer.layer_bone_name,
                    parent=layer.scale_bone_name,
                    length=half_scale * 0.1,
                ),
            )
        )

    document = SpineDocument(
        skeleton=deepcopy(dict(source_skeleton)),
        bones=tuple(bones),
        slots=(),
        skins=(),
    )
    SpineValidator().validate_or_raise(document)
    return document


def build_global_constraints(
    objects: Tuple[ConnectedObjectDocument, ...],
    layers: Tuple[ConnectedZLayer, ...],
    schedule: ConnectedConstraintSchedule,
    settings: ConnectedGroupSettings,
    profile: LegacyRigProfile,
    uniform_scale: float,
) -> tuple[Tuple[IKConstraint, ...], Tuple[TransformConstraint, ...]]:
    """Build the four global connected controls in their scheduled positions."""

    prefix = settings.group_prefix
    half_scale = uniform_scale / 2.0
    base = profile.base_bone(prefix)
    rotate = profile.rotate_x_bone(prefix)
    control_x, control_y, control_z = profile.control_bones(prefix)
    constraint, _, constraint_rotate, constraint_ik = profile.ik_chain_bones(prefix)
    scale_bones = tuple(layer.scale_bone_name for layer in layers)
    object_base_bones = tuple(profile.base_bone(item.prefix) for item in objects)

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
                "scaleY": -1,
                "mixX": 0,
                "mixScaleX": 0,
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
            bones=object_base_bones,
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


__all__ = ["build_global_bones_document", "build_global_constraints"]
