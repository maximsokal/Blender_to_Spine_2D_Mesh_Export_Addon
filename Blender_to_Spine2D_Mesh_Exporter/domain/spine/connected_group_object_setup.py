"""Normalize per-object control bones before connected A1 composition."""

from __future__ import annotations

from dataclasses import replace

from .connected_group_contracts import ConnectedObjectDocument
from .legacy_profile import LegacyRigProfile
from .rig_profiles import A1RigProfile, resolve_a1_rig_profile
from .two_axis_scale_profile import TwoAxisScaleRigProfile


def normalize_connected_object_control_space(
    item: ConnectedObjectDocument,
    profile: LegacyRigProfile,
) -> ConnectedObjectDocument:
    """Keep user controls in the same coordinate space as the connected object rig.

    The two-axis builder keeps ``<prefix>_scale`` under ``root`` for historical
    standalone compatibility and stores the object's main translation directly in the
    scale-control coordinates. Connected composition reparents ``<prefix>_main`` under
    a generated global layer, so leaving the scale control under ``root`` puts its
    target in a different transform space. The resulting relative transform constraint
    can move or deform the object and all scale icons overlap at the global origin.

    Before composition, convert that one control to ``<prefix>_main`` local space. The
    control's setup world position is preserved exactly, while later main-bone
    placement and global-layer transforms move the complete object rig and its controls
    together. No mesh, UV, attachment, constraint, or weighted-vertex data is changed.
    """

    if not isinstance(item, ConnectedObjectDocument):
        raise TypeError("item must be ConnectedObjectDocument")
    if not isinstance(profile, LegacyRigProfile):
        raise TypeError("profile must be LegacyRigProfile")

    profile_id = resolve_a1_rig_profile(profile.profile_id)
    if profile_id is not A1RigProfile.TWO_AXIS_ROTATION_SCALE:
        return item
    if not isinstance(profile, TwoAxisScaleRigProfile):
        raise TypeError(
            "TWO_AXIS_ROTATION_SCALE control normalization requires "
            "TwoAxisScaleRigProfile"
        )

    main_name = profile.main_bone(item.prefix)
    scale_name = profile.scale_control_bone(item.prefix)
    bone_by_name = {bone.name: bone for bone in item.document.bones}

    try:
        main_bone = bone_by_name[main_name]
        scale_bone = bone_by_name[scale_name]
    except KeyError as exc:
        raise ValueError(
            f"Connected object '{item.component_id}' is missing required two-axis "
            f"control bone: {exc.args[0]}"
        ) from exc

    if scale_bone.parent == main_name:
        return item
    if scale_bone.parent != profile.root_bone():
        raise ValueError(
            f"Connected object '{item.component_id}' scale control '{scale_name}' "
            f"must be parented to root or {main_name!r}, got {scale_bone.parent!r}"
        )

    local_x = round(float(scale_bone.x) - float(main_bone.x), 2)
    local_y = round(float(scale_bone.y) - float(main_bone.y), 2)
    updated_bones = tuple(
        replace(
            bone,
            parent=main_name,
            x=local_x,
            y=local_y,
        )
        if bone.name == scale_name
        else bone
        for bone in item.document.bones
    )
    return replace(item, document=replace(item.document, bones=updated_bones))


__all__ = ["normalize_connected_object_control_space"]
