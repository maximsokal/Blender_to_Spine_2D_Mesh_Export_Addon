"""Build exact legacy A1 bone groups from an immutable rig plan."""

from __future__ import annotations

from typing import Tuple

from .legacy_profile import LegacyRigProfile
from .legacy_rig_contracts import (
    LegacyRigBuildRequest,
    LegacyZGroupBuildInfo,
)
from .legacy_rig_plan import (
    LegacyRigBuildPlan,
    build_legacy_z_group_metadata,
)
from .legacy_rig_scale import require_finite_derived
from .model import Bone
from .rig_profiles import A1RigSetupPoseMode


def build_core_bones(plan: LegacyRigBuildPlan) -> Tuple[Bone, ...]:
    if not isinstance(plan, LegacyRigBuildPlan):
        raise TypeError("plan must be LegacyRigBuildPlan")
    return (
        Bone(name=plan.root_bone_name),
        Bone(
            name=plan.main_bone_name,
            parent=plan.root_bone_name,
            x=plan.main_x,
            y=plan.main_y,
        ),
        Bone(name=plan.base_bone_name, parent=plan.main_bone_name),
        Bone(
            name=plan.scale_bone_name,
            parent=plan.base_bone_name,
            length=plan.half_scale,
            y=-0.5,
            scale_x=0.0,
        ),
        Bone(
            name=plan.main_rotation_bone_name,
            parent=plan.scale_bone_name,
            color="ff0000ff",
        ),
    )


def build_legacy_z_group_bones(
    z_groups: Tuple[LegacyZGroupBuildInfo, ...],
    *,
    parent_bone_name: str,
    half_scale: float,
) -> Tuple[Bone, ...]:
    if not isinstance(z_groups, tuple) or not z_groups:
        raise ValueError("z_groups must be a non-empty tuple")
    if not all(isinstance(item, LegacyZGroupBuildInfo) for item in z_groups):
        raise TypeError("z_groups must contain LegacyZGroupBuildInfo values")
    if not isinstance(parent_bone_name, str) or not parent_bone_name.strip():
        raise ValueError("parent_bone_name must be a non-empty string")
    resolved_half_scale = require_finite_derived(half_scale, "half_scale")
    if resolved_half_scale <= 0.0:
        raise ValueError("half_scale must be positive")

    bones: list[Bone] = []
    for group in z_groups:
        bones.extend(
            (
                Bone(
                    name=group.scale_bone_name,
                    parent=parent_bone_name,
                    length=resolved_half_scale,
                    rotation=90.0,
                    y=group.y_offset_pixels,
                    color="abe323ff",
                    extras={"inherit": "onlyTranslation"},
                ),
                Bone(
                    name=group.bone_name,
                    parent=group.scale_bone_name,
                    rotation=-90.0,
                ),
            )
        )
    return tuple(bones)


def build_camera_view_setup_compensation_bones(
    plan: LegacyRigBuildPlan,
) -> Tuple[Bone, ...]:
    """Build inverse setup translations for Active Camera Object Root.

    Each ordinary depth pair resolves to a pure ``(0, depth)`` translation while camera
    setup rotations and depth scale remain neutral. Parenting vertex bones through a
    child at ``(0, -depth)`` therefore yields an exact identity setup transform without
    discarding the later depth-pair deformation driven by the X/Y controls.
    """

    if not isinstance(plan, LegacyRigBuildPlan):
        raise TypeError("plan must be LegacyRigBuildPlan")
    if plan.request.setup_pose_mode is not A1RigSetupPoseMode.CAMERA_VIEW_NORMAL:
        return ()

    return tuple(
        Bone(
            name=plan.profile.z_camera_setup_bone(plan.prefix, group.index),
            parent=group.bone_name,
            y=require_finite_derived(
                round(-float(group.y_offset_pixels), 2),
                f"camera_setup[{group.index}].y",
            ),
        )
        for group in plan.z_groups
    )


def build_z_group_bones_for_request(
    request: LegacyRigBuildRequest,
    profile: LegacyRigProfile,
    *,
    parent_bone_name: str,
    uniform_scale: float,
    half_scale: float,
) -> tuple[Tuple[Bone, ...], Tuple[LegacyZGroupBuildInfo, ...]]:
    """Compatibility entrypoint matching the historical private helper."""

    metadata = build_legacy_z_group_metadata(request, profile, uniform_scale)
    return (
        build_legacy_z_group_bones(
            metadata,
            parent_bone_name=parent_bone_name,
            half_scale=half_scale,
        ),
        metadata,
    )


def build_control_bones(plan: LegacyRigBuildPlan) -> Tuple[Bone, ...]:
    if not isinstance(plan, LegacyRigBuildPlan):
        raise TypeError("plan must be LegacyRigBuildPlan")
    control_x, control_y, control_z = plan.control_bone_names
    return (
        Bone(
            name=control_x,
            parent=plan.main_bone_name,
            length=plan.half_scale,
            x=plan.uniform_scale,
            y=plan.half_scale,
            color="ff0000ff",
        ),
        Bone(
            name=control_y,
            parent=plan.main_bone_name,
            length=plan.half_scale,
            x=plan.uniform_scale,
            color="00ff18ff",
        ),
        Bone(
            name=control_z,
            parent=plan.main_bone_name,
            length=plan.half_scale,
            x=plan.uniform_scale,
            y=-plan.half_scale,
            color="002cffff",
        ),
    )


def build_ik_chain_bones(plan: LegacyRigBuildPlan) -> Tuple[Bone, ...]:
    if not isinstance(plan, LegacyRigBuildPlan):
        raise TypeError("plan must be LegacyRigBuildPlan")
    constraint_bone, constraint_scale_ik, constraint_rotate_ik, constraint_ik = (
        plan.ik_chain_bone_names
    )
    return (
        Bone(
            name=constraint_bone,
            parent=plan.base_bone_name,
            length=plan.half_scale,
            rotation=90.0,
            y=-0.5,
            color="abe323ff",
        ),
        Bone(
            name=constraint_scale_ik,
            parent=plan.base_bone_name,
            y=plan.half_scale - 0.5,
            scale_x=0.0,
        ),
        Bone(
            name=constraint_rotate_ik,
            parent=constraint_scale_ik,
            x=-plan.half_scale,
        ),
        Bone(
            name=constraint_ik,
            parent=constraint_rotate_ik,
            rotation=90.0,
            x=plan.half_scale,
            color="ff3f00ff",
            icon="ik",
        ),
    )


def build_legacy_rig_bones(plan: LegacyRigBuildPlan) -> Tuple[Bone, ...]:
    """Build the exact deterministic legacy hierarchy for the selected setup mode."""

    return (
        *build_core_bones(plan),
        *build_legacy_z_group_bones(
            plan.z_groups,
            parent_bone_name=plan.main_rotation_bone_name,
            half_scale=plan.half_scale,
        ),
        *build_camera_view_setup_compensation_bones(plan),
        *build_control_bones(plan),
        *build_ik_chain_bones(plan),
    )


__all__ = [
    "build_camera_view_setup_compensation_bones",
    "build_control_bones",
    "build_core_bones",
    "build_ik_chain_bones",
    "build_legacy_rig_bones",
    "build_legacy_z_group_bones",
    "build_z_group_bones_for_request",
]