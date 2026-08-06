"""Build the exact legacy A1 IK and Transform constraint payload."""

from __future__ import annotations

from typing import Tuple

from .legacy_profile import LegacyRigProfile
from .legacy_rig_contracts import (
    LegacyRigBuildRequest,
    LegacyRigInfo,
)
from .legacy_rig_plan import LegacyRigBuildPlan
from .model import IKConstraint, TransformConstraint
from .rig_profiles import A1RigSetupPoseMode


def build_legacy_constraints(
    request: LegacyRigBuildRequest,
    profile: LegacyRigProfile,
    info: LegacyRigInfo,
) -> tuple[Tuple[IKConstraint, ...], Tuple[TransformConstraint, ...]]:
    """Compatibility-shaped constraint builder using resolved immutable metadata.

    Ordinary signed-axis model-space exports retain every historical setup offset.
    Camera-facing modes neutralize historical setup rotation. ``CAMERA_VIEW_NORMAL``
    also keeps the depth-scale transform full-rank; a dedicated inverse-setup child below
    each depth pair cancels only the authored setup translation before the projected
    vertex position is applied. This preserves exact setup shape and leaves live depth
    deformation available. ``CAMERA_DEPTH_SURFACE`` keeps its already-solved neutral
    depth-scale setup as before.
    """

    if not isinstance(request, LegacyRigBuildRequest):
        raise TypeError("request must be LegacyRigBuildRequest")
    if not isinstance(profile, LegacyRigProfile):
        raise TypeError("profile must be LegacyRigProfile")
    if not isinstance(info, LegacyRigInfo):
        raise TypeError("info must be LegacyRigInfo")

    neutral_camera_setup = request.setup_pose_mode in {
        A1RigSetupPoseMode.CAMERA_VIEW_NORMAL,
        A1RigSetupPoseMode.CAMERA_DEPTH_SURFACE,
    }
    neutral_depth_scale_setup = neutral_camera_setup
    prefix = request.prefix
    control_x, control_y, control_z = info.control_bone_names
    constraint_bone, _, constraint_rotate_ik, constraint_ik = (
        info.ik_chain_bone_names
    )

    ik = (
        IKConstraint(
            name=profile.scale_ik_constraint(prefix),
            order=3,
            bones=(constraint_bone,),
            target=constraint_ik,
            extras={"compress": True, "stretch": True},
        ),
    )

    transform = (
        TransformConstraint(
            name=profile.rotation_x_constraint(prefix),
            order=1,
            bones=info.sub_bone_scale_names + (info.base_bone_name,),
            target=control_x,
            extras={
                "rotation": 0.0 if neutral_camera_setup else 90,
                "local": True,
                "relative": True,
                "x": -(info.uniform_scale * 2.0),
                "y": -info.half_scale,
                "mixX": 0,
                "mixScaleX": 0,
                "mixShearY": 0,
            },
        ),
        TransformConstraint(
            name=profile.rotation_y_constraint(prefix),
            order=2,
            bones=(info.main_rotation_bone_name, constraint_rotate_ik),
            target=control_y,
            extras={
                "local": True,
                "relative": True,
                "x": info.uniform_scale,
                "scaleX": -1,
                "mixX": 0,
                "mixScaleX": 0,
                "mixShearY": 0,
            },
        ),
        TransformConstraint(
            name=profile.rotation_z_constraint(prefix),
            order=5,
            bones=info.sub_bone_names,
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
            order=4,
            bones=info.sub_bone_scale_names,
            target=constraint_bone,
            extras={
                "scaleX": 0.0 if neutral_depth_scale_setup else -1,
                "mixRotate": 0,
                "mixX": 0,
                "mixShearY": 0,
            },
        ),
        TransformConstraint(
            name=profile.scale_compensator_constraint(prefix),
            order=6,
            bones=tuple(reversed(info.sub_bone_scale_names)),
            target=info.base_bone_name,
            extras={
                "mixRotate": 0,
                "mixX": 0,
                "mixScaleX": 0,
                "mixScaleY": 0,
                "mixShearY": 0,
            },
        ),
    )
    return ik, transform


def build_legacy_rig_constraints(
    plan: LegacyRigBuildPlan,
) -> tuple[Tuple[IKConstraint, ...], Tuple[TransformConstraint, ...]]:
    if not isinstance(plan, LegacyRigBuildPlan):
        raise TypeError("plan must be LegacyRigBuildPlan")
    return build_legacy_constraints(plan.request, plan.profile, plan.info)


__all__ = [
    "build_legacy_constraints",
    "build_legacy_rig_constraints",
]