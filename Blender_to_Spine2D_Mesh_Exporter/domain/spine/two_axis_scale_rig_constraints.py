"""Build IK and Transform constraints for the two-axis scale rig."""

from __future__ import annotations

from typing import Tuple

from .legacy_rig_plan import LegacyRigBuildPlan
from .model import IKConstraint, TransformConstraint
from .rig_profiles import A1RigSetupPoseMode
from .two_axis_scale_profile import TwoAxisScaleRigProfile
from .two_axis_scale_rig_contracts import TwoAxisScaleRigLayout


def build_two_axis_scale_constraints(
    plan: LegacyRigBuildPlan,
    layout: TwoAxisScaleRigLayout,
) -> tuple[Tuple[IKConstraint, ...], Tuple[TransformConstraint, ...]]:
    """Build the exact five-phase schedule generalized from the reference rig.

    In normalized single-object mode visible X/Y controls have zero setup rotation. The
    reference setup angles are therefore moved to the matching transform-constraint
    offsets. Multi-object mode retains the previously validated setup-pose payload.
    """

    if not isinstance(plan, LegacyRigBuildPlan):
        raise TypeError("plan must be LegacyRigBuildPlan")
    if not isinstance(layout, TwoAxisScaleRigLayout):
        raise TypeError("layout must be TwoAxisScaleRigLayout")
    if not isinstance(plan.profile, TwoAxisScaleRigProfile):
        raise TypeError("plan.profile must be TwoAxisScaleRigProfile")

    profile = plan.profile
    normalized_single = (
        plan.request.setup_pose_mode is A1RigSetupPoseMode.NORMALIZED_SINGLE
    )
    control_x, control_y, scale_control = plan.control_bone_names
    constraint_bone, _scale_ik, rotate_ik, ik_target = plan.ik_chain_bone_names
    front_to_back_rotation_bones = tuple(reversed(plan.info.sub_bone_names))

    ik = (
        IKConstraint(
            name=profile.scale_ik_constraint(plan.prefix),
            order=1,
            bones=(constraint_bone,),
            target=ik_target,
            extras={"compress": True, "stretch": True},
        ),
    )

    rotation_x_extras = {
        "local": True,
        "relative": True,
        "x": -plan.uniform_scale,
        "y": plan.uniform_scale,
        "scaleX": -1,
        "mixX": 0,
        "mixScaleX": 0,
        "mixShearY": 0,
    }
    rotation_y_extras = {
        "local": True,
        "relative": True,
        "x": -plan.uniform_scale,
        "y": layout.maximum_depth_y,
        "mixX": 0,
        "mixScaleX": 0,
        "mixShearY": 0,
    }
    if normalized_single:
        rotation_x_extras["rotation"] = profile.rotation_x_setup_degrees
        rotation_y_extras["rotation"] = profile.rotation_y_setup_degrees

    transform = (
        TransformConstraint(
            name=profile.rotation_x_constraint(plan.prefix),
            order=0,
            bones=(rotate_ik, plan.main_rotation_bone_name),
            target=control_x,
            extras=rotation_x_extras,
        ),
        TransformConstraint(
            name=profile.rotation_y_constraint(plan.prefix),
            order=4,
            bones=front_to_back_rotation_bones,
            target=control_y,
            extras=rotation_y_extras,
        ),
        TransformConstraint(
            name=profile.scale_constraint(plan.prefix),
            order=2,
            bones=(plan.main_rotation_bone_name, *front_to_back_rotation_bones),
            target=scale_control,
            extras={
                "relative": True,
                "mixRotate": 0,
                "mixX": 0,
                "mixShearY": 0,
            },
        ),
        TransformConstraint(
            name=profile.scale_depth_constraint(plan.prefix),
            order=3,
            bones=plan.info.sub_bone_scale_names,
            target=constraint_bone,
            extras={
                "rotation": -90,
                "x": layout.minimum_depth_y,
                "scaleX": -1,
                "mixRotate": 0,
                "mixX": 0,
                "mixShearY": 0,
            },
        ),
    )
    return ik, transform


__all__ = ["build_two_axis_scale_constraints"]
