"""Build the generalized Spine bones for the two-axis scale rig."""

from __future__ import annotations

from typing import Tuple

from .legacy_rig_plan import LegacyRigBuildPlan
from .legacy_rig_scale import require_finite_derived
from .model import Bone
from .rig_profiles import A1RigSetupPoseMode
from .two_axis_scale_profile import TwoAxisScaleRigProfile
from .two_axis_scale_rig_contracts import TwoAxisScaleRigLayout


def _rounded(value: float, field_name: str) -> float:
    return require_finite_derived(round(float(value), 2), field_name)


def _build_model_space_bones(
    plan: LegacyRigBuildPlan,
    layout: TwoAxisScaleRigLayout,
) -> Tuple[Bone, ...]:
    """Preserve the historical model-space and normalized-single hierarchy."""

    normalized_single = (
        plan.request.setup_pose_mode is A1RigSetupPoseMode.NORMALIZED_SINGLE
    )
    control_x, control_y, scale_control = plan.control_bone_names
    constraint_bone, scale_ik, rotate_ik, ik_target = plan.ik_chain_bone_names

    main_x = 0.0 if normalized_single else plan.main_x
    main_y = 0.0 if normalized_single else plan.main_y
    base_x = plan.main_x if normalized_single else None
    base_y = plan.main_y if normalized_single else None
    control_origin_x = plan.main_x if normalized_single else 0.0
    control_origin_y = plan.main_y if normalized_single else 0.0

    z_bones: list[Bone] = []
    for group in plan.z_groups:
        z_bones.extend(
            (
                Bone(
                    name=group.scale_bone_name,
                    parent=plan.main_rotation_bone_name,
                    rotation=90.0,
                    y=group.y_offset_pixels,
                    extras={"inherit": "onlyTranslation"},
                ),
                Bone(
                    name=group.bone_name,
                    parent=group.scale_bone_name,
                    rotation=-90.0,
                ),
            )
        )

    return (
        Bone(name=plan.root_bone_name),
        Bone(
            name=plan.main_bone_name,
            parent=plan.root_bone_name,
            x=main_x,
            y=main_y,
            color="faff00ff",
            icon="square",
        ),
        Bone(
            name=plan.base_bone_name,
            parent=plan.main_bone_name,
            x=base_x,
            y=base_y,
            color="faff00ff",
        ),
        Bone(
            name=plan.scale_bone_name,
            parent=plan.base_bone_name,
            scale_x=0.0,
        ),
        Bone(
            name=plan.main_rotation_bone_name,
            parent=plan.scale_bone_name,
        ),
        *z_bones,
        Bone(
            name=control_y,
            parent=plan.main_bone_name,
            length=layout.control_length,
            rotation=0.0,
            x=_rounded(control_origin_x + layout.control_x, "rotation_y.x"),
            y=_rounded(control_origin_y, "rotation_y.y"),
            color="1aff00ff",
        ),
        Bone(
            name=constraint_bone,
            parent=plan.base_bone_name,
            length=layout.helper_span,
            rotation=90.0,
            color="abe323ff",
        ),
        Bone(
            name=scale_ik,
            parent=plan.base_bone_name,
            y=layout.helper_span,
            scale_x=0.0,
        ),
        Bone(
            name=rotate_ik,
            parent=scale_ik,
            x=-layout.helper_span,
        ),
        Bone(
            name=ik_target,
            parent=rotate_ik,
            x=layout.helper_span,
            color="ff3f00ff",
            icon="ik",
        ),
        Bone(
            name=control_x,
            parent=plan.main_bone_name,
            length=layout.control_length,
            rotation=0.0,
            x=_rounded(control_origin_x + layout.control_x, "rotation_x.x"),
            y=_rounded(
                control_origin_y + layout.control_y_spacing,
                "rotation_x.y",
            ),
            color="ff0000ff",
        ),
        Bone(
            name=scale_control,
            parent=plan.root_bone_name,
            x=_rounded(plan.main_x + layout.scale_control_x, "scale_control.x"),
            y=_rounded(plan.main_y + layout.scale_control_y, "scale_control.y"),
            color="abe323ff",
            icon="square",
        ),
    )


def _build_camera_relative_bones(
    plan: LegacyRigBuildPlan,
    layout: TwoAxisScaleRigLayout,
) -> Tuple[Bone, ...]:
    """Place Object Origin below the camera-orbital X/Y transform layers.

    The one depth layer is evaluated before ``base``. Its X/Y transformations therefore
    rotate and translate the complete projected Object Origin around camera-space zero.
    ``base`` then supplies the authored object position and becomes the local scale/pivot
    parent for every generated vertex bone.
    """

    if len(plan.z_groups) != 1:
        raise ValueError(
            "PREPROJECTED_SCREEN requires exactly one camera depth group"
        )

    group = plan.z_groups[0]
    control_x, control_y, scale_control = plan.control_bone_names
    constraint_bone, scale_ik, rotate_ik, ik_target = plan.ik_chain_bone_names

    # The depth helper contributes setup Y before the object base. Counter it on base so
    # base world position is exactly the projected Blender Object Origin.
    base_x = _rounded(plan.main_x, "camera_base.x")
    base_y = _rounded(
        plan.main_y - float(group.y_offset_pixels),
        "camera_base.y",
    )
    control_origin_x = plan.main_x
    control_origin_y = plan.main_y

    return (
        Bone(name=plan.root_bone_name),
        Bone(
            name=plan.main_bone_name,
            parent=plan.root_bone_name,
            x=0.0,
            y=0.0,
            color="faff00ff",
            icon="square",
        ),
        Bone(
            name=plan.scale_bone_name,
            parent=plan.main_bone_name,
            scale_x=0.0,
        ),
        Bone(
            name=plan.main_rotation_bone_name,
            parent=plan.scale_bone_name,
        ),
        Bone(
            name=group.scale_bone_name,
            parent=plan.main_rotation_bone_name,
            rotation=90.0,
            y=group.y_offset_pixels,
            extras={"inherit": "onlyTranslation"},
        ),
        Bone(
            name=group.bone_name,
            parent=group.scale_bone_name,
            rotation=-90.0,
        ),
        Bone(
            name=plan.base_bone_name,
            parent=group.bone_name,
            x=base_x,
            y=base_y,
            color="faff00ff",
        ),
        Bone(
            name=control_y,
            parent=plan.main_bone_name,
            length=layout.control_length,
            rotation=0.0,
            x=_rounded(control_origin_x + layout.control_x, "rotation_y.x"),
            y=_rounded(control_origin_y, "rotation_y.y"),
            color="1aff00ff",
        ),
        Bone(
            name=constraint_bone,
            parent=plan.main_bone_name,
            length=layout.helper_span,
            rotation=90.0,
            x=_rounded(control_origin_x, "camera_constraint.x"),
            y=_rounded(control_origin_y, "camera_constraint.y"),
            color="abe323ff",
        ),
        Bone(
            name=scale_ik,
            parent=plan.main_bone_name,
            x=_rounded(control_origin_x, "camera_scale_ik.x"),
            y=_rounded(
                control_origin_y + layout.helper_span,
                "camera_scale_ik.y",
            ),
            scale_x=0.0,
        ),
        Bone(
            name=rotate_ik,
            parent=scale_ik,
            x=-layout.helper_span,
        ),
        Bone(
            name=ik_target,
            parent=rotate_ik,
            x=layout.helper_span,
            color="ff3f00ff",
            icon="ik",
        ),
        Bone(
            name=control_x,
            parent=plan.main_bone_name,
            length=layout.control_length,
            rotation=0.0,
            x=_rounded(control_origin_x + layout.control_x, "rotation_x.x"),
            y=_rounded(
                control_origin_y + layout.control_y_spacing,
                "rotation_x.y",
            ),
            color="ff0000ff",
        ),
        Bone(
            name=scale_control,
            parent=plan.root_bone_name,
            x=_rounded(plan.main_x + layout.scale_control_x, "scale_control.x"),
            y=_rounded(plan.main_y + layout.scale_control_y, "scale_control.y"),
            color="abe323ff",
            icon="square",
        ),
    )


def build_two_axis_scale_bones(
    plan: LegacyRigBuildPlan,
    layout: TwoAxisScaleRigLayout,
) -> Tuple[Bone, ...]:
    """Build the selected deterministic model-space or camera-relative hierarchy."""

    if not isinstance(plan, LegacyRigBuildPlan):
        raise TypeError("plan must be LegacyRigBuildPlan")
    if not isinstance(layout, TwoAxisScaleRigLayout):
        raise TypeError("layout must be TwoAxisScaleRigLayout")
    if not isinstance(plan.profile, TwoAxisScaleRigProfile):
        raise TypeError("plan.profile must be TwoAxisScaleRigProfile")

    if plan.request.setup_pose_mode is A1RigSetupPoseMode.PREPROJECTED_SCREEN:
        return _build_camera_relative_bones(plan, layout)
    return _build_model_space_bones(plan, layout)


__all__ = ["build_two_axis_scale_bones"]
