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


def build_two_axis_scale_bones(
    plan: LegacyRigBuildPlan,
    layout: TwoAxisScaleRigLayout,
) -> Tuple[Bone, ...]:
    """Build the namespaced reference hierarchy in deterministic JSON order.

    ``NORMALIZED_SINGLE`` keeps the visible main neutral and moves object placement into
    the internal base layer.

    ``PREPROJECTED_SCREEN`` now uses the same placement split for a different reason:
    ``main`` is the camera-space origin and ``base`` is the projected Blender Object
    Origin. With one Object-Origin depth group, X/Y constraints therefore transform one
    complete camera-relative object layer instead of deforming separate vertex depths.

    Ordinary composed model-space documents retain their calculated main placement.
    """

    if not isinstance(plan, LegacyRigBuildPlan):
        raise TypeError("plan must be LegacyRigBuildPlan")
    if not isinstance(layout, TwoAxisScaleRigLayout):
        raise TypeError("layout must be TwoAxisScaleRigLayout")
    if not isinstance(plan.profile, TwoAxisScaleRigProfile):
        raise TypeError("plan.profile must be TwoAxisScaleRigProfile")

    normalized_single = (
        plan.request.setup_pose_mode is A1RigSetupPoseMode.NORMALIZED_SINGLE
    )
    camera_relative = (
        plan.request.setup_pose_mode is A1RigSetupPoseMode.PREPROJECTED_SCREEN
    )
    placement_in_base = normalized_single or camera_relative

    control_x, control_y, scale_control = plan.control_bone_names
    constraint_bone, scale_ik, rotate_ik, ik_target = plan.ik_chain_bone_names

    # Camera-relative and normalized standalone documents keep root/main at zero. The
    # projected/authored Object Origin is stored on base, before local scale/deformation,
    # so the full object orbits the camera origin while local Scale stays object-centred.
    main_x = 0.0 if placement_in_base else plan.main_x
    main_y = 0.0 if placement_in_base else plan.main_y
    base_x = plan.main_x if placement_in_base else None
    base_y = plan.main_y if placement_in_base else None
    control_origin_x = plan.main_x if placement_in_base else 0.0
    control_origin_y = plan.main_y if placement_in_base else 0.0

    # Visible controls are authoring handles. Setup angles remain constraint-owned.
    control_x_rotation = 0.0
    control_y_rotation = 0.0

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
            rotation=control_y_rotation,
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
            rotation=control_x_rotation,
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


__all__ = ["build_two_axis_scale_bones"]
