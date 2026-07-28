"""Build the generalized Spine bones for the two-axis scale rig."""

from __future__ import annotations

from typing import Tuple

from .legacy_rig_plan import LegacyRigBuildPlan
from .legacy_rig_scale import require_finite_derived
from .model import Bone
from .two_axis_scale_rig_contracts import TwoAxisScaleRigLayout


_REFERENCE_ROTATION_X_SETUP = -134.67
_REFERENCE_ROTATION_Y_SETUP = -17.43


def _rounded(value: float, field_name: str) -> float:
    return require_finite_derived(round(float(value), 2), field_name)


def build_two_axis_scale_bones(
    plan: LegacyRigBuildPlan,
    layout: TwoAxisScaleRigLayout,
) -> Tuple[Bone, ...]:
    """Build the namespaced reference hierarchy in deterministic JSON order."""

    if not isinstance(plan, LegacyRigBuildPlan):
        raise TypeError("plan must be LegacyRigBuildPlan")
    if not isinstance(layout, TwoAxisScaleRigLayout):
        raise TypeError("layout must be TwoAxisScaleRigLayout")

    control_x, control_y, scale_control = plan.control_bone_names
    constraint_bone, scale_ik, rotate_ik, ik_target = plan.ik_chain_bone_names

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
            x=plan.main_x,
            y=plan.main_y,
            color="faff00ff",
            icon="square",
        ),
        Bone(
            name=plan.base_bone_name,
            parent=plan.main_bone_name,
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
            rotation=_REFERENCE_ROTATION_Y_SETUP,
            x=layout.control_x,
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
            rotation=_REFERENCE_ROTATION_X_SETUP,
            x=layout.control_x,
            y=layout.control_y_spacing,
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
