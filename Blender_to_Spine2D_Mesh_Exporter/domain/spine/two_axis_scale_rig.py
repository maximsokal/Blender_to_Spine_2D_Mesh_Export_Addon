"""Build and validate the selectable X/Y rotation plus uniform scale rig.

The implementation preserves the behavior of the user-provided Spine 4.2.43 box rig
without copying its model-specific names or dimensions. Every name is namespaced and
all control placement is derived from the resolved texture scale and Z-group layout.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from math import isfinite
from typing import Tuple

from .legacy_rig_contracts import LegacyRigBuildRequest, LegacyRigBuildResult
from .legacy_rig_error import LegacyRigBuildError
from .legacy_rig_plan import LegacyRigBuildPlan, build_legacy_rig_plan
from .legacy_rig_scale import require_finite_derived
from .legacy_rig_validation import (
    validate_legacy_rig_numeric_payload,
    validate_legacy_rig_plan,
)
from .model import Bone, IKConstraint, SpineDocument, TransformConstraint
from .two_axis_scale_profile import TwoAxisScaleRigProfile
from .validator import SpineValidator


logger = logging.getLogger(__name__)

# Ratios are normalized from the 500x501 reference skeleton. They control only the
# visual placement and length of editor controls; deformation math uses uniform_scale
# and the actual Z-group offsets directly.
_CONTROL_LENGTH_RATIO = 0.4
_CONTROL_X_RATIO = 1.635
_CONTROL_Y_SPACING_RATIO = 0.2
_HELPER_SPAN_RATIO = 1.2
_SCALE_CONTROL_X_RATIO = 2.248
_SCALE_CONTROL_Y_RATIO = 0.12


@dataclass(frozen=True, slots=True)
class TwoAxisScaleRigLayout:
    """Finite model-independent dimensions for one generated control hierarchy."""

    control_length: float
    control_x: float
    control_y_spacing: float
    helper_span: float
    scale_control_x: float
    scale_control_y: float
    minimum_depth_y: float
    maximum_depth_y: float

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be numeric")
            if not isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite")
        for field_name in (
            "control_length",
            "control_x",
            "helper_span",
            "scale_control_x",
        ):
            if float(getattr(self, field_name)) <= 0.0:
                raise ValueError(f"{field_name} must be positive")
        if self.minimum_depth_y > self.maximum_depth_y:
            raise ValueError("minimum_depth_y cannot exceed maximum_depth_y")


def _rounded(value: float, field_name: str) -> float:
    return require_finite_derived(round(float(value), 2), field_name)


def build_two_axis_scale_layout(plan: LegacyRigBuildPlan) -> TwoAxisScaleRigLayout:
    """Resolve editor-control layout without hard-coding one reference object size."""

    if not isinstance(plan, LegacyRigBuildPlan):
        raise TypeError("plan must be LegacyRigBuildPlan")
    if not plan.z_groups:
        raise ValueError("plan.z_groups cannot be empty")

    scale = require_finite_derived(plan.uniform_scale, "uniform_scale")
    depth_values = tuple(float(group.y_offset_pixels) for group in plan.z_groups)
    return TwoAxisScaleRigLayout(
        control_length=_rounded(scale * _CONTROL_LENGTH_RATIO, "control_length"),
        control_x=_rounded(scale * _CONTROL_X_RATIO, "control_x"),
        control_y_spacing=_rounded(
            scale * _CONTROL_Y_SPACING_RATIO,
            "control_y_spacing",
        ),
        helper_span=_rounded(scale * _HELPER_SPAN_RATIO, "helper_span"),
        scale_control_x=_rounded(
            scale * _SCALE_CONTROL_X_RATIO,
            "scale_control_x",
        ),
        scale_control_y=_rounded(
            scale * _SCALE_CONTROL_Y_RATIO,
            "scale_control_y",
        ),
        minimum_depth_y=_rounded(min(depth_values), "minimum_depth_y"),
        maximum_depth_y=_rounded(max(depth_values), "maximum_depth_y"),
    )


def build_two_axis_scale_bones(
    plan: LegacyRigBuildPlan,
    layout: TwoAxisScaleRigLayout,
) -> Tuple[Bone, ...]:
    """Build the generalized reference hierarchy in deterministic JSON order."""

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


def build_two_axis_scale_constraints(
    plan: LegacyRigBuildPlan,
    layout: TwoAxisScaleRigLayout,
) -> tuple[Tuple[IKConstraint, ...], Tuple[TransformConstraint, ...]]:
    """Build the exact cross-constraint evaluation schedule from the reference rig."""

    if not isinstance(plan, LegacyRigBuildPlan):
        raise TypeError("plan must be LegacyRigBuildPlan")
    if not isinstance(layout, TwoAxisScaleRigLayout):
        raise TypeError("layout must be TwoAxisScaleRigLayout")
    if not isinstance(plan.profile, TwoAxisScaleRigProfile):
        raise TypeError("plan.profile must be TwoAxisScaleRigProfile")

    profile = plan.profile
    control_x, control_y, scale_control = plan.control_bone_names
    constraint_bone, _scale_ik, rotate_ik, ik_target = plan.ik_chain_bone_names

    ik = (
        IKConstraint(
            name=profile.scale_ik_constraint(plan.prefix),
            order=1,
            bones=(constraint_bone,),
            target=ik_target,
            extras={"compress": True, "stretch": True},
        ),
    )

    transform = (
        TransformConstraint(
            name=profile.rotation_x_constraint(plan.prefix),
            order=0,
            bones=(rotate_ik, plan.main_rotation_bone_name),
            target=control_x,
            extras={
                "local": True,
                "relative": True,
                "x": -plan.uniform_scale,
                "y": plan.uniform_scale,
                "scaleX": -1,
                "mixX": 0,
                "mixScaleX": 0,
                "mixShearY": 0,
            },
        ),
        TransformConstraint(
            name=profile.rotation_y_constraint(plan.prefix),
            order=4,
            bones=plan.info.sub_bone_names,
            target=control_y,
            extras={
                "local": True,
                "relative": True,
                "x": -plan.uniform_scale,
                "y": layout.maximum_depth_y,
                "mixX": 0,
                "mixScaleX": 0,
                "mixShearY": 0,
            },
        ),
        TransformConstraint(
            name=profile.scale_constraint(plan.prefix),
            order=2,
            bones=(plan.main_rotation_bone_name, *plan.info.sub_bone_names),
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


def validate_two_axis_scale_rig_result(result: LegacyRigBuildResult) -> None:
    """Validate deterministic structure, finite payload, and Spine references."""

    if not isinstance(result, LegacyRigBuildResult):
        raise TypeError("result must be LegacyRigBuildResult")
    if not isinstance(result.profile, TwoAxisScaleRigProfile):
        raise TypeError("result.profile must be TwoAxisScaleRigProfile")

    plan = build_legacy_rig_plan(result.request, result.profile)
    validate_legacy_rig_plan(plan)
    layout = build_two_axis_scale_layout(plan)
    expected_bones = build_two_axis_scale_bones(plan, layout)
    expected_ik, expected_transform = build_two_axis_scale_constraints(plan, layout)

    if result.info != plan.info:
        raise LegacyRigBuildError("two-axis rig info differs from its deterministic plan")
    if result.bones != expected_bones:
        raise LegacyRigBuildError("two-axis rig bones differ from the deterministic plan")
    if result.ik != expected_ik:
        raise LegacyRigBuildError("two-axis rig IK differs from the deterministic plan")
    if result.transform != expected_transform:
        raise LegacyRigBuildError(
            "two-axis rig Transform constraints differ from the deterministic plan"
        )

    if tuple(item.order for item in result.transform) != (0, 4, 2, 3):
        raise LegacyRigBuildError("two-axis transform JSON order changed unexpectedly")
    combined_orders = tuple(item.order for item in result.transform) + tuple(
        item.order for item in result.ik
    )
    if set(combined_orders) != {0, 1, 2, 3, 4}:
        raise LegacyRigBuildError(
            f"two-axis constraint orders must cover 0..4 exactly, got {combined_orders}"
        )

    validate_legacy_rig_numeric_payload(result)
    document = SpineDocument(
        skeleton={"spine": result.profile.spine_version},
        bones=result.bones,
        slots=(),
        skins=(),
        ik=result.ik,
        transform=result.transform,
    )
    try:
        SpineValidator().validate_or_raise(document)
    except Exception as exc:
        raise LegacyRigBuildError(
            f"two-axis rig failed Spine cross-reference validation: {exc}"
        ) from exc


def build_two_axis_scale_rig(
    request: LegacyRigBuildRequest,
    profile: TwoAxisScaleRigProfile | None = None,
) -> LegacyRigBuildResult:
    """Build the complete generalized X/Y plus scale rig."""

    if not isinstance(request, LegacyRigBuildRequest):
        raise TypeError("request must be LegacyRigBuildRequest")
    resolved_profile = TwoAxisScaleRigProfile() if profile is None else profile
    if not isinstance(resolved_profile, TwoAxisScaleRigProfile):
        raise TypeError("profile must be TwoAxisScaleRigProfile")

    stage = "PLAN"
    try:
        plan = build_legacy_rig_plan(request, resolved_profile)
        validate_legacy_rig_plan(plan)

        stage = "LAYOUT"
        layout = build_two_axis_scale_layout(plan)

        stage = "BONES"
        bones = build_two_axis_scale_bones(plan, layout)

        stage = "CONSTRAINTS"
        ik, transform = build_two_axis_scale_constraints(plan, layout)

        stage = "RESULT"
        result = LegacyRigBuildResult(
            request=request,
            profile=resolved_profile,
            bones=bones,
            ik=ik,
            transform=transform,
            info=plan.info,
        )
        validate_two_axis_scale_rig_result(result)
        logger.debug(
            "Built two-axis scale rig for %s: bones=%d z_groups=%d",
            request.prefix,
            len(result.bones),
            len(result.info.z_groups),
        )
        return result
    except LegacyRigBuildError as exc:
        raise LegacyRigBuildError(
            f"Unable to build two-axis scale rig for '{request.prefix}' at {stage}: {exc}"
        ) from exc
    except Exception as exc:
        raise LegacyRigBuildError(
            f"Unable to build two-axis scale rig for '{request.prefix}' at {stage}: {exc}"
        ) from exc


__all__ = [
    "TwoAxisScaleRigLayout",
    "build_two_axis_scale_bones",
    "build_two_axis_scale_constraints",
    "build_two_axis_scale_layout",
    "build_two_axis_scale_rig",
    "validate_two_axis_scale_rig_result",
]
