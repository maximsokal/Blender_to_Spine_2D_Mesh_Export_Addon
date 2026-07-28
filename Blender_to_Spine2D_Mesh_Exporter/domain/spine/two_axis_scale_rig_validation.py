"""Validation owner for the two-axis rotation plus scale rig."""

from __future__ import annotations

from .legacy_rig_contracts import LegacyRigBuildResult
from .legacy_rig_error import LegacyRigBuildError
from .legacy_rig_plan import build_legacy_rig_plan
from .legacy_rig_validation import (
    validate_legacy_rig_numeric_payload,
    validate_legacy_rig_plan,
)
from .model import SpineDocument
from .two_axis_scale_profile import TwoAxisScaleRigProfile
from .two_axis_scale_rig_bones import build_two_axis_scale_bones
from .two_axis_scale_rig_constraints import build_two_axis_scale_constraints
from .two_axis_scale_rig_plan import build_two_axis_scale_layout
from .validator import SpineValidator


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


__all__ = ["validate_two_axis_scale_rig_result"]
