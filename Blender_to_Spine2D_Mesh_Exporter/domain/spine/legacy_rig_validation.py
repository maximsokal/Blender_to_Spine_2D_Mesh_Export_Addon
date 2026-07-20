"""Exact semantic validation for generated legacy A1 rigs."""

from __future__ import annotations

from math import isfinite
from typing import Mapping, Sequence

from .legacy_rig_bones import build_legacy_rig_bones
from .legacy_rig_constraints import build_legacy_rig_constraints
from .legacy_rig_contracts import LegacyRigBuildResult
from .legacy_rig_error import LegacyRigBuildError
from .legacy_rig_plan import (
    LegacyRigBuildPlan,
    build_legacy_rig_plan,
    planned_bone_names,
)
from .legacy_rig_scale import calculate_uniform_scale, resolve_main_position
from .model import SpineDocument
from .validator import SpineValidator


def _require_finite(value: object, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LegacyRigBuildError(f"{path} must be numeric")
    if not isfinite(float(value)):
        raise LegacyRigBuildError(f"{path} must be finite")


def _validate_nested_numeric_payload(value: object, path: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        _require_finite(value, path)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_nested_numeric_payload(item, f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for index, item in enumerate(value):
            _validate_nested_numeric_payload(item, f"{path}[{index}]")


def validate_legacy_rig_plan(plan: LegacyRigBuildPlan) -> None:
    """Validate internal plan consistency before model-object creation."""

    if not isinstance(plan, LegacyRigBuildPlan):
        raise TypeError("plan must be LegacyRigBuildPlan")
    if plan.request.prefix != plan.prefix or plan.info.prefix != plan.prefix:
        raise LegacyRigBuildError("request, plan, and info prefixes do not match")
    if plan.info.profile_id != plan.profile.profile_id:
        raise LegacyRigBuildError("rig info profile_id does not match profile")
    expected_scale = calculate_uniform_scale(
        plan.request.texture_width,
        plan.request.texture_height,
        plan.request.scale_mode,
    )
    if plan.uniform_scale != expected_scale or plan.info.uniform_scale != expected_scale:
        raise LegacyRigBuildError("rig uniform scale is inconsistent with request")
    expected_half = expected_scale / 2.0
    if plan.half_scale != expected_half or plan.info.half_scale != expected_half:
        raise LegacyRigBuildError("rig half scale is inconsistent with uniform scale")
    expected_main = resolve_main_position(plan.request)
    if (plan.main_x, plan.main_y) != expected_main:
        raise LegacyRigBuildError("rig main position is inconsistent with request")

    ordered_z = tuple(group.z_value for group in plan.z_groups)
    if ordered_z != tuple(sorted(ordered_z)):
        raise LegacyRigBuildError("rig Z-group metadata is not sorted")
    expected_indices = tuple(
        range(plan.profile.z_index_base, plan.profile.z_index_base + len(plan.z_groups))
    )
    if tuple(group.index for group in plan.z_groups) != expected_indices:
        raise LegacyRigBuildError("rig Z-group indices are not dense")
    if plan.info.z_groups != plan.z_groups:
        raise LegacyRigBuildError("rig info Z-group metadata differs from plan")
    if plan.info.sub_bone_scale_names != tuple(
        group.scale_bone_name for group in plan.z_groups
    ):
        raise LegacyRigBuildError("rig scale-bone metadata differs from Z groups")
    if plan.info.sub_bone_names != tuple(group.bone_name for group in plan.z_groups):
        raise LegacyRigBuildError("rig bone metadata differs from Z groups")
    names = planned_bone_names(plan)
    if len(names) != len(set(names)):
        raise LegacyRigBuildError("rig plan contains duplicate bone names")

    for field_name in (
        "uniform_scale",
        "half_scale",
        "main_x",
        "main_y",
    ):
        _require_finite(getattr(plan, field_name), f"plan.{field_name}")
    for index, group in enumerate(plan.z_groups):
        _require_finite(group.z_value, f"plan.z_groups[{index}].z_value")
        _require_finite(
            group.y_offset_pixels,
            f"plan.z_groups[{index}].y_offset_pixels",
        )


def validate_legacy_rig_numeric_payload(result: LegacyRigBuildResult) -> None:
    """Reject NaN/Infinity in generated bone and constraint payloads."""

    if not isinstance(result, LegacyRigBuildResult):
        raise TypeError("result must be LegacyRigBuildResult")
    for bone_index, bone in enumerate(result.bones):
        for field_name in ("length", "x", "y", "rotation", "scale_x", "scale_y"):
            value = getattr(bone, field_name)
            if value is not None:
                _require_finite(value, f"bones[{bone_index}].{field_name}")
        _validate_nested_numeric_payload(bone.extras, f"bones[{bone_index}].extras")

    for collection_name, constraints in (
        ("ik", result.ik),
        ("transform", result.transform),
    ):
        for index, constraint in enumerate(constraints):
            if isinstance(constraint.order, bool):
                raise LegacyRigBuildError(
                    f"{collection_name}[{index}].order cannot be bool"
                )
            _validate_nested_numeric_payload(
                constraint.extras,
                f"{collection_name}[{index}].extras",
            )


def validate_legacy_rig_result(result: LegacyRigBuildResult) -> None:
    """Validate exact generated structure and generic Spine cross-references."""

    if not isinstance(result, LegacyRigBuildResult):
        raise TypeError("result must be LegacyRigBuildResult")

    expected_plan = build_legacy_rig_plan(result.request, result.profile)
    validate_legacy_rig_plan(expected_plan)
    expected_bones = build_legacy_rig_bones(expected_plan)
    expected_ik, expected_transform = build_legacy_rig_constraints(expected_plan)

    if result.info != expected_plan.info:
        raise LegacyRigBuildError("rig info does not match the deterministic build plan")
    if result.bones != expected_bones:
        raise LegacyRigBuildError("rig bones do not match the deterministic build plan")
    if result.ik != expected_ik:
        raise LegacyRigBuildError("rig IK constraints do not match the legacy schema")
    if result.transform != expected_transform:
        raise LegacyRigBuildError(
            "rig Transform constraints do not match the legacy schema"
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
            f"generated rig failed Spine cross-reference validation: {exc}"
        ) from exc


__all__ = [
    "validate_legacy_rig_numeric_payload",
    "validate_legacy_rig_plan",
    "validate_legacy_rig_result",
]
