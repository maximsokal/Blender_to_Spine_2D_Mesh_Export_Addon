"""Resolve the immutable name, scale, and Z-layout plan for a legacy A1 rig."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .legacy_profile import LegacyRigProfile
from .legacy_rig_contracts import (
    LegacyRigBuildRequest,
    LegacyRigInfo,
    LegacyZGroupBuildInfo,
    LegacyZGroupOriginMode,
)
from .legacy_rig_scale import (
    calculate_uniform_scale,
    require_finite_derived,
    resolve_main_position,
)


@dataclass(frozen=True, slots=True)
class LegacyRigBuildPlan:
    request: LegacyRigBuildRequest
    profile: LegacyRigProfile
    prefix: str
    uniform_scale: float
    half_scale: float
    main_x: float
    main_y: float
    root_bone_name: str
    main_bone_name: str
    base_bone_name: str
    scale_bone_name: str
    main_rotation_bone_name: str
    control_bone_names: Tuple[str, str, str]
    ik_chain_bone_names: Tuple[str, str, str, str]
    z_groups: Tuple[LegacyZGroupBuildInfo, ...]
    info: LegacyRigInfo


def _duplicates(values: Tuple[str, ...]) -> Tuple[str, ...]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return tuple(sorted(duplicates))


def _resolve_z_reference(
    request: LegacyRigBuildRequest,
    ordered_groups: tuple,
) -> float:
    """Resolve one explicit finite source-Z reference for depth offsets."""

    if not isinstance(request, LegacyRigBuildRequest):
        raise TypeError("request must be LegacyRigBuildRequest")
    if not isinstance(ordered_groups, tuple) or not ordered_groups:
        raise ValueError("ordered_groups must be a non-empty tuple")

    mode = request.z_group_origin_mode
    if mode is LegacyZGroupOriginMode.MINIMUM_Z:
        reference = float(ordered_groups[0].z_value)
    elif mode is LegacyZGroupOriginMode.OBJECT_ORIGIN:
        reference = 0.0
    else:
        raise TypeError(f"Unsupported z_group_origin_mode: {mode!r}")
    return require_finite_derived(reference, "z_group_reference_z")


def build_legacy_z_group_metadata(
    request: LegacyRigBuildRequest,
    profile: LegacyRigProfile,
    uniform_scale: float,
) -> Tuple[LegacyZGroupBuildInfo, ...]:
    """Sort source Z values and resolve dense names and policy-based offsets.

    Explicit ``height_real_pixels`` values are already-resolved absolute Spine offsets
    and therefore are never re-referenced. Derived values use either the historical
    minimum-Z reference or Blender local Z=0, selected by ``request``.
    """

    if not isinstance(request, LegacyRigBuildRequest):
        raise TypeError("request must be LegacyRigBuildRequest")
    if not isinstance(profile, LegacyRigProfile):
        raise TypeError("profile must be LegacyRigProfile")
    resolved_scale = require_finite_derived(uniform_scale, "uniform_scale")
    if resolved_scale <= 0.0:
        raise ValueError("uniform_scale must be positive")

    ordered_groups = tuple(
        sorted(request.z_groups, key=lambda group: float(group.z_value))
    )
    reference_z = _resolve_z_reference(request, ordered_groups)
    result: list[LegacyZGroupBuildInfo] = []

    for offset, group in enumerate(ordered_groups):
        index = profile.z_index_base + offset
        scale_bone_name = profile.z_scale_bone(request.prefix, index)
        bone_name = profile.z_bone(request.prefix, index)
        if group.height_real_pixels is not None:
            y_offset = require_finite_derived(
                float(group.height_real_pixels),
                f"z_groups[{offset}].height_real_pixels",
            )
            calculation_method = "height_real_pixels"
        else:
            delta = require_finite_derived(
                float(group.z_value) - reference_z,
                f"z_groups[{offset}].delta",
            )
            y_offset = require_finite_derived(
                delta * resolved_scale,
                f"z_groups[{offset}].y_offset_pixels",
            )
            calculation_method = "direct_3d_scaling"

        rounded_y = require_finite_derived(
            round(y_offset, 2),
            f"z_groups[{offset}].rounded_y_offset_pixels",
        )
        # Python can retain a textual negative zero after rounding. Spine JSON and
        # diagnostics must expose one canonical zero regardless of source sign.
        if rounded_y == 0.0:
            rounded_y = 0.0
        result.append(
            LegacyZGroupBuildInfo(
                z_value=float(group.z_value),
                index=index,
                y_offset_pixels=rounded_y,
                calculation_method=calculation_method,
                scale_bone_name=scale_bone_name,
                bone_name=bone_name,
            )
        )

    return tuple(result)


def planned_bone_names(plan: LegacyRigBuildPlan) -> Tuple[str, ...]:
    """Return exact generated bone order for namespace validation."""

    if not isinstance(plan, LegacyRigBuildPlan):
        raise TypeError("plan must be LegacyRigBuildPlan")
    return (
        plan.root_bone_name,
        plan.main_bone_name,
        plan.base_bone_name,
        plan.scale_bone_name,
        plan.main_rotation_bone_name,
        *(
            name
            for group in plan.z_groups
            for name in (group.scale_bone_name, group.bone_name)
        ),
        *plan.control_bone_names,
        *plan.ik_chain_bone_names,
    )


def build_legacy_rig_plan(
    request: LegacyRigBuildRequest,
    profile: LegacyRigProfile,
) -> LegacyRigBuildPlan:
    """Resolve all deterministic data before any Spine model objects are created."""

    if not isinstance(request, LegacyRigBuildRequest):
        raise TypeError("request must be LegacyRigBuildRequest")
    if not isinstance(profile, LegacyRigProfile):
        raise TypeError("profile must be LegacyRigProfile")

    prefix = request.prefix
    uniform_scale = calculate_uniform_scale(
        request.texture_width,
        request.texture_height,
        request.scale_mode,
    )
    half_scale = require_finite_derived(uniform_scale / 2.0, "half_scale")
    if half_scale <= 0.0:
        raise ValueError("half_scale must be positive")
    require_finite_derived(uniform_scale * 2.0, "double_uniform_scale")
    main_x, main_y = resolve_main_position(request)

    root_name = profile.root_bone()
    main_name = profile.main_bone(prefix)
    base_name = profile.base_bone(prefix)
    scale_name = profile.scale_rotate_x_bone(prefix)
    rotate_name = profile.rotate_x_bone(prefix)
    control_names = profile.control_bones(prefix)
    ik_chain_names = profile.ik_chain_bones(prefix)
    z_groups = build_legacy_z_group_metadata(request, profile, uniform_scale)

    info = LegacyRigInfo(
        profile_id=profile.profile_id,
        prefix=prefix,
        uniform_scale=uniform_scale,
        half_scale=half_scale,
        root_bone_name=root_name,
        main_bone_name=main_name,
        base_bone_name=base_name,
        scale_bone_name=scale_name,
        main_rotation_bone_name=rotate_name,
        control_bone_names=control_names,
        ik_chain_bone_names=ik_chain_names,
        z_groups=z_groups,
        sub_bone_scale_names=tuple(group.scale_bone_name for group in z_groups),
        sub_bone_names=tuple(group.bone_name for group in z_groups),
    )
    plan = LegacyRigBuildPlan(
        request=request,
        profile=profile,
        prefix=prefix,
        uniform_scale=uniform_scale,
        half_scale=half_scale,
        main_x=main_x,
        main_y=main_y,
        root_bone_name=root_name,
        main_bone_name=main_name,
        base_bone_name=base_name,
        scale_bone_name=scale_name,
        main_rotation_bone_name=rotate_name,
        control_bone_names=control_names,
        ik_chain_bone_names=ik_chain_names,
        z_groups=z_groups,
        info=info,
    )
    duplicates = _duplicates(planned_bone_names(plan))
    if duplicates:
        raise ValueError(f"legacy rig bone namespace contains duplicates: {duplicates}")
    return plan


__all__ = [
    "LegacyRigBuildPlan",
    "build_legacy_rig_plan",
    "build_legacy_z_group_metadata",
    "planned_bone_names",
]
