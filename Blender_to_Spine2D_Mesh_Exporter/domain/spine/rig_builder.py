"""Route one immutable rig request to the selected profile and Spine target builder."""

from __future__ import annotations

from .legacy_rig_assembly import build_legacy_rig
from .legacy_rig_contracts import LegacyRigBuildRequest, LegacyRigBuildResult
from .rig_profiles import A1RigProfile, resolve_a1_rig_profile
from .two_axis_scale_rig import build_two_axis_scale_rig
from .two_axis_scale_spine41 import adapt_two_axis_scale_rig_for_spine41
from .version_target import (
    DEFAULT_SPINE_JSON_TARGET,
    SpineJsonTarget,
    resolve_spine_json_target,
)


def build_rig(
    request: LegacyRigBuildRequest,
    rig_profile: A1RigProfile | str = A1RigProfile.THREE_AXIS_ROTATION,
    *,
    spine_target: object = DEFAULT_SPINE_JSON_TARGET,
) -> LegacyRigBuildResult:
    """Build one validated target-aware rig without downstream profile guessing."""

    if not isinstance(request, LegacyRigBuildRequest):
        raise TypeError("request must be LegacyRigBuildRequest")
    resolved_profile = resolve_a1_rig_profile(rig_profile)
    resolved_target = resolve_spine_json_target(spine_target)

    if resolved_profile is A1RigProfile.THREE_AXIS_ROTATION:
        if resolved_target is not SpineJsonTarget.SPINE_4_2:
            raise ValueError(
                "THREE_AXIS_ROTATION is not yet runtime-validated for "
                f"{resolved_target.label} ({resolved_target.exact_version})"
            )
        return build_legacy_rig(request)

    if resolved_profile is A1RigProfile.TWO_AXIS_ROTATION_SCALE:
        rig = build_two_axis_scale_rig(request)
        if resolved_target is SpineJsonTarget.SPINE_4_2:
            return rig
        if resolved_target is SpineJsonTarget.SPINE_4_1:
            return adapt_two_axis_scale_rig_for_spine41(rig)
        raise ValueError(
            "TWO_AXIS_ROTATION_SCALE is not implemented for "
            f"{resolved_target.label} ({resolved_target.exact_version})"
        )

    raise AssertionError(f"Unhandled rig profile: {resolved_profile}")


__all__ = ["build_rig"]
