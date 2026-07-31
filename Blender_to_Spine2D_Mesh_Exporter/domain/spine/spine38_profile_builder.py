"""Build canonical profile documents selected for Spine 3.8 adaptation."""

from __future__ import annotations

from .legacy_rig_assembly import build_legacy_rig
from .legacy_rig_contracts import LegacyRigBuildRequest, LegacyRigBuildResult
from .rig_profiles import A1RigProfile, resolve_a1_rig_profile
from .two_axis_scale_rig import build_two_axis_scale_rig


def build_spine38_profile(
    request: LegacyRigBuildRequest,
    profile: A1RigProfile | str,
) -> LegacyRigBuildResult:
    """Build the unchanged canonical profile before target-specific finalization."""

    if not isinstance(request, LegacyRigBuildRequest):
        raise TypeError("request must be LegacyRigBuildRequest")
    resolved = resolve_a1_rig_profile(profile)
    if resolved is A1RigProfile.THREE_AXIS_ROTATION:
        return build_legacy_rig(request)
    if resolved is A1RigProfile.TWO_AXIS_ROTATION_SCALE:
        return build_two_axis_scale_rig(request)
    raise AssertionError(f"Unhandled profile: {resolved}")


__all__ = ["build_spine38_profile"]
