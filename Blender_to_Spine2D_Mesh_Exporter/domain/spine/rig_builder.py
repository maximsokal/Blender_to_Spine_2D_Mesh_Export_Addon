"""Route one immutable rig request to the selected profile builder."""

from __future__ import annotations

from .legacy_rig_assembly import build_legacy_rig
from .legacy_rig_contracts import LegacyRigBuildRequest, LegacyRigBuildResult
from .rig_profiles import A1RigProfile, resolve_a1_rig_profile
from .two_axis_scale_rig import build_two_axis_scale_rig


def build_rig(
    request: LegacyRigBuildRequest,
    rig_profile: A1RigProfile | str = A1RigProfile.THREE_AXIS_ROTATION,
) -> LegacyRigBuildResult:
    """Build one validated rig without leaking profile branching downstream."""

    if not isinstance(request, LegacyRigBuildRequest):
        raise TypeError("request must be LegacyRigBuildRequest")
    resolved = resolve_a1_rig_profile(rig_profile)
    if resolved is A1RigProfile.THREE_AXIS_ROTATION:
        return build_legacy_rig(request)
    if resolved is A1RigProfile.TWO_AXIS_ROTATION_SCALE:
        return build_two_axis_scale_rig(request)
    raise AssertionError(f"Unhandled rig profile: {resolved}")


__all__ = ["build_rig"]
