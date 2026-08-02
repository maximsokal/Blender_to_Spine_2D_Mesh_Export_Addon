"""Rebuild canonical two-axis rigs for already-projected screen geometry."""

from __future__ import annotations

from dataclasses import replace

from .legacy_rig_contracts import LegacyRigBuildResult
from .rig_profiles import (
    A1RigProfile,
    A1RigSetupPoseMode,
    resolve_a1_rig_profile,
)
from .two_axis_scale_rig import build_two_axis_scale_rig


def ensure_preprojected_screen_rig(
    rig: LegacyRigBuildResult,
) -> LegacyRigBuildResult:
    """Return a validated two-axis rig with an identity screen-space setup pose.

    Active Camera Normal / UV Segments and rendered Camera Projection have already
    converted source geometry into final camera-screen X/Y. Reapplying the historical
    model-space setup offsets would project that geometry a second time. This helper
    preserves the complete two-axis hierarchy and constraint topology while selecting
    the dedicated setup-pose policy that neutralizes only those initial offsets.
    """

    if not isinstance(rig, LegacyRigBuildResult):
        raise TypeError("rig must be LegacyRigBuildResult")

    resolved_profile = resolve_a1_rig_profile(rig.profile.profile_id)
    if resolved_profile is not A1RigProfile.TWO_AXIS_ROTATION_SCALE:
        raise ValueError(
            "Preprojected screen setup requires TWO_AXIS_ROTATION_SCALE"
        )

    if rig.request.setup_pose_mode is A1RigSetupPoseMode.PREPROJECTED_SCREEN:
        rig.validate()
        return rig

    rebuilt = build_two_axis_scale_rig(
        replace(
            rig.request,
            setup_pose_mode=A1RigSetupPoseMode.PREPROJECTED_SCREEN,
        )
    )
    rebuilt.validate()
    return rebuilt


__all__ = ["ensure_preprojected_screen_rig"]
