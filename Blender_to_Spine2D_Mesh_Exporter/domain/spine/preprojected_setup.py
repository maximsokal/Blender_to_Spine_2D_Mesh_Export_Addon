"""Rebuild canonical two-axis rigs for already-projected camera geometry."""

from __future__ import annotations

from dataclasses import replace
from math import isfinite

from .legacy_rig_contracts import (
    LegacyRigBuildResult,
    LegacyZGroup,
    LegacyZGroupOriginMode,
)
from .rig_profiles import (
    A1RigProfile,
    A1RigSetupPoseMode,
    resolve_a1_rig_profile,
)
from .two_axis_scale_rig import build_two_axis_scale_rig


def _finite_camera_depth(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("camera_depth must be a finite number")
    resolved = float(value)
    if not isfinite(resolved):
        raise ValueError("camera_depth must be finite")
    return 0.0 if resolved == 0.0 else resolved


def ensure_preprojected_screen_rig(
    rig: LegacyRigBuildResult,
    *,
    main_position_pixels: tuple[float, float] | None = None,
    camera_depth: float | None = None,
) -> LegacyRigBuildResult:
    """Return one validated rigid layer positioned relative to camera-space zero.

    Active Camera Normal / UV Segments and rendered Camera Projection already converted
    source geometry into final camera-screen X/Y. They must not keep per-vertex depth
    groups: doing so applies different X/Y transforms to vertices of the same object and
    visibly deforms the mesh.

    The camera-relative contract is therefore:

    * one depth group for the complete object;
    * ``main`` at camera-space zero;
    * projected Blender Object Origin stored on the internal base layer;
    * vertex bones local to that rigid object layer;
    * live X/Y/Scale constraints retained.

    ``camera_depth`` is required when an existing non-camera rig must be collapsed during
    rendered Camera Projection finalization. Active Camera preparation already supplies a
    single Object-Origin-depth group and can omit it.
    """

    if not isinstance(rig, LegacyRigBuildResult):
        raise TypeError("rig must be LegacyRigBuildResult")

    resolved_profile = resolve_a1_rig_profile(rig.profile.profile_id)
    if resolved_profile is not A1RigProfile.TWO_AXIS_ROTATION_SCALE:
        raise ValueError(
            "Preprojected screen setup requires TWO_AXIS_ROTATION_SCALE"
        )

    request = rig.request
    if camera_depth is None:
        if len(request.z_groups) != 1:
            raise ValueError(
                "Camera-relative preprojected setup requires exactly one Object-Origin "
                f"depth group; received {len(request.z_groups)}"
            )
        resolved_z_groups = request.z_groups
    else:
        resolved_z_groups = (
            LegacyZGroup(z_value=_finite_camera_depth(camera_depth)),
        )

    resolved_main_position = (
        request.main_position_pixels
        if main_position_pixels is None
        else main_position_pixels
    )
    resolved_request = replace(
        request,
        z_groups=resolved_z_groups,
        main_position_pixels=resolved_main_position,
        setup_pose_mode=A1RigSetupPoseMode.PREPROJECTED_SCREEN,
        z_group_origin_mode=LegacyZGroupOriginMode.OBJECT_ORIGIN,
    )

    if resolved_request == request:
        rig.validate()
        return rig

    rebuilt = build_two_axis_scale_rig(resolved_request)
    rebuilt.validate()
    return rebuilt


__all__ = ["ensure_preprojected_screen_rig"]
