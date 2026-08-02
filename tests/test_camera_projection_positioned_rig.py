"""Pure profile routing tests for rendered Camera Projection pivot placement."""

from __future__ import annotations

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_projection_finalization import (
    _positioned_projection_rig,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    A1RigProfile,
    A1RigSetupPoseMode,
    LegacyRigBuildRequest,
    LegacyZGroup,
    SpineJsonTarget,
    build_rig,
)


def _rig(profile: A1RigProfile):
    return build_rig(
        LegacyRigBuildRequest(
            prefix="Rendered",
            texture_width=128,
            texture_height=128,
            z_groups=(LegacyZGroup(0.0),),
            main_position_pixels=None,
            setup_pose_mode=A1RigSetupPoseMode.PRESERVE_COMPOSITION,
        ),
        profile,
        spine_target=SpineJsonTarget.SPINE_4_2,
    )


def test_two_axis_rendered_projection_uses_screen_neutral_setup() -> None:
    source = _rig(A1RigProfile.TWO_AXIS_ROTATION_SCALE)

    result = _positioned_projection_rig(source, (14.5, -6.25))

    assert result is not source
    assert result.request.main_position_pixels == (14.5, -6.25)
    assert result.request.setup_pose_mode is A1RigSetupPoseMode.PREPROJECTED_SCREEN
    assert source.request.main_position_pixels is None
    assert source.request.setup_pose_mode is A1RigSetupPoseMode.PRESERVE_COMPOSITION
    result.validate()


def test_three_axis_rendered_projection_preserves_historical_setup_contract() -> None:
    source = _rig(A1RigProfile.THREE_AXIS_ROTATION)

    result = _positioned_projection_rig(source, (-9.0, 4.0))

    assert result is not source
    assert result.request.main_position_pixels == (-9.0, 4.0)
    assert result.request.setup_pose_mode is A1RigSetupPoseMode.PRESERVE_COMPOSITION
    assert result.profile.profile_id == source.profile.profile_id
    result.validate()
