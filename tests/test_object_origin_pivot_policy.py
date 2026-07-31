"""Contracts for Blender Object Origin based two-axis object-bake placement."""

from __future__ import annotations

from math import copysign
from pathlib import Path
from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1GeometryPreparationSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_document_preparation import (
    _resolve_z_group_origin_mode,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_ui_export_plan import (
    _single_setup_pose_mode,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_ui_scene_capture import (
    _SceneExportProfile,
    _resolve_rig_profile,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    A1TextureExportMode,
    BakeExecutionSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.legacy_profile import (
    LegacyRigProfile,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.legacy_rig_contracts import (
    LegacyRigBuildRequest,
    LegacyZGroup,
    LegacyZGroupOriginMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.legacy_rig_plan import (
    build_legacy_z_group_metadata,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_profiles import (
    A1RigProfile,
    A1RigSetupPoseMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.two_axis_scale_rig import (
    build_two_axis_scale_rig,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import (
    SpineJsonTarget,
)


def _request(
    z_values: tuple[float, ...],
    *,
    mode: LegacyZGroupOriginMode,
    main_position_pixels: tuple[float, float] = (0.0, 0.0),
    groups: tuple[LegacyZGroup, ...] | None = None,
) -> LegacyRigBuildRequest:
    return LegacyRigBuildRequest(
        prefix="Cone",
        texture_width=100,
        texture_height=100,
        z_groups=(
            groups
            if groups is not None
            else tuple(LegacyZGroup(value) for value in z_values)
        ),
        main_position_pixels=main_position_pixels,
        setup_pose_mode=A1RigSetupPoseMode.PRESERVE_COMPOSITION,
        z_group_origin_mode=mode,
    )


def _offsets(
    request: LegacyRigBuildRequest,
    *,
    uniform_scale: float = 10.0,
) -> tuple[float, ...]:
    metadata = build_legacy_z_group_metadata(
        request,
        LegacyRigProfile(),
        uniform_scale,
    )
    return tuple(item.y_offset_pixels for item in metadata)


@pytest.mark.parametrize(
    ("z_values", "expected"),
    (
        ((-2.0, 0.0, 3.0), (-20.0, 0.0, 30.0)),
        ((1.0, 2.0, 4.0), (10.0, 20.0, 40.0)),
        ((-4.0, -2.0, -1.0), (-40.0, -20.0, -10.0)),
    ),
)
def test_object_origin_preserves_signed_local_z_offsets(
    z_values: tuple[float, ...],
    expected: tuple[float, ...],
) -> None:
    request = _request(
        z_values,
        mode=LegacyZGroupOriginMode.OBJECT_ORIGIN,
    )

    assert _offsets(request) == expected


def test_minimum_z_mode_preserves_compatibility_behavior() -> None:
    request = _request(
        (-2.0, 0.0, 3.0),
        mode=LegacyZGroupOriginMode.MINIMUM_Z,
    )

    assert _offsets(request) == (0.0, 20.0, 50.0)


def test_height_override_remains_an_absolute_spine_offset() -> None:
    request = _request(
        (),
        mode=LegacyZGroupOriginMode.OBJECT_ORIGIN,
        groups=(
            LegacyZGroup(-2.0, height_real_pixels=17.25),
            LegacyZGroup(3.0),
        ),
    )

    assert _offsets(request) == (17.25, 30.0)


def test_object_origin_normalizes_rounded_negative_zero() -> None:
    request = _request(
        (-0.0001,),
        mode=LegacyZGroupOriginMode.OBJECT_ORIGIN,
    )

    value = _offsets(request, uniform_scale=1.0)[0]
    assert value == 0.0
    assert copysign(1.0, value) == 1.0


def test_request_rejects_untyped_origin_mode() -> None:
    with pytest.raises(TypeError, match="LegacyZGroupOriginMode"):
        LegacyRigBuildRequest(
            prefix="Cone",
            texture_width=100,
            texture_height=100,
            z_groups=(LegacyZGroup(0.0),),
            z_group_origin_mode="OBJECT_ORIGIN",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("camera_projection", "rig_profile", "expected"),
    (
        (
            False,
            A1RigProfile.TWO_AXIS_ROTATION_SCALE,
            LegacyZGroupOriginMode.OBJECT_ORIGIN,
        ),
        (
            True,
            A1RigProfile.TWO_AXIS_ROTATION_SCALE,
            LegacyZGroupOriginMode.MINIMUM_Z,
        ),
        (
            False,
            A1RigProfile.THREE_AXIS_ROTATION,
            LegacyZGroupOriginMode.MINIMUM_Z,
        ),
    ),
)
def test_document_route_selects_only_approved_object_origin_policy(
    camera_projection: bool,
    rig_profile: A1RigProfile,
    expected: LegacyZGroupOriginMode,
) -> None:
    assert (
        _resolve_z_group_origin_mode(
            camera_projection=camera_projection,
            rig_profile=rig_profile,
        )
        is expected
    )


def _scene_profile(mode: A1TextureExportMode) -> _SceneExportProfile:
    return _SceneExportProfile(
        output_directory=Path("object-origin-output"),
        images_relative_path="images",
        texture_size=1024,
        seam_mode="AUTO",
        angle_limit_degrees=30.0,
        geometry=A1GeometryPreparationSettings(),
        bake_execution=BakeExecutionSettings(texture_export_mode=mode),
        include_control_icons=False,
        include_preview_animation=False,
        spine_target=SpineJsonTarget.SPINE_4_2,
        rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE,
        texture_export_mode=mode,
    )


def test_normal_single_export_keeps_main_bone_at_object_origin() -> None:
    assert (
        _single_setup_pose_mode(
            _scene_profile(A1TextureExportMode.NORMAL_UV_SEGMENTS)
        )
        is A1RigSetupPoseMode.PRESERVE_COMPOSITION
    )


def test_camera_single_export_keeps_previous_normalized_setup() -> None:
    assert (
        _single_setup_pose_mode(
            _scene_profile(A1TextureExportMode.CAMERA_PROJECTION)
        )
        is A1RigSetupPoseMode.NORMALIZED_SINGLE
    )


def test_public_scene_capture_normalizes_hidden_three_axis_profile() -> None:
    scene = SimpleNamespace(
        spine2d_rig_profile=A1RigProfile.THREE_AXIS_ROTATION.value
    )

    assert _resolve_rig_profile(scene) is A1RigProfile.TWO_AXIS_ROTATION_SCALE


def test_public_scene_capture_defaults_to_two_axis_profile() -> None:
    assert _resolve_rig_profile(SimpleNamespace()) is (
        A1RigProfile.TWO_AXIS_ROTATION_SCALE
    )


def test_two_axis_rig_places_main_and_depth_bones_around_authored_origin() -> None:
    request = _request(
        (-1.0, 0.0, 2.0),
        mode=LegacyZGroupOriginMode.OBJECT_ORIGIN,
        main_position_pixels=(125.0, -75.0),
    )

    result = build_two_axis_scale_rig(request)
    bones = {bone.name: bone for bone in result.bones}
    main = bones[result.info.main_bone_name]

    assert (main.x, main.y) == (125.0, -75.0)
    assert tuple(
        bones[group.scale_bone_name].y for group in result.info.z_groups
    ) == (-100.0, 0.0, 200.0)
