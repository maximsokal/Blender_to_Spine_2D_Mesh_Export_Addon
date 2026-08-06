"""Regression contracts for the two Normal / UV Active Camera root modes."""

from __future__ import annotations

from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1GeometryPreparationSettings,
    A1SingleObjectExportSettings,
    ExportSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_document_preparation import (
    _camera_layer_projection_kind,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_ui_scene_capture import (
    _SceneExportProfile,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_ui_settings import (
    _effective_projection_direction,
    _effective_rig_setup_pose_mode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    A1TextureExportMode,
    BakeExecutionSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.camera_projection import (
    A1CameraProjectionKind,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.projection import (
    A1ProjectionDirection,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_profiles import (
    A1CameraLayerProjectionKind,
    A1RigProfile,
    A1RigSetupPoseMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import (
    SpineJsonTarget,
)


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _scene(direction: A1ProjectionDirection) -> _SceneExportProfile:
    return _SceneExportProfile(
        output_directory=Path("exports"),
        images_relative_path="images",
        texture_size=256,
        seam_mode="AUTO",
        angle_limit_degrees=30.0,
        geometry=A1GeometryPreparationSettings(),
        bake_execution=BakeExecutionSettings(
            texture_export_mode=A1TextureExportMode.NORMAL_UV_SEGMENTS,
        ),
        include_control_icons=False,
        include_preview_animation=False,
        spine_target=SpineJsonTarget.SPINE_4_2,
        rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE,
        texture_export_mode=A1TextureExportMode.NORMAL_UV_SEGMENTS,
        projection_direction=direction,
    )


def test_object_root_keeps_existing_active_camera_geometry_contract() -> None:
    scene = _scene(A1ProjectionDirection.ACTIVE_CAMERA)

    assert _effective_projection_direction(scene) is (
        A1ProjectionDirection.ACTIVE_CAMERA
    )
    assert _effective_rig_setup_pose_mode(
        scene,
        A1RigSetupPoseMode.PRESERVE_COMPOSITION,
    ) is A1RigSetupPoseMode.PRESERVE_COMPOSITION


def test_camera_root_reuses_geometry_but_selects_preprojected_rig() -> None:
    scene = _scene(A1ProjectionDirection.ACTIVE_CAMERA_CAMERA_ROOT)

    assert _effective_projection_direction(scene) is (
        A1ProjectionDirection.ACTIVE_CAMERA
    )
    assert _effective_rig_setup_pose_mode(
        scene,
        A1RigSetupPoseMode.NORMALIZED_SINGLE,
    ) is A1RigSetupPoseMode.PREPROJECTED_SCREEN


def test_direct_camera_root_settings_normalize_at_application_boundary() -> None:
    settings = A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=256,
            texture_height=256,
            output_directory=Path("exports"),
        ),
        bake_execution=BakeExecutionSettings(
            texture_export_mode=A1TextureExportMode.NORMAL_UV_SEGMENTS,
        ),
        projection_direction=A1ProjectionDirection.ACTIVE_CAMERA_CAMERA_ROOT,
    )

    assert settings.projection_direction is A1ProjectionDirection.ACTIVE_CAMERA
    assert settings.rig_setup_pose_mode is A1RigSetupPoseMode.PREPROJECTED_SCREEN


def test_direct_camera_root_rejects_non_normal_texture_route() -> None:
    with pytest.raises(ValueError, match="available only for Normal / UV Segments"):
        A1SingleObjectExportSettings(
            export=ExportSettings(
                texture_width=256,
                texture_height=256,
                output_directory=Path("exports"),
            ),
            bake_execution=BakeExecutionSettings(
                texture_export_mode=A1TextureExportMode.CAMERA_PROJECTION,
            ),
            projection_direction=A1ProjectionDirection.ACTIVE_CAMERA_CAMERA_ROOT,
        )


def test_camera_root_does_not_change_non_normal_export_setup() -> None:
    scene = _SceneExportProfile(
        output_directory=Path("exports"),
        images_relative_path="images",
        texture_size=256,
        seam_mode="AUTO",
        angle_limit_degrees=30.0,
        geometry=A1GeometryPreparationSettings(),
        bake_execution=BakeExecutionSettings(
            texture_export_mode=A1TextureExportMode.CAMERA_PROJECTION,
        ),
        include_control_icons=False,
        include_preview_animation=False,
        spine_target=SpineJsonTarget.SPINE_4_2,
        rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE,
        texture_export_mode=A1TextureExportMode.CAMERA_PROJECTION,
        projection_direction=A1ProjectionDirection.ACTIVE_CAMERA_CAMERA_ROOT,
    )

    assert _effective_projection_direction(scene) is A1ProjectionDirection.POSITIVE_Z
    assert _effective_rig_setup_pose_mode(
        scene,
        A1RigSetupPoseMode.NORMALIZED_SINGLE,
    ) is A1RigSetupPoseMode.NORMALIZED_SINGLE


def test_camera_projection_kind_maps_to_rigid_layer_kind() -> None:
    assert _camera_layer_projection_kind(
        A1CameraProjectionKind.PERSPECTIVE
    ) is A1CameraLayerProjectionKind.PERSPECTIVE
    assert _camera_layer_projection_kind(
        A1CameraProjectionKind.ORTHOGRAPHIC
    ) is A1CameraLayerProjectionKind.ORTHOGRAPHIC


def test_camera_root_document_contract_is_single_layer_and_compensated() -> None:
    source = _read(
        "Blender_to_Spine2D_Mesh_Exporter/blender_adapter/"
        "a1_document_preparation.py"
    )

    assert "def _camera_root_z_group_plan(" in source
    assert "groups=(LegacyZGroup(z_value=origin_depth),)" in source
    assert "z_group_index=group_index" in source
    assert "A1RigSetupPoseMode.PREPROJECTED_SCREEN" in source
    assert "camera_layer_projection_kind=camera_layer_kind" in source
    assert "compensate_depth_setup_y=camera_root_normal" in source
    assert '"normal_active_camera_root_mode": active_camera_root_mode' in source
    assert '"camera_relative_depth_group_count"' in source
    assert '"normal_active_camera_depth_group_count"' in source


def test_prepared_object_publishes_assembled_z_group_plan() -> None:
    source = _read(
        "Blender_to_Spine2D_Mesh_Exporter/blender_adapter/"
        "a1_object_preparation.py"
    )

    assert "prepared_z_groups = (" in source
    assert "assembly.z_groups" in source
    assert "z_groups=prepared_z_groups" in source


def test_ui_exposes_both_root_modes_without_migrating_existing_id() -> None:
    properties = _read(
        "Blender_to_Spine2D_Mesh_Exporter/blender_adapter/scene_properties.py"
    )
    ui = _read("Blender_to_Spine2D_Mesh_Exporter/rig_ui.py")

    assert 'ACTIVE_CAMERA = "ACTIVE_CAMERA"' in _read(
        "Blender_to_Spine2D_Mesh_Exporter/domain/projection.py"
    )
    assert "ACTIVE_CAMERA_CAMERA_ROOT" in properties
    assert "Object Root Bone" in properties
    assert "Camera Root Bone" in properties
    assert "direction.camera_root" in ui
    assert "Main bone pivot: active camera / camera-space zero" in ui
    assert "Main bone pivot: each object's Blender Object Origin" in ui
