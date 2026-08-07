from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from Blender_to_Spine2D_Mesh_Exporter import rig_ui, ui
from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1GeometryPreparationSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_ui_scene_capture import (
    _SceneExportProfile,
    _resolve_projection_direction,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_ui_selection import (
    _ObjectExportProfile,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_ui_settings import (
    _effective_projection_direction,
    _settings_from_profiles,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    A1TextureExportMode,
    BakeExecutionSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.projection import (
    A1ProjectionDirection,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_profiles import (
    A1RigProfile,
    A1RigSetupPoseMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import (
    DEFAULT_SPINE_JSON_TARGET,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"


def _scene_profile(
    direction: A1ProjectionDirection,
    *,
    texture_mode: A1TextureExportMode = A1TextureExportMode.NORMAL_UV_SEGMENTS,
) -> _SceneExportProfile:
    return _SceneExportProfile(
        output_directory=Path("output"),
        images_relative_path="images",
        texture_size=256,
        seam_mode="AUTO",
        angle_limit_degrees=30.0,
        geometry=A1GeometryPreparationSettings(),
        bake_execution=BakeExecutionSettings(
            texture_export_mode=texture_mode,
        ),
        include_control_icons=False,
        include_preview_animation=False,
        spine_target=DEFAULT_SPINE_JSON_TARGET,
        rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE,
        texture_export_mode=texture_mode,
        projection_direction=direction,
    )


def _object_profile() -> _ObjectExportProfile:
    return _ObjectExportProfile(
        source_object=object(),
        object_name="ProjectionObject",
        sequence_start_frame=0,
        sequence_frame_count=0,
        connect_enabled=False,
    )


def test_scene_profile_appends_projection_direction_after_texture_mode() -> None:
    names = tuple(field.name for field in fields(_SceneExportProfile))

    assert names[-2:] == ("texture_export_mode", "projection_direction")
    assert _SceneExportProfile.__dataclass_fields__["projection_direction"].default is (
        A1ProjectionDirection.POSITIVE_Z
    )


def test_scene_projection_resolver_accepts_every_stable_identifier() -> None:
    for direction in A1ProjectionDirection:
        scene = SimpleNamespace(spine2d_projection_direction=direction.value)

        assert _resolve_projection_direction(scene) is direction

    assert _resolve_projection_direction(SimpleNamespace()) is (
        A1ProjectionDirection.POSITIVE_Z
    )


@pytest.mark.parametrize("raw", ("", "CAMERA", "POSITIVE_Q", 12, None))
def test_scene_projection_resolver_fails_closed_for_invalid_values(raw: object) -> None:
    scene = SimpleNamespace(spine2d_projection_direction=raw)

    with pytest.raises((TypeError, ValueError)):
        _resolve_projection_direction(scene)


def test_normal_uv_ui_selection_is_preserved_until_application_normalization() -> None:
    source = _object_profile()

    for direction in A1ProjectionDirection:
        scene = _scene_profile(direction)

        # Scene capture owns the exact persisted UI choice. The application-facing
        # geometry route deliberately collapses both Active Camera root modes onto the
        # same camera projection while carrying Camera Root ownership through setup mode.
        assert scene.projection_direction is direction

        expected_geometry_direction = (
            A1ProjectionDirection.ACTIVE_CAMERA
            if direction.camera_root
            else direction
        )
        assert _effective_projection_direction(scene) is expected_geometry_direction

        settings = _settings_from_profiles(source, scene)
        assert settings.projection_direction is expected_geometry_direction
        if direction.camera_root:
            assert settings.rig_setup_pose_mode is A1RigSetupPoseMode.PREPROJECTED_SCREEN
        else:
            assert settings.rig_setup_pose_mode is A1RigSetupPoseMode.PRESERVE_COMPOSITION


@pytest.mark.parametrize(
    ("texture_mode", "expected_direction"),
    (
        (
            A1TextureExportMode.CAMERA_PROJECTION,
            A1ProjectionDirection.POSITIVE_Z,
        ),
        (
            A1TextureExportMode.DEPTH_CAMERA_PROJECTION,
            A1ProjectionDirection.ACTIVE_CAMERA,
        ),
    ),
)
def test_rendered_camera_modes_use_their_explicit_geometry_projection_contract(
    texture_mode: A1TextureExportMode,
    expected_direction: A1ProjectionDirection,
) -> None:
    source = _object_profile()
    scene = _scene_profile(
        A1ProjectionDirection.NEGATIVE_X,
        texture_mode=texture_mode,
    )

    assert _effective_projection_direction(scene) is expected_direction
    assert _settings_from_profiles(source, scene).projection_direction is (
        expected_direction
    )


def test_scene_rna_exposes_exact_public_projection_identifiers() -> None:
    source = (
        PACKAGE / "blender_adapter" / "scene_properties.py"
    ).read_text(encoding="utf-8")

    assert '"spine2d_projection_direction"' in source
    assert "items=projection_direction_rna_enum_items()" in source
    assert "default=A1ProjectionDirection.POSITIVE_Z.value" in source
    assert "update=_update_projection_direction" in source
    for direction in A1ProjectionDirection:
        assert direction.name in source or direction.value in source


def test_public_panel_draws_projection_only_for_normal_uv_and_reset_restores_z() -> None:
    source = (PACKAGE / "rig_ui.py").read_text(encoding="utf-8")

    assert '"spine2d_projection_direction"' in source
    assert "_draw_projection_direction(layout, scene)" in source
    assert "A1TextureExportMode.CAMERA_PROJECTION.value" in source
    assert "direction.active_camera" in source
    assert "direction.camera_root" in source
    assert (
        "context.scene.spine2d_projection_direction = (" in source
        and "A1ProjectionDirection.POSITIVE_Z.value" in source
    )


def test_projection_reset_delegates_base_reset_then_restores_positive_z(
    monkeypatch,
) -> None:
    scene = SimpleNamespace(
        spine2d_projection_direction=A1ProjectionDirection.NEGATIVE_X.value,
    )
    context = SimpleNamespace(scene=scene)
    calls: list[object] = []

    def execute_base(_self, resolved_context):
        calls.append(resolved_context)
        return {"FINISHED"}

    monkeypatch.setattr(ui.SPINE2D_OT_ResetSettings, "execute", execute_base)
    operator = rig_ui.SPINE2D_OT_ResetSettingsWithProjection()
    operator.report = MagicMock()

    result = operator.execute(context)

    assert result == {"FINISHED"}
    assert calls == [context]
    assert scene.spine2d_projection_direction == A1ProjectionDirection.POSITIVE_Z.value
    operator.report.assert_not_called()


def test_projection_reset_preserves_base_cancellation(monkeypatch) -> None:
    scene = SimpleNamespace(
        spine2d_projection_direction=A1ProjectionDirection.NEGATIVE_X.value,
    )
    context = SimpleNamespace(scene=scene)
    monkeypatch.setattr(
        ui.SPINE2D_OT_ResetSettings,
        "execute",
        lambda _self, _context: {"CANCELLED"},
    )
    operator = rig_ui.SPINE2D_OT_ResetSettingsWithProjection()
    operator.report = MagicMock()

    result = operator.execute(context)

    assert result == {"CANCELLED"}
    assert scene.spine2d_projection_direction == A1ProjectionDirection.NEGATIVE_X.value
    operator.report.assert_not_called()


def test_depth_feature_uses_schema_eight_and_keeps_public_multi_standalone() -> None:
    migration = (
        PACKAGE / "blender_adapter" / "scene_settings_migration.py"
    ).read_text(encoding="utf-8")
    planner = (
        PACKAGE / "blender_adapter" / "a1_ui_export_plan.py"
    ).read_text(encoding="utf-8")

    assert "CURRENT_SETTINGS_SCHEMA_VERSION = 8" in migration
    assert '("spine2d_depth_parallax_horizon_angle", 0.0)' in migration
    assert "mode=A1MultiObjectMode.STANDALONE" in planner
    assert "build_development_connected_ui_export_plan" not in planner
