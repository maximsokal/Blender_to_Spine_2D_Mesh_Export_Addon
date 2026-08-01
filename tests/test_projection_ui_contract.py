from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest

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


def test_normal_uv_settings_preserve_every_selected_projection_direction() -> None:
    source = _object_profile()

    for direction in A1ProjectionDirection:
        scene = _scene_profile(direction)

        assert _effective_projection_direction(scene) is direction
        settings = _settings_from_profiles(source, scene)
        assert settings.projection_direction is direction


def test_rendered_camera_projection_cannot_enter_object_bake_camera_route() -> None:
    source = _object_profile()
    scene = _scene_profile(
        A1ProjectionDirection.ACTIVE_CAMERA,
        texture_mode=A1TextureExportMode.CAMERA_PROJECTION,
    )

    assert _effective_projection_direction(scene) is A1ProjectionDirection.POSITIVE_Z
    assert _settings_from_profiles(source, scene).projection_direction is (
        A1ProjectionDirection.POSITIVE_Z
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
    assert "A1ProjectionDirection.ACTIVE_CAMERA" in source
    assert (
        "context.scene.spine2d_projection_direction = (" in source
        and "A1ProjectionDirection.POSITIVE_Z.value" in source
    )


def test_slice_six_does_not_change_scene_schema_or_public_multi_mode() -> None:
    migration = (
        PACKAGE / "blender_adapter" / "scene_settings_migration.py"
    ).read_text(encoding="utf-8")
    planner = (
        PACKAGE / "blender_adapter" / "a1_ui_export_plan.py"
    ).read_text(encoding="utf-8")

    assert "CURRENT_SETTINGS_SCHEMA_VERSION = 6" in migration
    assert "mode=A1MultiObjectMode.STANDALONE" in planner
    assert "build_development_connected_ui_export_plan" not in planner
