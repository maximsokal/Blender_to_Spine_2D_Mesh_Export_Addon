from pathlib import Path
from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_ui_scene_capture import (
    _capture_scene_profile,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.production_shader_capability_routing import (
    normal_mode_camera_requirement_message,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    A1TextureExportMode,
    BakeExecutionSettings,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"
ADAPTER = PACKAGE / "blender_adapter"


def _scene(mode: str, *, render_engine: str = "BLENDER_EEVEE"):
    return SimpleNamespace(
        spine2d_texture_export_mode=mode,
        spine2d_seam_maker_mode="AUTO",
        spine2d_angle_limit=30.0,
        spine2d_angular_mode="SEED_CONE",
        spine2d_local_angle_limit=30.0,
        spine2d_projection_alpha_threshold=1.0 / 255.0,
        spine2d_depth_smoothing=0.35,
        spine2d_depth_edge_threshold=0.08,
        spine2d_depth_mesh_error_pixels=4.0,
        spine2d_depth_max_points=128,
        spine2d_depth_base_mode="FARTHEST_VISIBLE",
        spine2d_control_icons=True,
        spine2d_export_preview_animation=True,
        spine2d_material_source_policy="REQUIRE_SOURCE",
        spine2d_generated_material_pattern="SOLID_GRAY",
        spine2d_generated_gray_color=(0.5, 0.5, 0.5),
        render=SimpleNamespace(engine=render_engine),
    )


def test_texture_export_mode_default_is_normal_uv_segments():
    settings = BakeExecutionSettings()

    assert (
        settings.texture_export_mode
        is A1TextureExportMode.NORMAL_UV_SEGMENTS
    )


def test_texture_export_mode_rejects_untyped_strings():
    with pytest.raises(TypeError, match="texture_export_mode"):
        BakeExecutionSettings(texture_export_mode="CAMERA_PROJECTION")


@pytest.mark.parametrize(
    "mode",
    (
        A1TextureExportMode.NORMAL_UV_SEGMENTS,
        A1TextureExportMode.CAMERA_PROJECTION,
        A1TextureExportMode.DEPTH_CAMERA_PROJECTION,
    ),
)
def test_scene_profile_captures_one_typed_mode_for_all_downstream_settings(
    tmp_path: Path,
    mode: A1TextureExportMode,
):
    profile = _capture_scene_profile(
        _scene(mode.value),
        output_directory=tmp_path,
        texture_size=128,
        images_relative_path="images",
    )

    assert profile.texture_export_mode is mode
    assert profile.bake_execution.texture_export_mode is mode
    assert profile.bake_execution.render_engine == "BLENDER_EEVEE"


def test_ui_exposes_exact_three_mode_labels_and_reset_default():
    ui_source = (PACKAGE / "ui.py").read_text(encoding="utf-8")
    property_source = (ADAPTER / "scene_properties.py").read_text(encoding="utf-8")

    assert '"spine2d_texture_export_mode"' in property_source
    assert '"Normal / UV Segments"' in property_source
    assert '"Camera Projection"' in property_source
    assert '"Depth Camera Projection"' in property_source
    assert "A1TextureExportMode.NORMAL_UV_SEGMENTS.value" in ui_source
    assert 'text="Export mode"' in ui_source


def test_normal_mode_runtime_temporarily_resolves_eevee_to_cycles():
    source = (ADAPTER / "semantic_bake_validation.py").read_text(
        encoding="utf-8"
    )

    assert "requested_renderer.uses_eevee" in source
    assert 'replace(resolved_settings, render_engine="CYCLES")' in source
    assert "Camera Projection never reaches this semantic-bake path" in source


def test_routing_no_longer_uses_eevee_as_camera_projection_switch():
    source = (ADAPTER / "production_shader_capability_routing.py").read_text(
        encoding="utf-8"
    )
    message = normal_mode_camera_requirement_message(())

    assert "renderer.uses_eevee or" not in source
    assert "_CAMERA_RENDER_MODES" in source
    assert "A1TextureExportMode.CAMERA_PROJECTION" in source
    assert "A1TextureExportMode.DEPTH_CAMERA_PROJECTION" in source
    assert "texture_export_mode in _CAMERA_RENDER_MODES" in source
    assert (
        "Select Export Mode: Camera Projection or Depth Camera Projection"
        in message
    )
