from __future__ import annotations

from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"
MANIFEST = PACKAGE / "blender_manifest.toml"
SCENE_PROPERTIES = PACKAGE / "blender_adapter" / "scene_properties.py"
SCENE_MIGRATION = PACKAGE / "blender_adapter" / "scene_settings_migration.py"
SCENE_CAPTURE = PACKAGE / "blender_adapter" / "a1_ui_scene_capture.py"
DEPTH_GEOMETRY = PACKAGE / "application" / "a1_depth_camera_geometry_preparation.py"
DEPTH_FINALIZATION = PACKAGE / "application" / "a1_depth_camera_projection_finalization.py"
OUTPUT_DISPATCH = PACKAGE / "blender_adapter" / "a1_output_dispatch.py"
PROXY = PACKAGE / "blender_adapter" / "depth_camera_projection_subject.py"
EXECUTION = PACKAGE / "blender_adapter" / "a1_camera_projection_execution.py"
RIG_UI = PACKAGE / "rig_ui.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_depth_camera_projection_scene_schema_contains_public_controls() -> None:
    source = _read(SCENE_PROPERTIES)

    required = (
        "spine2d_depth_smoothing",
        "spine2d_depth_edge_threshold",
        "spine2d_depth_mesh_error_px",
        "spine2d_depth_max_points",
        "spine2d_depth_base_mode",
        "spine2d_depth_parallax_horizon_angle",
    )
    for field_name in required:
        assert field_name in source


def test_depth_camera_projection_ui_exposes_current_controls_only() -> None:
    source = _read(RIG_UI)

    required = (
        'prop(scene, "spine2d_depth_smoothing")',
        'prop(scene, "spine2d_depth_edge_threshold")',
        'prop(scene, "spine2d_depth_mesh_error_px")',
        'prop(scene, "spine2d_depth_max_points")',
        'prop(scene, "spine2d_depth_parallax_horizon_angle")',
    )
    for fragment in required:
        assert fragment in source

    assert 'prop(scene, "spine2d_depth_base_mode")' not in source


def test_depth_geometry_keeps_generated_topology_bounded_by_source_vertices() -> None:
    source = _read(DEPTH_GEOMETRY)

    assert "source_vertex_budget" in source
    assert "min(" in source
    assert "settings.max_points" in source


def test_depth_crop_finalization_preserves_weighted_topology() -> None:
    source = _read(DEPTH_FINALIZATION)
    dispatch = _read(OUTPUT_DISPATCH)

    assert "component.attachment.triangles != attachment.triangles" in source
    assert "component.attachment.vertices != attachment.vertices" in source
    assert "component.attachment.hull != attachment.hull" in source
    assert "_crop_uv(" in source
    assert "finalize_prepared_depth_camera_projection(" in dispatch
    assert "finalize_prepared_camera_projection(" in dispatch


def test_depth_camera_projection_proxy_uses_evaluated_geometry_and_restores_state() -> None:
    proxy = _read(PROXY)
    execution = _read(EXECUTION)

    assert "meshes.new_from_object(" in proxy
    assert "evaluated.matrix_world" in proxy
    assert "animation_data_clear" in proxy
    assert "constraints" in proxy
    assert "modifiers" in proxy
    assert "scene.camera = proxy" in proxy
    assert "finally:" in proxy
    assert "frozen_depth_camera_projection_subject(runtime)" in execution
    assert "nullcontext(runtime.source_object)" in execution
    assert "A1TextureExportMode.DEPTH_CAMERA_PROJECTION" in execution


def test_scene_schema_and_manifest_are_current_release() -> None:
    manifest = tomllib.loads(_read(MANIFEST))
    migration = _read(SCENE_MIGRATION)

    assert manifest["version"] == "0.150.0"
    assert manifest["blender_version_min"] == "5.2.0"
    assert "CURRENT_SETTINGS_SCHEMA_VERSION = 8" in migration
    assert "_initialize_depth_defaults(" in migration
    assert "spine2d_depth_base_mode" in migration
    assert "spine2d_depth_parallax_horizon_angle" in migration


def test_scene_capture_owns_depth_settings_in_immutable_request() -> None:
    source = _read(SCENE_CAPTURE)

    assert "def _resolve_depth_projection_settings(" in source
    assert "DepthCameraProjectionSettings(" in source
    assert "depth_projection=_resolve_depth_projection_settings(scene)" in source
    assert "def _resolve_depth_parallax_settings(" in source
    assert "depth_parallax=_resolve_depth_parallax_settings(scene)" in source
