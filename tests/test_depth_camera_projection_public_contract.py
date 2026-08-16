"""Public and architectural contracts for the current Depth Camera Projection mode."""

from __future__ import annotations

import ast
from pathlib import Path
import tomllib

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1GeometryPreparationSettings,
    A1SourceGeometryMode,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_depth_document_preparation import (
    _resolve_depth_z_group_origin_mode,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_ui_scene_capture import (
    _SceneExportProfile,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_ui_settings import (
    _effective_projection_direction,
    _effective_source_geometry_mode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    A1TextureExportMode,
    BakeExecutionSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import DepthProjectionBaseMode
from Blender_to_Spine2D_Mesh_Exporter.domain.projection import A1ProjectionDirection
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    A1RigProfile,
    SpineJsonTarget,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.legacy_rig_contracts import (
    LegacyZGroupOriginMode,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"
MANIFEST = PACKAGE / "blender_manifest.toml"
SCENE_PROPERTIES = PACKAGE / "blender_adapter" / "scene_properties.py"
SCENE_CAPTURE = PACKAGE / "blender_adapter" / "a1_ui_scene_capture.py"
OBJECT_PREPARATION = PACKAGE / "blender_adapter" / "a1_object_preparation.py"
DEPTH_SOURCE = PACKAGE / "blender_adapter" / "a1_depth_source_geometry_preparation.py"
DEPTH_DOCUMENT = PACKAGE / "blender_adapter" / "a1_depth_document_preparation.py"
DEPTH_ASSEMBLY = PACKAGE / "blender_adapter" / "a1_depth_document_assembly.py"
DEPTH_PROJECTOR = PACKAGE / "application" / "a1_depth_attachment_projection.py"
DEPTH_DISTANCE = PACKAGE / "domain" / "geometry" / "depth_camera_distance.py"
DEPTH_RENDER_PROXY = (
    PACKAGE / "blender_adapter" / "depth_camera_projection_render_proxy.py"
)
CAMERA_EXECUTION = PACKAGE / "blender_adapter" / "camera_projection_execution.py"
DEPTH_FINALIZATION = PACKAGE / "blender_adapter" / "a1_depth_projection_finalization.py"
OUTPUT_DISPATCH = PACKAGE / "blender_adapter" / "a1_rendered_projection_finalization.py"
SCENE_MIGRATION = PACKAGE / "blender_adapter" / "scene_settings_migration.py"
UI = PACKAGE / "ui.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _profile(mode: A1TextureExportMode) -> _SceneExportProfile:
    return _SceneExportProfile(
        output_directory=ROOT,
        images_relative_path="images",
        texture_size=128,
        seam_mode="AUTO",
        angle_limit_degrees=30.0,
        geometry=A1GeometryPreparationSettings(),
        bake_execution=BakeExecutionSettings(texture_export_mode=mode),
        include_control_icons=False,
        include_preview_animation=False,
        spine_target=SpineJsonTarget.SPINE_4_2,
        rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE,
        texture_export_mode=mode,
        projection_direction=A1ProjectionDirection.NEGATIVE_X,
    )


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


def _attribute_path(node: ast.AST) -> tuple[str, ...] | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return tuple(reversed(parts))


def _depth_z_group_source_path() -> tuple[str, ...]:
    tree = ast.parse(_read(DEPTH_SOURCE), filename=DEPTH_SOURCE.name)
    public = _function(tree, "prepare_a1_depth_source_geometry")
    calls = tuple(
        node
        for node in ast.walk(public)
        if isinstance(node, ast.Call)
        and _call_name(node) == "build_a1_z_group_assignment"
    )
    assert len(calls) == 1
    assert len(calls[0].args) == 1
    path = _attribute_path(calls[0].args[0])
    assert path is not None
    return path


def test_public_texture_mode_enum_contains_exactly_three_choices() -> None:
    assert tuple(mode.value for mode in A1TextureExportMode) == (
        "NORMAL_UV_SEGMENTS",
        "CAMERA_PROJECTION",
        "DEPTH_CAMERA_PROJECTION",
    )


def test_ui_exposes_exactly_the_three_requested_user_labels() -> None:
    source = _read(SCENE_PROPERTIES)
    labels = (
        '"Normal / UV Segments"',
        '"Camera Projection"',
        '"Depth Camera Projection"',
    )
    assert all(label in source for label in labels)
    assert source.count("A1TextureExportMode.NORMAL_UV_SEGMENTS.value") >= 2
    assert source.count("A1TextureExportMode.CAMERA_PROJECTION.value") >= 1
    assert source.count("A1TextureExportMode.DEPTH_CAMERA_PROJECTION.value") >= 1


def test_depth_mode_forces_evaluated_active_camera_without_changing_old_modes() -> None:
    normal = _profile(A1TextureExportMode.NORMAL_UV_SEGMENTS)
    flat = _profile(A1TextureExportMode.CAMERA_PROJECTION)
    depth = _profile(A1TextureExportMode.DEPTH_CAMERA_PROJECTION)

    assert _effective_source_geometry_mode(normal) is A1SourceGeometryMode.ORIGINAL
    assert _effective_projection_direction(normal) is A1ProjectionDirection.NEGATIVE_X
    assert _effective_source_geometry_mode(flat) is A1SourceGeometryMode.ORIGINAL
    assert _effective_projection_direction(flat) is A1ProjectionDirection.POSITIVE_Z
    assert _effective_source_geometry_mode(depth) is A1SourceGeometryMode.EVALUATED
    assert _effective_projection_direction(depth) is A1ProjectionDirection.ACTIVE_CAMERA


def test_depth_base_supports_both_policies_but_ui_uses_farthest_visible() -> None:
    assert tuple(mode.value for mode in DepthProjectionBaseMode) == (
        "FARTHEST_VISIBLE",
        "OBJECT_ORIGIN",
    )
    properties = _read(SCENE_PROPERTIES)
    ui = _read(UI)
    assert 'options={"HIDDEN"}' in properties
    assert "DepthProjectionBaseMode.FARTHEST_VISIBLE.value" in properties
    assert "DepthProjectionBaseMode.OBJECT_ORIGIN.value" in properties
    assert "Depth base: Farthest visible point" in ui
    assert 'column.prop(scene, "spine2d_depth_base_mode"' not in ui


def test_both_relief_base_policies_keep_camera_as_global_rig_zero() -> None:
    assert _resolve_depth_z_group_origin_mode(
        DepthProjectionBaseMode.FARTHEST_VISIBLE
    ) is LegacyZGroupOriginMode.OBJECT_ORIGIN
    assert _resolve_depth_z_group_origin_mode(
        DepthProjectionBaseMode.OBJECT_ORIGIN
    ) is LegacyZGroupOriginMode.OBJECT_ORIGIN


def test_depth_ui_contains_depth_quality_and_parallax_controls() -> None:
    properties = _read(SCENE_PROPERTIES)
    ui = _read(UI)
    for property_name in (
        "spine2d_depth_smoothing",
        "spine2d_depth_edge_threshold",
        "spine2d_depth_mesh_error_pixels",
        "spine2d_depth_max_points",
        "spine2d_depth_parallax_horizon_angle",
    ):
        assert property_name in properties
    assert "One generated vertex bone per retained depth point" in ui


def test_depth_route_uses_shared_camera_distance_and_compensated_view_attachments() -> None:
    preparation = _read(OBJECT_PREPARATION)
    source = _read(DEPTH_SOURCE)
    document = _read(DEPTH_DOCUMENT)
    assembly = _read(DEPTH_ASSEMBLY)
    projector = _read(DEPTH_PROJECTOR)
    distance = _read(DEPTH_DISTANCE)

    assert "prepare_a1_depth_source_geometry(" in preparation
    assert "prepare_a1_depth_document(" in preparation
    assert "CameraProjectionPlan" in preparation
    assert "build_depth_camera_projection_surface(" in source
    assert "build_depth_parallax_geometry_package(" in source
    assert "_package_to_camera_distance(" in source
    assert "convert_depth_snapshot_to_camera_distance(" in source
    assert _depth_z_group_source_path() == (
        "projection_stage",
        "depth_package",
        "union_snapshot",
    )
    assert "positive distances from shared camera zero" in source
    assert "LegacyZGroupOriginMode.OBJECT_ORIGIN" in document
    assert "LegacyZGroupOriginMode.MINIMUM_Z" not in document
    assert "expected_attachment_count = package.attachment_count" in document
    assert "component_count != expected_attachment_count" in document
    assert "assemble_and_finalize_a1_depth_document(" in document
    assert "for surface in package.reserve_surfaces" in assembly
    assert "front_name = rig.profile.segment_slot(source.prefix, 0)" in assembly
    assert "project_depth_camera_attachment(" in assembly
    assert "setup_y - parent_y" in projector
    assert "camera-distance conversion changed projected X/Y" in distance


def test_depth_sequence_uses_frozen_source_and_camera_proxies_only_for_depth_mode() -> None:
    proxy = _read(DEPTH_RENDER_PROXY)
    execution = _read(CAMERA_EXECUTION)

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


def test_depth_crop_finalization_preserves_weighted_topology() -> None:
    source = _read(DEPTH_FINALIZATION)
    dispatch = _read(OUTPUT_DISPATCH)

    assert "component.attachment.triangles != attachment.triangles" in source
    assert "component.attachment.vertices != attachment.vertices" in source
    assert "component.attachment.hull != attachment.hull" in source
    assert "_crop_uv(" in source
    assert "finalize_prepared_depth_camera_projection(" in dispatch
    assert "finalize_prepared_camera_projection(" in dispatch


def test_scene_schema_and_manifest_are_current_release() -> None:
    manifest = tomllib.loads(_read(MANIFEST))
    migration = _read(SCENE_MIGRATION)

    assert manifest["version"] == "0.152.0"
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
