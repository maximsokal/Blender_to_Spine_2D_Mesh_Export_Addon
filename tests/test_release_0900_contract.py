"""Current release-scope contracts for Depth parallax reserve 0.90.0."""

from __future__ import annotations

import ast
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"
MANIFEST = PACKAGE / "blender_manifest.toml"
RELEASE_NOTE = ROOT / "docs" / "releases" / "0.90.0.md"
SCENE_PROPERTIES = PACKAGE / "blender_adapter" / "scene_properties.py"
SCENE_CAPTURE = PACKAGE / "blender_adapter" / "a1_ui_scene_capture.py"
SCENE_MIGRATION = PACKAGE / "blender_adapter" / "scene_settings_migration.py"
RIG_UI = PACKAGE / "rig_ui.py"
UI_LAYOUT = PACKAGE / "ui_layout.py"
GEOMETRY = PACKAGE / "domain" / "geometry" / "depth_parallax.py"
CAMERA_VIEWS = PACKAGE / "blender_adapter" / "depth_parallax_camera_views.py"
CAMERA_PLAN = PACKAGE / "domain" / "baking" / "camera_projection.py"
RENDER_PROXY = (
    PACKAGE / "blender_adapter" / "depth_camera_projection_render_proxy.py"
)
DOCUMENT = PACKAGE / "blender_adapter" / "a1_depth_document_assembly.py"
DEPTH_SOURCE = (
    PACKAGE / "blender_adapter" / "a1_depth_source_geometry_preparation.py"
)
STAGING = PACKAGE / "blender_adapter" / "texture_executor.py"
FINALIZATION = (
    PACKAGE / "blender_adapter" / "a1_depth_projection_finalization.py"
)
SINGLE_OUTPUT = PACKAGE / "blender_adapter" / "a1_single_object_export.py"
MULTI_OUTPUT = PACKAGE / "blender_adapter" / "a1_output_staging.py"
MULTI_PREP = PACKAGE / "blender_adapter" / "a1_multi_object_export.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _normalized(value: str) -> str:
    return " ".join(
        value.lower().replace("-", " ").replace("`", " ").split()
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


def test_current_manifest_and_scene_schema_are_0900() -> None:
    with MANIFEST.open("rb") as stream:
        manifest = tomllib.load(stream)
    migration = _read(SCENE_MIGRATION)

    assert manifest["version"] == "0.90.0"
    assert manifest["blender_version_min"] == "5.2.0"
    assert "CURRENT_SETTINGS_SCHEMA_VERSION = 8" in migration
    assert '("spine2d_depth_parallax_horizon_angle", 0.0)' in migration
    assert "if current < 8" in migration


def test_scene_rna_stores_radians_and_cut_displays_rotation_angle() -> None:
    properties = _read(SCENE_PROPERTIES)
    capture = _read(SCENE_CAPTURE)
    rig_ui = _read(RIG_UI)
    ui_layout = _read(UI_LAYOUT)

    assert '"spine2d_depth_parallax_horizon_angle"' in properties
    assert 'name="Parallax Horizon Angle"' in properties
    assert 'subtype="ANGLE"' in properties
    assert 'unit="ROTATION"' in properties
    assert "default=0.0" in properties
    assert "max=radians(89.0)" in properties
    assert "soft_max=radians(45.0)" in properties
    assert "def _resolve_depth_parallax_settings(" in capture
    assert "DepthParallaxSettings(" in capture
    assert "depth_parallax=_resolve_depth_parallax_settings(scene)" in capture

    assert "def _draw_depth_parallax_cut_settings(" in ui_layout
    assert 'text="Parallax reserve"' in ui_layout
    assert 'text="Parallax Horizon Angle"' in ui_layout
    assert "A1TextureExportMode.DEPTH_CAMERA_PROJECTION" in ui_layout
    assert "_draw_depth_parallax_horizon" not in rig_ui
    assert 'text="Parallax reserve"' not in rig_ui
    assert "spine2d_depth_parallax_horizon_angle = 0.0" in rig_ui


def test_geometry_contract_uses_accumulated_dihedral_angle_and_union_snapshot() -> None:
    geometry = _read(GEOMETRY)

    assert "def _face_adjacency(" in geometry
    assert "def _dihedral_angle(" in geometry
    assert "def _accumulated_horizon_costs(" in geometry
    assert "heappush(" in geometry
    assert "candidate = current_cost + _dihedral_angle(" in geometry
    assert "set(costs) - set(front_faces)" in geometry
    assert "class DepthParallaxGeometryPackage" in geometry
    assert "union_snapshot" in geometry
    assert "material_index=view.material_index" in geometry
    assert "Parallax Horizon Angle exceeds Max Depth Points" in geometry


def test_virtual_texture_views_are_fitted_without_mutating_source_camera() -> None:
    views = _read(CAMERA_VIEWS)
    proxy = _read(RENDER_PROXY)

    for view_id in (
        "RIGHT",
        "UP_RIGHT",
        "UP",
        "UP_LEFT",
        "LEFT",
        "DOWN_LEFT",
        "DOWN",
        "DOWN_RIGHT",
    ):
        assert f"DepthParallaxViewId.{view_id}" in views
    assert "def _virtual_camera_world_matrix(" in views
    assert "def _fit_projection_scale(" in views
    assert "camera_world_matrix_override" in proxy
    assert "camera_data.lens = fitted" in proxy
    assert "camera_data.ortho_scale = fitted" in proxy
    assert "runtime.scene.camera = original_camera" in proxy
    assert "_remove_proxy_resources(" in proxy


def test_camera_plans_and_output_namespaces_are_view_owned() -> None:
    plan = _read(CAMERA_PLAN)
    multi = _read(MULTI_PREP)

    assert 'view_id: str = "FRONT"' in plan
    assert "camera_world_matrix_override" in plan
    assert "lens_scale: float = 1.0" in plan
    assert "def build_camera_projection_view_plan(" in plan
    assert 'f"{front_plan.settings.output_stem}_Parallax_{suffix}"' in plan
    assert "*tuple(getattr(item, \"reserve_bake_plans\", ()))" in multi
    assert "Windows output path collision" in multi


def test_reserve_attachments_share_rig_and_remain_below_front() -> None:
    document = _read(DOCUMENT)

    assert "def _reserve_slot_name(" in document
    assert 'f"{prefix}_Parallax_{surface.view.view_id.value}"' in document
    assert "for surface in package.reserve_surfaces" in document
    assert "projections.append(" in document
    assert "front_name = rig.profile.segment_slot(source.prefix, 0)" in document
    assert _depth_z_group_source_path() == (
        "projection_stage",
        "depth_package",
        "union_snapshot",
    )
    assert "optimize_shared_vertex_bones" in document


def test_each_view_owns_staging_and_crop_layout() -> None:
    staging = _read(STAGING)
    finalization = _read(FINALIZATION)
    single = _read(SINGLE_OUTPUT)
    multi = _read(MULTI_OUTPUT)

    assert "class ProjectionViewStage" in staging
    assert "reserve_projection_stages" in staging
    assert "all_reservations = primary_stage.reservations" in staging
    assert "for reserve_plan in reserve_plans" in staging
    assert "def _resolved_view_layouts(" in finalization
    assert "layout_by_slot" in finalization
    assert "_view_id_for_slot" in finalization
    assert "reserve_layouts=reserve_layouts" in single
    assert "staged.reserve_projection_stages" in multi


def test_release_note_records_complete_parallax_reserve_contract() -> None:
    note = _read(RELEASE_NOTE)
    normalized = _normalized(note)

    assert note.lstrip().startswith("# Release 0.90.0")
    assert "parallax horizon angle" in normalized
    assert "blender stores the setting in radians and displays it in degrees" in normalized
    assert "default is **0°**" in note.lower()
    assert "deterministic dijkstra traversal" in normalized
    assert "eight deterministic virtual camera directions" in normalized
    assert "one union meshsnapshot" in normalized
    assert "max depth points" in normalized
    assert "reserve slots are emitted before the established front slot" in normalized
    assert "every virtual view owns its own alpha union crop" in normalized
    assert "same atomicfiletransaction" in normalized
    assert "Scene settings schema is **8**." in note
    assert "blender_to_spine2d_mesh_exporter 0.90.0.zip" in normalized
