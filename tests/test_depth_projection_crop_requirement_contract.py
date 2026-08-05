"""Static architecture contract for per-view Depth UV crop requirements."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"
DOMAIN = PACKAGE / "domain" / "baking" / "projection_crop_requirements.py"
COLLECTOR = (
    PACKAGE
    / "blender_adapter"
    / "a1_depth_projection_crop_requirements.py"
)
OUTPUT_STAGING = PACKAGE / "blender_adapter" / "a1_output_staging.py"
TEXTURE_EXECUTOR = PACKAGE / "blender_adapter" / "texture_executor.py"
CAMERA_OUTPUT = PACKAGE / "blender_adapter" / "camera_projection_output.py"
POSTPROCESS = PACKAGE / "blender_adapter" / "camera_projection_postprocess.py"
FINALIZATION = (
    PACKAGE / "blender_adapter" / "a1_depth_projection_finalization.py"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_crop_requirement_pipeline_expands_before_physical_image_rewrite() -> None:
    domain = _read(DOMAIN)
    collector = _read(COLLECTOR)
    output_staging = _read(OUTPUT_STAGING)
    texture_executor = _read(TEXTURE_EXECUTOR)
    camera_output = _read(CAMERA_OUTPUT)
    postprocess = _read(POSTPROCESS)

    assert "class ProjectionUvBounds:" in domain
    assert "def expand_projection_layout_to_uv_bounds(" in domain
    assert "return replace(layout, crop=merged)" in domain

    assert "def depth_projection_required_uv_bounds(" in collector
    assert "expected_reserve = {" in collector
    assert "expected_all.update(expected_reserve)" in collector
    assert "if view_id == _FRONT_VIEW_ID:" in collector
    assert "projection.ordered_vertex_keys" in collector
    assert "attachment.uvs" in collector
    assert "if not resolved_reserve:" in collector
    assert "return MappingProxyType(resolved_reserve)" in collector

    assert "depth_projection_required_uv_bounds(item)" in output_staging
    assert "projection_uv_bounds_by_view=projection_uv_bounds_by_view" in output_staging
    assert "projection_uv_bounds_by_view:" in texture_executor
    assert "expected_reserve_view_ids = tuple(" in texture_executor
    assert "required_uv_bounds=bounds_by_view.get(view_id)" in texture_executor
    assert "required_uv_bounds=bounds_by_view.get(_FRONT_VIEW_ID)" not in texture_executor
    assert "required_uv_bounds=request.required_uv_bounds" in texture_executor
    assert "required_uv_bounds: ProjectionUvBounds | None" in camera_output
    assert "build_camera_projection_postprocess_request(" in camera_output
    assert "required_uv_bounds" in postprocess

    expand_index = postprocess.index("expand_projection_layout_to_uv_bounds(")
    rewrite_index = postprocess.index("rewrite_staged_image_with_crop(")
    assert expand_index < rewrite_index


def test_front_crop_compatibility_is_preserved_without_reserve_bounds() -> None:
    collector = _read(COLLECTOR)
    texture_executor = _read(TEXTURE_EXECUTOR)

    assert "if not resolved_reserve:" in collector
    assert "return None" in collector
    assert "expected_reserve_view_ids" in texture_executor
    assert "required_uv_bounds=bounds_by_view.get(view_id)" in texture_executor
    assert "required_uv_bounds=bounds_by_view.get(_FRONT_VIEW_ID)" not in texture_executor


def test_finalization_remains_strict_and_never_clamps_large_crop_errors() -> None:
    source = _read(FINALIZATION)

    assert "def _clamped_unit(" in source
    assert "lies outside its camera render crop" in source
    assert "raise A1DepthProjectionFinalizationError(" in source
    assert "return min(1.0, max(0.0, resolved))" not in source


def test_new_crop_requirement_owners_are_blender_independent() -> None:
    for path in (DOMAIN, COLLECTOR):
        source = _read(path)
        assert "import bpy" not in source
        assert "import bmesh" not in source
        assert "bpy.ops" not in source
        assert "bmesh.new" not in source
