"""Static ownership and performance contracts for dense parallax budgeting."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_INIT = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "geometry"
    / "__init__.py"
)
BUDGETED = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "geometry"
    / "depth_parallax_budgeted.py"
)
OPTIMIZED = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "geometry"
    / "depth_parallax_optimized.py"
)
PROJECTION_OWNER = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "geometry"
    / "depth_camera_projection_owner.py"
)
IDENTITY = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "geometry"
    / "depth_parallax_identity.py"
)
FUNCTIONAL = ROOT / "tests" / "test_depth_parallax_budgeted.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_public_geometry_route_uses_optimized_parallax_owner() -> None:
    source = _read(GEOMETRY_INIT)

    assert (
        "from .depth_parallax_optimized import "
        "build_depth_parallax_geometry_package"
    ) in source
    assert (
        "from .depth_parallax_budgeted import "
        "build_depth_parallax_geometry_package"
    ) not in source
    assert "build_depth_parallax_geometry_package," not in source.split(
        "from .depth_parallax import (",
        1,
    )[1].split(")", 1)[0]


def test_budgeted_helpers_keep_isolated_proxy_and_complete_ownership() -> None:
    source = _read(BUDGETED)

    assert "def _accumulated_horizon_costs_cached(" in source
    assert "edge_costs: dict[tuple[int, int], float]" in source
    assert "def _proxy_records_for_view(" in source
    assert "def _merge_view_assignments(" in source
    assert "generated_source_vertex_base" in source
    assert "SourceVertexId(" in source
    assert "points_per_view * view_count" in source
    assert "import bpy" not in source
    assert "import bmesh" not in source
    assert "bpy.ops" not in source


def test_optimized_owner_uses_one_projection_analysis_and_occupied_grid() -> None:
    source = _read(OPTIMIZED)

    assert "class _VisibilityTriangle:" in source
    assert "class _OccupiedScreenGrid:" in source
    assert "class _ParallaxSourceAnalysis:" in source
    assert "def _build_occupied_grid(" in source
    assert "def _build_source_analysis(" in source
    assert "def _front_visible_face_indices(" in source
    assert source.count("triangulate_snapshot(source)") == 1
    assert "face_indices = set(self.buckets.get((column, row), ()))" in source
    assert "expected_face_index" in source
    assert "first.index" in source
    assert "second.index" in source
    assert "exact_upper_bound > max_points" in source
    assert '"parallax-budget-proxy" if compacted else "parallax-union"' in source
    assert "len(union.vertices) > max_points" in source
    assert "analysis_elapsed" in source
    assert "perf_counter()" in source
    assert "_projected_triangles(" not in source
    assert "_face_geometry(source)" not in source
    assert "import bpy" not in source
    assert "import bmesh" not in source
    assert "bpy.ops" not in source


def test_projection_owner_bypasses_predictable_sparse_lattice_work() -> None:
    source = _read(PROJECTION_OWNER)

    assert "def _face_component_count_at_least(" in source
    assert "def _prefers_component_envelope(" in source
    assert "if _prefers_component_envelope(snapshot, settings):" in source
    assert 'reason="dense-disconnected-preflight"' in source
    assert "frame.project_world_point(" in source
    assert "_projected_triangles(" not in source
    assert "triangulate_snapshot(" not in source
    assert "import bpy" not in source
    assert "import bmesh" not in source
    assert "bpy.ops" not in source


def test_functional_proxy_does_not_share_front_topology() -> None:
    source = _read(FUNCTIONAL)

    assert "test_proxy_records_add_isolated_vertices_within_reserved_budget" in source
    assert "len(union.vertices) == len(front.vertices) + 4" in source
    assert "front_indices.isdisjoint(reserve_indices)" in source
    assert "test_three_point_proxy_uses_one_triangle" in source
    assert "test_low_budget_view_assignments_merge_to_nearest_retained_direction" in source


def test_identity_uses_explicit_render_owners_only_for_marked_budget_proxy() -> None:
    source = _read(IDENTITY)

    assert '_BUDGET_PROXY_MARKER = ":parallax-budget-proxy:"' in source
    assert "if _BUDGET_PROXY_MARKER in surface.snapshot.snapshot_id:" in source
    assert "resolved = tuple(sorted(set(surface.source_face_indices)))" in source
    assert "int(face.source_id.face_index)" in source
    assert source.index(
        "if _BUDGET_PROXY_MARKER in surface.snapshot.snapshot_id:"
    ) < source.index("int(face.source_id.face_index)")
