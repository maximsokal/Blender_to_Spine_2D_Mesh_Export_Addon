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
IDENTITY = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "geometry"
    / "depth_parallax_identity.py"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_public_geometry_route_uses_budgeted_parallax_owner() -> None:
    source = _read(GEOMETRY_INIT)

    assert "from .depth_parallax_budgeted import build_depth_parallax_geometry_package" in source
    assert "build_depth_parallax_geometry_package," not in source.split(
        "from .depth_parallax import (",
        1,
    )[1].split(")", 1)[0]


def test_budgeted_owner_uses_local_visibility_and_front_shared_proxy_vertices() -> None:
    source = _read(BUDGETED)

    assert "class _ScreenGrid:" in source
    assert "def _front_visible_face_indices_fast(" in source
    assert "for candidate in grid.candidates(x, y):" in source
    assert "def _accumulated_horizon_costs_cached(" in source
    assert "edge_costs: dict[tuple[int, int], float]" in source
    assert "exact_upper_bound > max_points" in source
    assert "def _proxy_records_for_view(" in source
    assert "front_source_ids" not in source
    assert '"parallax-budget-proxy" if compacted else "parallax-union"' in source
    assert "len(union.vertices) > max_points" in source
    assert "source_face_indices=_evaluated_owner_indices(geometry, face_indices)" in source
    assert "logger.info(" in source
    assert "perf_counter()" in source
    assert "import bpy" not in source
    assert "import bmesh" not in source
    assert "bpy.ops" not in source


def test_identity_uses_explicit_render_owners_only_for_marked_budget_proxy() -> None:
    source = _read(IDENTITY)

    assert '_BUDGET_PROXY_MARKER = ":parallax-budget-proxy:"' in source
    assert "if _BUDGET_PROXY_MARKER in surface.snapshot.snapshot_id:" in source
    assert "resolved = tuple(sorted(set(surface.source_face_indices)))" in source
    assert "int(face.source_id.face_index)" in source
    assert source.index(
        "if _BUDGET_PROXY_MARKER in surface.snapshot.snapshot_id:"
    ) < source.index("int(face.source_id.face_index)")
