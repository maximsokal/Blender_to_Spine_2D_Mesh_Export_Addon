"""Source contracts for camera-only projection mode UI and routing."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"
UI = PACKAGE / "ui.py"
DEPTH_SOURCE = PACKAGE / "blender_adapter" / "a1_depth_source_geometry_preparation.py"
GEOMETRY_PUBLIC = PACKAGE / "domain" / "geometry" / "__init__.py"
DEPTH_OWNER = (
    PACKAGE
    / "domain"
    / "geometry"
    / "depth_camera_projection_owner.py"
)
VISIBLE_TOPOLOGY = (
    PACKAGE
    / "domain"
    / "geometry"
    / "depth_camera_projection_visible_topology.py"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_camera_modes_show_fixed_active_camera_instead_of_axis_selector() -> None:
    source = _read(UI)

    # The canonical main panel owns export mode and projection selection. Camera-only
    # routes show a fixed Active Camera explanation and expose the axis selector only
    # in the Normal / UV branch.
    assert "A1TextureExportMode.CAMERA_PROJECTION" in source
    assert "A1TextureExportMode.DEPTH_CAMERA_PROJECTION" in source
    assert source.count('"spine2d_projection_direction"') == 1
    assert 'text="Active camera render → flat screen-space mesh"' in source
    assert 'text="Active camera render → optimized depth-relief mesh"' in source
    assert 'text="Projection direction"' in source


def test_depth_preparation_uses_visible_topology_projection_owner() -> None:
    source = _read(DEPTH_SOURCE)
    public_geometry = _read(GEOMETRY_PUBLIC)
    owner = _read(DEPTH_OWNER)
    visible_topology = _read(VISIBLE_TOPOLOGY)

    assert "build_depth_camera_projection_surface(" in source
    assert (
        "from .depth_camera_projection_owner import "
        "build_depth_camera_projection_surface"
    ) in public_geometry
    assert "def _crosses_camera_frame(" in owner
    assert "_build_visible_topology_surface(" in owner
    assert "_build_bounded_surface(" in owner
    assert "def _clip_triangle_to_frame(" in visible_topology
    assert "def _fit_clipped_rings_to_budget(" in visible_topology
    assert "if not ring.clipped" in visible_topology
