"""Source contracts for camera-only projection mode UI and routing."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"
RIG_UI = PACKAGE / "rig_ui.py"
DEPTH_SOURCE = PACKAGE / "blender_adapter" / "a1_depth_source_geometry_preparation.py"
GEOMETRY_PUBLIC = PACKAGE / "domain" / "geometry" / "__init__.py"
VISIBLE_OWNER = (
    PACKAGE
    / "domain"
    / "geometry"
    / "depth_camera_projection_visible_topology.py"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_camera_modes_show_fixed_active_camera_instead_of_axis_selector() -> None:
    source = _read(RIG_UI)

    assert "def _draw_forced_active_camera_projection(" in source
    assert source.count("_draw_forced_active_camera_projection(layout)") == 2
    assert source.count("_draw_projection_direction(layout, scene)") == 1
    assert "A1TextureExportMode.CAMERA_PROJECTION.value" in source
    assert "A1TextureExportMode.DEPTH_CAMERA_PROJECTION.value" in source
    assert 'row.label(text="Active Camera", icon="CAMERA_DATA")' in source


def test_depth_preparation_uses_visible_topology_projection_owner() -> None:
    source = _read(DEPTH_SOURCE)
    public_geometry = _read(GEOMETRY_PUBLIC)
    visible_owner = _read(VISIBLE_OWNER)

    assert "build_depth_camera_projection_surface(" in source
    assert (
        "from .depth_camera_projection_visible_topology import ("
        in public_geometry
    )
    assert "def _clip_triangle_to_frame(" in visible_owner
    assert "def _fit_clipped_rings_to_budget(" in visible_owner
    assert "if not ring.clipped" in visible_owner
    assert "_build_bounded_surface(" in visible_owner
