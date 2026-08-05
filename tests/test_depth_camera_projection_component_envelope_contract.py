"""Static ownership contracts for sparse Depth component envelopes."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENVELOPE = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "geometry"
    / "depth_camera_projection_component_envelope.py"
)
OWNER = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "geometry"
    / "depth_camera_projection_owner.py"
)
FUNCTIONAL = (
    ROOT
    / "tests"
    / "test_depth_camera_projection_component_envelope.py"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_owner_repairs_only_sparse_full_frame_lattice_failures() -> None:
    source = _read(OWNER)

    assert "def _build_full_frame_surface(" in source
    assert "except DepthCameraProjectionError as lattice_error:" in source
    assert "if not is_sparse_lattice_failure(lattice_error):" in source
    assert "_build_component_envelope_surface(" in source
    assert "except _ComponentEnvelopeUnavailable as envelope_error:" in source
    assert "depth lattice and component-envelope fallback both failed" in source
    assert "if not _crosses_camera_frame(snapshot, frame):" in source


def test_component_envelope_preserves_components_with_a_hard_point_budget() -> None:
    source = _read(ENVELOPE)

    assert "class _ProjectedComponent:" in source
    assert "class _ComponentCluster:" in source
    assert "def _connected_face_components(" in source
    assert "def _partition_components(" in source
    assert "def _split_cluster(" in source
    assert "cluster_budget = effective_point_budget // 4" in source
    assert "len(samples) > effective_point_budget" in source
    assert "component envelope violated the Depth point budget" in source
    assert "_dense_surface_snapshot(" in source
    assert "_smooth_samples(" in source
    assert "bpy" not in source
    assert "bmesh" not in source
    assert "bpy.ops" not in source


def test_functional_regression_uses_more_than_direct_source_vertex_limit() -> None:
    source = _read(FUNCTIONAL)

    assert "component_count=130" in source
    assert "len(source.vertices) == 520" in source
    assert "first.sampled_point_count <= 128" in source
    assert "first.sampled_point_count % 4 == 0" in source
    assert "first == second" in source
    assert "settings=_settings(max_points=4)" in source
