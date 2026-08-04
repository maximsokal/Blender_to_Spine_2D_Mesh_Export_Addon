"""Static contract for the real-scene geometry regression runner."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_real_scene_geometry_regressions_integration.py"
)
TRIANGULATION = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "geometry"
    / "triangulation.py"
)
GEOMETRY_PREPARATION = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "application"
    / "a1_geometry_preparation.py"
)
PARALLAX_IDENTITY = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "geometry"
    / "depth_parallax_identity.py"
)
DEPTH_PREPARATION = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "blender_adapter"
    / "a1_depth_source_geometry_preparation.py"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_runner_reproduces_both_public_multi_object_failures() -> None:
    source = _read(RUNNER)

    assert '_MUSHROOMS_OBJECT_NAME = "Plane.008"' in source
    assert '_FLOWER_SHOP_OBJECT_NAME = "banco"' in source
    assert "_WARP_HEIGHT = 0.0007581877679385422" in source
    assert "_HORIZON_ANGLE = radians(20.0)" in source
    assert "_RESERVE_FOLD_ANGLE = radians(15.0)" in source
    assert 'type="ARRAY"' in source
    assert "modifier.count = _COPY_COUNT" in source
    assert "prepare_a1_multi_object(" in source
    assert "PreparedDepthA1Object" in source
    assert 'issue.code == "EVALUATED_IDENTITY_REBASED"' in source
    assert "banco_package.front_face_indices == (2, 3)" in source
    assert "banco_package.reserve_face_indices == (0, 1)" in source
    assert "banco_prepared_lineage == banco_union_lineage" in source
    assert "region.transfer_report.complete" in source
    assert "_object_fingerprint(mushrooms, array_modifier)" in source
    assert "_temporary_datablock_names() == temporary_before" in source
    assert "[REAL-SCENE-GEOMETRY-REGRESSIONS] PASS" in source


def test_triangulation_uses_absolute_and_scale_relative_planarity() -> None:
    source = _read(TRIANGULATION)

    assert "relative_planarity_tolerance: float = 5e-2" in source
    assert "def _polygon_scale(" in source
    assert "def _centroid(" in source
    assert "effective_tolerance = max(" in source
    assert "def _validate_triangle_orientation(" in source
    assert "Polygon is too non-planar for deterministic projection" in source
    assert "import bpy" not in source
    assert "import bmesh" not in source
    assert "bpy.ops" not in source


def test_geometry_coverage_uses_local_ids_and_lineage_multiplicity() -> None:
    source = _read(GEOMETRY_PREPARATION)

    assert "from collections import Counter" in source
    assert "Decomposition regions overlap in local face coverage" in source
    assert "expected_counts = Counter(" in source
    assert "actual_counts = Counter(prepared_source_face_ids)" in source
    assert "Prepared regions do not cover SourceFaceId multiplicity exactly" in source
    assert "Prepared regions overlap in SourceFaceId coverage" not in source


def test_parallax_union_is_canonical_before_geometry_and_uv_consumers() -> None:
    identity_source = _read(PARALLAX_IDENTITY)
    preparation_source = _read(DEPTH_PREPARATION)

    assert "def canonicalize_depth_parallax_package_identity(" in identity_source
    assert "rebase_mesh_snapshot_to_evaluated_identity(" in identity_source
    assert "duplicate SourceFaceId values" in identity_source
    assert "duplicate SourceLoopId values" in identity_source
    assert "_subset_material(" in identity_source
    assert "source_face_indices" in identity_source
    assert "import bpy" not in identity_source
    assert "import bmesh" not in identity_source
    assert "bpy.ops" not in identity_source

    build_index = preparation_source.index(
        "camera_z_package = build_depth_parallax_geometry_package("
    )
    canonicalize_index = preparation_source.index(
        "camera_z_package = canonicalize_depth_parallax_package_identity("
    )
    distance_index = preparation_source.index(
        "depth_package = _package_to_camera_distance(camera_z_package)"
    )
    geometry_index = preparation_source.index(
        "geometry = prepare_a1_geometry_regions("
    )
    assert build_index < canonicalize_index < distance_index < geometry_index
