"""Static contracts for captured real-scene geometry regressions."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_real_scene_geometry_regressions_integration.py"
)
CAPTURED_PLANARITY_RUNNER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_mushrooms_cube012_planarity_integration.py"
)
REAL_BLEND_RUNNER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_mushrooms_real_blend_integration.py"
)
FLOAT32_PLANARITY_TEST = (
    ROOT
    / "tests"
    / "test_mushrooms_planarity_float32_roundtrip.py"
)
REAL_SCENE_REGRESSIONS = (
    ROOT
    / "tests"
    / "test_real_scene_geometry_regressions_0900.py"
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


def test_runner_reproduces_plane008_and_banco_public_multi_object_failures() -> None:
    source = _read(RUNNER)

    assert '_MUSHROOMS_OBJECT_NAME = "Plane.008"' in source
    assert '_FLOWER_SHOP_OBJECT_NAME = "banco"' in source
    assert "_WARP_HEIGHT = 0.0007581877679385422" in source
    assert "_HORIZON_ANGLE = radians(20.0)" in source
    assert "_RESERVE_FOLD_ANGLE = radians(15.0)" in source
    assert "folded_x = 0.9 - cos(_RESERVE_FOLD_ANGLE)" in source
    assert "folded_z = -sin(_RESERVE_FOLD_ANGLE)" in source
    assert 'type="ARRAY"' in source
    assert "modifier.count = _COPY_COUNT" in source
    assert "prepare_a1_multi_object(" in source
    assert "PreparedDepthA1Object" in source
    assert 'issue.code == "EVALUATED_IDENTITY_REBASED"' in source
    assert "banco_package.front_face_indices == (2, 3)" in source
    assert "banco_package.reserve_face_indices == (0, 1)" in source
    assert "reserve_owned_faces == (0,)" in source
    assert "banco_prepared_lineage == banco_union_lineage" in source
    assert "region.transfer_report.complete" in source
    assert "_source_fingerprint(mushrooms, array_modifier)" in source
    assert "_temporary_datablock_names() == temporary_before" in source
    assert "[REAL-SCENE-GEOMETRY-REGRESSIONS] PASS" in source


def test_captured_mushrooms_runner_reproduces_both_traceback_metrics_publicly() -> None:
    source = _read(CAPTURED_PLANARITY_RUNNER)

    assert '_CUBE_OBJECT_NAME = "Cube.012"' in source
    assert '_CUBE_COMPONENT_ID = "object_1:Cube.012"' in source
    assert '_PLANE_OBJECT_NAME = "Plane.008"' in source
    assert '_PLANE_COMPONENT_ID = "object_2:Plane.008"' in source

    assert "_CUBE_SIDE_LENGTH = 0.09277534946637461" in source
    assert "_CUBE_WARP_HEIGHT = 0.00030260673225328884" in source
    assert "_CUBE_CAPTURED_MAXIMUM_PLANE_DISTANCE = 7.565148185365435e-05" in source
    assert "_CUBE_CAPTURED_POLYGON_SCALE = 0.13120450643194492" in source

    assert "_PLANE_SIDE_LENGTH = 0.0164835268501562" in source
    assert "_PLANE_WARP_HEIGHT = 0.000379143881919382" in source
    assert "_PLANE_CAPTURED_MAXIMUM_PLANE_DISTANCE = 9.477343601658832e-05" in source
    assert "_PLANE_CAPTURED_POLYGON_SCALE = 0.023314310303391664" in source

    assert "_DISTANCE_ABS_TOLERANCE = 1.0e-11" in source
    assert "_SCALE_ABS_TOLERANCE = 1.0e-8" in source
    assert "def _propagated_normalized_warp_tolerance(" in source
    assert "distance_term = _DISTANCE_ABS_TOLERANCE / lower_scale" in source
    assert "scale_term = (" in source
    assert "ratio_tolerance = _propagated_normalized_warp_tolerance(" in source
    assert "propagated_tolerance={ratio_tolerance}" in source
    assert "<= 1.0e-10" not in source

    assert "def _create_cube012_source(" in source
    assert "def _create_plane008_source(" in source
    assert "sources = (" in source
    assert source.count("A1MultiObjectSource(") == 2
    assert "len(sources) == 2" in source
    assert "prepare_a1_multi_object(" in source
    assert "len(prepared_multi.objects) == 2" in source
    assert "def _prepared_by_component(" in source
    assert "prepared_by_component[_CUBE_COMPONENT_ID]" in source
    assert "prepared_by_component[_PLANE_COMPONENT_ID]" in source
    assert "PreparedDepthA1Object" in source
    assert '"depth_projection_source_triangle_count"' in source
    assert "_source_fingerprint(cube) == cube_before" in source
    assert "_source_fingerprint(plane) == plane_before" in source
    assert "_temporary_datablock_names() == temporary_before" in source
    assert "[MUSHROOMS-CAPTURED-PLANARITY] PASS" in source
    assert '"triangles=2+2 sources=2 pipeline=public-multi-object"' in source
    assert "import bmesh" not in source
    assert "bpy.ops" not in source


def test_direct_real_blend_runner_opens_no_synthetic_scene_and_scans_all_ngons() -> None:
    source = _read(REAL_BLEND_RUNNER)

    assert 'parser.add_argument(\n        "--expected-blend"' in source
    assert "bpy.data.filepath" in source
    assert '_OBJECT_NAMES = ("Plane.008", "Cube.012")' in source
    assert "def _require_loaded_blend(" in source
    assert "def _require_source_objects(" in source
    assert "def _read_normalized_snapshot(" in source
    assert "_read_source_snapshot(" in source
    assert "_canonicalize_depth_evaluated_identity(" in source
    assert "_normalize_source_geometry(" in source
    assert "def _scan_snapshot_planarity(" in source
    assert "Real mushrooms n-gon planarity scan found all blockers" in source
    assert "for face in sorted(snapshot.faces" in source
    assert "violations.append(record)" in source
    assert "triangulate_snapshot(snapshot)" in source
    assert "prepare_a1_multi_object(" in source
    assert "[MUSHROOMS-REAL-BLEND] PASS" in source
    assert "_create_mesh_object" not in source
    assert "_clear_scene" not in source
    assert "bpy.ops" not in source
    assert "import bmesh" not in source


def test_float32_roundtrip_regression_locks_the_runner_failure() -> None:
    source = _read(FLOAT32_PLANARITY_TEST)

    assert 'unpack("<f", pack("<f", float(value)))[0]' in source
    assert '_RETIRED_RATIO_ABS_TOLERANCE = 1.0e-10' in source
    assert "def _propagated_ratio_tolerance(" in source
    assert "distance_term = _DISTANCE_ABS_TOLERANCE / lower_scale" in source
    assert "scale_term = (" in source
    assert "plane_ratio_delta > _RETIRED_RATIO_ABS_TOLERANCE" in source
    assert "triangulate_snapshot(snapshot)" in source
    assert '"Cube.012"' in source
    assert '"Plane.008"' in source
    assert "import bpy" not in source
    assert "import bmesh" not in source
    assert "bpy.ops" not in source


def test_real_scene_pure_regression_locks_plane008_face15_traceback() -> None:
    source = _read(REAL_SCENE_REGRESSIONS)

    assert "_PLANE008_FACE15_SIDE_LENGTH = 0.01813398508349017" in source
    assert "_PLANE008_FACE15_WARP_HEIGHT = 0.00060171214277301" in source
    assert (
        "_PLANE008_FACE15_CAPTURED_MAXIMUM_PLANE_DISTANCE = "
        "0.0001503866471090267"
    ) in source
    assert (
        "_PLANE008_FACE15_CAPTURED_POLYGON_SCALE = "
        "0.025652385610684413"
    ) in source
    assert "0.005862481930194809" in source
    assert "test_real_mushrooms_plane008_face15_uses_bounded_absolute_floor" in source


def test_triangulation_uses_bounded_absolute_relative_and_hard_warp_limits() -> None:
    source = _read(TRIANGULATION)

    assert "planarity_tolerance: float = 2.0e-4" in source
    assert "planarity_tolerance: float = 1.0e-4" not in source
    assert "relative_planarity_tolerance: float = 1.0e-3" in source
    assert "maximum_relative_planarity_warp: float = 1.0e-2" in source
    assert "relative_planarity_tolerance: float = 2.5e-4" not in source
    assert "relative_planarity_tolerance: float = 5e-2" not in source
    assert "normal_alignment_tolerance_degrees: float = 1.0" in source
    assert "def _polygon_scale(" in source
    assert "def _centroid(" in source
    assert "normalized_warp = maximum / scale" in source
    assert "effective_tolerance = max(" in source
    assert "or normalized_warp > maximum_relative_warp" in source
    assert "hard ceiling" in source
    assert "source face 15" in source
    assert "def _validate_declared_normal_alignment(" in source
    assert "def _validate_triangle_orientation(" in source
    assert "Polygon is not planar within deterministic tolerance" in source
    assert "Polygon is not planar relative to its declared face normal" in source
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

    assert "def _evaluated_render_face_indices(" in identity_source
    assert "for face in surface.snapshot.faces" in identity_source
    assert "def _canonical_reserve_surface(" in identity_source
    assert "source_face_indices=render_face_indices" in identity_source
    assert "def _validate_front_only_package_identity(" in identity_source
    assert (
        "Reserve-free Depth package must share one unchanged FRONT snapshot"
        in identity_source
    )
    assert "if not package.reserve_surfaces:" in identity_source
    assert "return package" in identity_source
    assert "def canonicalize_depth_parallax_package_identity(" in identity_source
    assert "rebase_mesh_snapshot_to_evaluated_identity(" in identity_source
    assert "duplicate SourceFaceId values" in identity_source
    assert "duplicate SourceLoopId values" in identity_source
    assert "_subset_material(" in identity_source
    assert "import bpy" not in identity_source
    assert "import bmesh" not in identity_source
    assert "bpy.ops" not in identity_source

    front_only_index = identity_source.index(
        "if not package.reserve_surfaces:"
    )
    rebase_index = identity_source.index(
        "rebase = rebase_mesh_snapshot_to_evaluated_identity("
    )
    assert front_only_index < rebase_index

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
