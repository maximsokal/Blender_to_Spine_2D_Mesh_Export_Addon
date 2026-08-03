"""Static guards for the real Blender Depth Camera Projection runners."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SINGLE_RUNNER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_depth_camera_projection_integration.py"
)
MULTI_RUNNER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_depth_camera_projection_multi_object_integration.py"
)


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _called_names(path: Path) -> set[str]:
    tree = ast.parse(_source(path), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Name):
            names.add(function.id)
        elif isinstance(function, ast.Attribute):
            names.add(function.attr)
    return names


def test_single_runner_pins_targets_camera_types_sequence_and_limits() -> None:
    source = _source(SINGLE_RUNNER)

    for target in (
        "SpineJsonTarget.SPINE_3_8",
        "SpineJsonTarget.SPINE_4_0",
        "SpineJsonTarget.SPINE_4_1",
        "SpineJsonTarget.SPINE_4_2",
        "SpineJsonTarget.SPINE_4_3",
    ):
        assert target in source
    assert '_Case(SpineJsonTarget.SPINE_4_2, "ORTHO")' in source
    assert '_Case(SpineJsonTarget.SPINE_4_2, "PERSP", _SEQUENCE_COUNT)' in source
    assert "_TEXTURE_SIZE = 96" in source
    assert "_MAX_DEPTH_POINTS = 36" in source
    assert "_SEQUENCE_COUNT = 2" in source
    assert "len(cases) == 7" in source


def test_single_runner_checks_global_camera_zero_and_compensated_one_mesh() -> None:
    source = _source(SINGLE_RUNNER)
    called = _called_names(SINGLE_RUNNER)

    for required in (
        "prepare_a1_object",
        "export_a1_single_object",
        "decode_weighted_vertices",
        "attachment_setup_positions",
        "_read_image",
        "json.loads",
    ):
        if "." in required:
            assert required in source
        else:
            assert required in called
    for evidence in (
        "A1TextureExportMode.DEPTH_CAMERA_PROJECTION",
        "CameraProjectionPlan",
        "edge_threshold_fraction=0.0",
        "all(distance > 0.0 for distance in distances)",
        "all(offset > 0.0 for offset in offsets)",
        "depth_camera_global_camera_zero",
        "depth_camera_absolute_distance_retained",
        "depth_camera_parent_y_compensated",
        "depth_camera_single_attachment",
        "len(prepared.document_assembly.projections) == 1",
        "parent depth compensation failed",
        "unexpected Segment_1 slot",
        "crop changed triangles",
        "PNG_SIGNATURE",
        "cropped UV outside 0..1",
        "_assert_bone_schema",
        "_assert_constraint_schema",
        "_capture_context",
        "_capture_scene_bake_state",
        "_scene_render_fingerprint",
        "_material_fingerprint",
        "_temporary_datablock_names",
    ):
        assert evidence in source


def test_multi_runner_pins_material_sequence_and_static_depth_object() -> None:
    source = _source(MULTI_RUNNER)
    called = _called_names(MULTI_RUNNER)

    assert "SpineJsonTarget.SPINE_4_2" in source
    assert "A1MultiObjectMode.STANDALONE" in source
    assert "_SEQUENCE_COUNT = 2" in source
    assert 'component_id="sequence_depth"' in source
    assert 'component_id="static_depth"' in source
    assert "export_a1_multi_object" in called
    assert "prepare_a1_multi_object" in called
    assert "len(result.output_files) == 4" in source
    assert "len(sequence_paths) == 2" in source
    assert "len(static_paths) == 1" in source
    assert "static sibling inherited sequence metadata" in source
    assert "static sibling inherited sequence timeline" in source
    assert "generated relief bones missing" in source
    assert "all_objects" in source


def test_multi_runner_animates_object_and_camera_but_freezes_depth_silhouette() -> None:
    source = _source(MULTI_RUNNER)

    for evidence in (
        "_keyframe_sequence_source",
        "_keyframe_active_camera",
        "animated object/camera changed sequence crop",
        "animated object/camera changed Depth sequence silhouette",
        "animated material did not change visible sequence RGB",
        "objects lost shared camera-zero depth ordering",
        "projected object origins collapsed",
        "serialized main bones lost projected relative placement",
        "must serialize only Segment_0",
        "unexpected second segment",
    ):
        assert evidence in source


def test_multi_runner_checks_atomic_files_and_blender_state_restoration() -> None:
    source = _source(MULTI_RUNNER)

    for evidence in (
        "PNG_SIGNATURE",
        "json.loads",
        "_read_image",
        "_capture_context",
        "_capture_scene_bake_state",
        "_scene_render_fingerprint",
        "_material_fingerprint",
        "_temporary_datablock_names",
        "texture output names collided",
        "material sequence PNGs are byte-identical",
        "export changed source materials",
    ):
        assert evidence in source
