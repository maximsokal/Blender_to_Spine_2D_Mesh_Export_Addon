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


def test_single_runner_uses_public_export_and_checks_real_relief_outputs() -> None:
    source = _source(SINGLE_RUNNER)
    called = _called_names(SINGLE_RUNNER)

    for required in (
        "prepare_a1_object",
        "export_a1_single_object",
        "decode_weighted_vertices",
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
        "depth_camera_offsets_toward_camera_only",
        "min(offsets) == 0.0",
        "all(offset >= 0.0 for offset in offsets)",
        "component.attachment.triangles",
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


def test_multi_runner_pins_one_sequence_and_one_static_depth_object() -> None:
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
        "sequence PNGs are identical",
        "static PNG collided with sequence",
        "export changed source materials",
    ):
        assert evidence in source
