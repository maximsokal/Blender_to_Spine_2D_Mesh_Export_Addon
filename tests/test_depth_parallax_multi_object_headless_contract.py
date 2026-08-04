"""Static contract for Depth parallax multi-object and rollback acceptance."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_depth_parallax_multi_object_integration.py"
)


def test_multi_runner_covers_shared_transaction_and_object_namespaces() -> None:
    assert RUNNER.is_file(), RUNNER
    source = RUNNER.read_text(encoding="utf-8")

    for required in (
        "A1MultiObjectMode.STANDALONE",
        "prepare_a1_multi_object(",
        "export_a1_multi_object(",
        "all(isinstance(item, PreparedDepthA1Object)",
        "len(prepared.texture_output_paths) == 4",
        "reserve_plan.source_face_indices == (2, 3)",
        "projection_order == (reserve_slot, front_slot)",
        "len(result.output_files) == 5",
        "serialized object/reserve slot order differs",
        "component.{_LEFT_COMPONENT}.depth_parallax_cropped_view_count",
        "component.{_RIGHT_COMPONENT}.parallax_texture_output_count",
        "_capture_scene_bake_state() == bake_before",
        "_camera_fingerprint(camera) == camera_before",
        "_temporary_datablock_names() == temporary_before",
        "[DEPTH-PARALLAX-MULTI] PASS",
    ):
        assert required in source


def test_rollback_uses_public_progress_failpoint_and_requires_no_files() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    for required in (
        "def fail_before_second_object(update: A1ExportProgressUpdate)",
        'update.stage == "STAGE_OUTPUTS"',
        "update.object_id == _RIGHT_COMPONENT",
        "bpy.context.scene.camera = None",
        "progress_callback=fail_before_second_object",
        "rollback failpoint did not reach object two staging",
        "atomic rollback left output or staging files",
        "not result.success",
    ):
        assert required in source
    assert "unittest.mock" not in source
    assert "mock.patch" not in source
