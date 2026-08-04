"""Static contract for the real Blender 5.2 Depth parallax smoke runner."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_depth_parallax_integration.py"
)


def test_depth_parallax_blender_runner_covers_runtime_boundaries() -> None:
    assert RUNNER.is_file(), RUNNER
    source = RUNNER.read_text(encoding="utf-8")

    for required in (
        "DepthParallaxSettings(",
        "PreparedDepthA1Object",
        "package.front_face_indices == (0, 1)",
        "package.reserve_face_indices == (2, 3)",
        "len(package.reserve_surfaces) == 1",
        "reserve_plan.camera_world_matrix_override is not None",
        "slot_order == (reserve_slot, front_slot)",
        "front and reserve attachments do not share hinge vertex bones",
        "reserve texture does not reveal folded green surface",
        "front material leaked into face-isolated reserve texture",
        "depth_parallax_cropped_view_count",
        "parallax_texture_output_count",
        "_capture_scene_bake_state() == bake_before",
        "_camera_fingerprint(camera) == camera_before",
        "_temporary_datablock_names() == temporary_before",
        "[DEPTH-PARALLAX] PASS",
    ):
        assert required in source


def test_depth_parallax_runner_uses_public_prepare_and_export_routes() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "prepare_a1_object(" in source
    assert "export_a1_single_object(" in source
    assert "_create_folded_surface(" in source
    assert "_assert_serialized_attachment(" in source
    assert "PNG_SIGNATURE" in source
    assert "json.loads(" in source
