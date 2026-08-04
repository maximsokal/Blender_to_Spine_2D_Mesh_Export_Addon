"""Static contract for the Blender Depth parallax camera/sequence matrix."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_depth_parallax_matrix_integration.py"
)


def test_matrix_runner_covers_orthographic_and_sequence_boundaries() -> None:
    assert RUNNER.is_file(), RUNNER
    source = RUNNER.read_text(encoding="utf-8")

    for required in (
        '_Case("orthographic-static", "ORTHO")',
        '_Case("perspective-sequence", "PERSP", _SEQUENCE_COUNT)',
        "camera.data.ortho_scale = 3.8",
        "TextureSequenceTiming(",
        "sequence_frame_count=case.sequence_count",
        "len(prepared.bake_plan.frame_tasks) == case.frame_count",
        "len(prepared.reserve_bake_plans[0].frame_tasks) == case.frame_count",
        "def _surface_ownership_counts(",
        "if red > green * 1.35",
        "if green > red * 1.35",
        "expect_front=True",
        "expect_front=False",
        "sequence crop changed between frames",
        "sequence frames are pixel-identical",
        "_assert_sequence_metadata(document, reserve_slot, case.sequence_count)",
        "_assert_sequence_metadata(document, front_slot, case.sequence_count)",
        "depth_parallax_cropped_view_count",
        "parallax_texture_output_count",
        "_capture_scene_bake_state() == bake_before",
        "_camera_fingerprint(camera) == camera_before",
        "_temporary_datablock_names() == temporary_before",
        "[DEPTH-PARALLAX-MATRIX] PASS",
    ):
        assert required in source


def test_matrix_runner_uses_public_export_and_physical_pngs() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "prepare_a1_object(" in source
    assert "export_a1_single_object(" in source
    assert "PNG_SIGNATURE" in source
    assert "_read_image(" in source
    assert "json.loads(" in source
    assert "unittest.mock" not in source
