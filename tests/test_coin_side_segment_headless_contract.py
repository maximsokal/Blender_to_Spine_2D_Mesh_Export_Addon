from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_coin_star_normal_side_segment_retention_integration.py"
)


def test_real_coin_side_segment_runner_is_fail_closed_and_fixture_derived() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    for marker in (
        "[COIN-NORMAL-SIDE-SEGMENTS] PASS",
        "regions={len(axis.uv_regions.snapshots)}",
        "axis_edge_on_regions=",
        "axis_collapsed_faces=",
        "axis_collapsed_triangles=",
        "ownership=all-regions-all-triangles",
        "len(projections) == len(regions)",
        "triangle_count == face_count",
        "len(projection.loop_to_attachment_index) == len(snapshot.loops)",
        "axis_edge_on > 0",
        "axis_collapsed_faces > 0",
        "axis_collapsed_triangles == axis_collapsed_faces",
        "_scene_fingerprint() == scene_before",
        "_object_fingerprint(source) == object_before",
        "_temporary_datablock_names() == temporary_before",
    ):
        assert marker in source

    assert "== 48" not in source
    assert "segments=48" not in source
    assert "--expected-blend" in source
    assert "if __name__ == \"__main__\":" in source
