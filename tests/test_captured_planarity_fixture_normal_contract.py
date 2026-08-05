"""Static contract separating captured planarity from stale-normal regressions."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAPTURED_REGRESSIONS = (
    ROOT
    / "tests"
    / "test_real_scene_geometry_regressions_0900.py"
)


def test_captured_planarity_fixture_uses_geometric_normal_by_default() -> None:
    source = CAPTURED_REGRESSIONS.read_text(encoding="utf-8")

    assert "def _newell_unit_normal(" in source
    assert "def _resolved_face_normal(" in source
    assert "face_normal: Vector3 | None = None" in source
    assert "declared_normal = _resolved_face_normal(positions, face_normal)" in source
    assert "normal=declared_normal" in source
    assert "return _newell_unit_normal(positions)" in source


def test_stale_flat_normal_remains_an_explicit_negative_fixture() -> None:
    source = CAPTURED_REGRESSIONS.read_text(encoding="utf-8")

    assert "def test_face15_with_explicit_stale_flat_normal_remains_rejected(" in source
    assert "face_normal=(0.0, 0.0, 1.0)" in source
    assert 'match="declared face normal"' in source
    assert '"exceeds tolerance 1.0 degrees"' in source


def test_quad_helper_does_not_unconditionally_stamp_a_flat_face_normal() -> None:
    source = CAPTURED_REGRESSIONS.read_text(encoding="utf-8")
    helper_start = source.index("def _quad_snapshot(")
    helper_end = source.index("\ndef _captured_planarity_metrics(", helper_start)
    helper = source[helper_start:helper_end]

    assert "normal=(0.0, 0.0, 1.0)" not in helper
