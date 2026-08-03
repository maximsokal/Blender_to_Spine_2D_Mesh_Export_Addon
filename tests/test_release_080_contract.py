"""Historical release contracts for the 0.80.0 all-sequence milestone."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE_NOTE = ROOT / "docs" / "releases" / "0.80.0.md"
STANDALONE_RUNNER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_multi_object_sequence_mode_matrix_integration.py"
)
CONNECTED_MIXED_RUNNER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_connected_mixed_sequence_mode_matrix_integration.py"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _normalized(value: str) -> str:
    return " ".join(value.lower().replace("-", " ").replace("`", " ").split())


def test_release_note_is_historical_version_0800() -> None:
    note = _read(RELEASE_NOTE)

    assert "# Release 0.80.0" in note
    assert "blender_to_spine2d_mesh_exporter-0.80.0.zip" in note


def test_release_note_records_complete_standalone_sequence_matrix() -> None:
    note = _read(RELEASE_NOTE)
    normalized = _normalized(note)

    for version in ("3.8", "4.0", "4.1", "4.2", "4.3"):
        assert version in note
    assert "normal uv segments and camera projection" in normalized
    assert "two animated objects per case" in normalized
    assert "legacy attachment swap serialization" in normalized
    assert "native sequence serialization" in normalized
    assert "128x128" in note


def test_release_note_records_connected_and_mixed_supported_matrix() -> None:
    note = _read(RELEASE_NOTE)
    normalized = _normalized(note)

    assert "spine 4.2" in normalized
    assert "three axis rotation" in normalized
    assert "two axis rotation + scale" in normalized
    assert "two object connected output" in normalized
    assert "two object connected subgroup and one standalone object" in normalized
    assert "native loop sequence timelines" in normalized
    assert "unsupported target, profile, and composition combinations remain fail closed" in normalized


def test_release_note_records_runtime_restoration() -> None:
    note = _read(RELEASE_NOTE)
    normalized = _normalized(note)

    for phrase in (
        "source object transforms",
        "current frame",
        "active object",
        "selection",
        "materials",
        "scene bake settings",
        "render settings",
        "camera state",
        "temporary blender datablocks",
    ):
        assert phrase in normalized
    assert "scene settings schema remains version 6" in normalized


def test_historical_release_runners_still_exist() -> None:
    standalone = _read(STANDALONE_RUNNER)
    connected_mixed = _read(CONNECTED_MIXED_RUNNER)

    assert "export_a1_multi_object(" in standalone
    assert "SpineJsonTarget.SPINE_3_8" in standalone
    assert "SpineJsonTarget.SPINE_4_3" in standalone
    assert "A1TextureExportMode.NORMAL_UV_SEGMENTS" in standalone
    assert "A1TextureExportMode.CAMERA_PROJECTION" in standalone

    assert "export_a1_multi_object(" in connected_mixed
    assert "export_a1_mixed_object(" in connected_mixed
    assert "A1MultiObjectMode.CONNECTED" in connected_mixed
    assert "A1MultiObjectMode.MIXED" in connected_mixed
    assert "A1RigProfile.THREE_AXIS_ROTATION" in connected_mixed
    assert "A1RigProfile.TWO_AXIS_ROTATION_SCALE" in connected_mixed
