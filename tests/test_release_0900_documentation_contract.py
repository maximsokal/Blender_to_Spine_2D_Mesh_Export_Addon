"""Current public documentation contracts for release 0.90.0."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
INSTALLATION = DOCS / "installation.md"
SETTINGS = DOCS / "settings-reference.md"
USAGE = DOCS / "usage.md"
TESTING = DOCS / "testing.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _normalized(value: str) -> str:
    return " ".join(
        value.lower().replace("-", " ").replace("`", " ").split()
    )


def test_current_guides_do_not_advertise_the_retired_0810_release() -> None:
    for path in (INSTALLATION, SETTINGS, USAGE, TESTING):
        source = _read(path)
        assert "0.81.0" not in source, path
        assert "Scene settings schema 7" not in source, path
        assert "Scene schema 7" not in source, path


def test_installation_uses_0900_archive_schema_and_manual_parallax_gate() -> None:
    source = _read(INSTALLATION)
    normalized = _normalized(source)

    assert "blender_to_spine2d_mesh_exporter-0.90.0.zip" in source
    assert "Version 0.90.0 exposes three independent modes" in source
    assert "Version 0.90.0 uses Scene settings schema 8" in source
    assert "Parallax Horizon Angle" in source
    assert "The default is `0°`" in source
    assert "reserve attachments reuse the same generated vertex-bone rig" in normalized
    assert "reserve slots are serialized before the front slot" in normalized
    assert "Repeat with an Orthographic camera" in source
    assert "two-frame material sequence" in source


def test_settings_reference_describes_the_complete_public_parallax_contract() -> None:
    source = _read(SETTINGS)
    normalized = _normalized(source)

    assert "version **0.90.0**" in source
    assert "settings schema 8" in source
    assert "### Parallax Horizon Angle" in source
    assert "0° through 89°" in source
    assert "0° through 45°" in source
    assert "The persisted value and the domain contract use radians" in source
    assert "deterministic Dijkstra traversal" in source
    assert "accumulated unsigned dihedral angle" in source
    for direction in (
        "RIGHT",
        "UP_RIGHT",
        "UP",
        "UP_LEFT",
        "LEFT",
        "DOWN_LEFT",
        "DOWN",
        "DOWN_RIGHT",
    ):
        assert direction in source
    assert "one temporary face-isolated render proxy" in source
    assert "Reserve slots are emitted before the FRONT slot" in source
    assert "one shared Z-group assignment" in source
    assert "fail closed" in normalized


def test_usage_explains_front_reserve_ownership_draw_order_and_atomicity() -> None:
    source = _read(USAGE)
    normalized = _normalized(source)

    assert "### Depth Camera Projection" in source
    assert "## Configure parallax reserve" in source
    assert "Parallax Horizon Angle" in source
    assert "one union MeshSnapshot" in source
    assert "face-isolated FRONT and reserve camera renders" in source
    assert "reserve attachments followed by FRONT attachment" in source
    assert "Each reserve texture therefore contains its own surface" in source
    assert "Reserve slots are serialized before the FRONT slot" in source
    assert "shared generated bones" in source
    assert "rolls back the JSON and every staged texture" in source
    assert "front and reserve views keep independent crop rectangles" in normalized


def test_testing_guide_requires_all_0900_runtime_and_package_evidence() -> None:
    source = _read(TESTING)

    assert "current extension candidate is **0.90.0**" in source
    assert "Scene settings schema 8" in source
    assert "Every Blender headless command must include `--python-exit-code 1`" in source
    for runner in (
        "run_depth_camera_projection_integration.py",
        "run_depth_parallax_integration.py",
        "run_depth_parallax_matrix_integration.py",
        "run_depth_parallax_multi_object_integration.py",
        "run_depth_camera_projection_multi_object_integration.py",
    ):
        assert runner in source
    for marker in (
        "[DEPTH-CAMERA] PASS",
        "[DEPTH-PARALLAX] PASS",
        "[DEPTH-PARALLAX-MATRIX] PASS",
        "[DEPTH-PARALLAX-MULTI] PASS",
        "[DEPTH-CAMERA-MULTI] PASS",
    ):
        assert marker in source
    assert "blender_to_spine2d_mesh_exporter-0.90.0.zip" in source
    assert "extension_install_gate_0.90.0" in source
    assert 'throw "Real bpy environment not found; release gate is incomplete"' in source
    assert "archive path, byte size, and SHA256" in source
