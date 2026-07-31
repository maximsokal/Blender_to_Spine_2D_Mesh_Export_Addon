"""Release contract for the Object Origin pivot implementation."""

from __future__ import annotations

from pathlib import Path
import tomllib


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    REPOSITORY_ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "blender_manifest.toml"
)
TASK_PATH = (
    REPOSITORY_ROOT
    / "docs"
    / "tasks"
    / "normal_uv_segments_object_origin_pivot.md"
)
CORRECTION_RELEASE_PATH = REPOSITORY_ROOT / "docs" / "releases" / "0.55.1.md"


def test_object_origin_release_uses_version_0551() -> None:
    manifest = tomllib.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["id"] == "blender_to_spine2d_mesh_exporter"
    assert manifest["version"] == "0.55.1"


def test_object_origin_release_keeps_approved_task_document() -> None:
    task = TASK_PATH.read_text(encoding="utf-8")

    assert "Approved for implementation" in task
    assert "`0.55.0`" in task
    assert "TWO_AXIS_ROTATION_SCALE" in task
    assert "OBJECT_ORIGIN" in task
    assert "Camera Projection" in task


def test_object_origin_acceptance_correction_has_release_note() -> None:
    release_note = CORRECTION_RELEASE_PATH.read_text(encoding="utf-8")

    assert "0.55.1" in release_note
    assert "view_layer.update" in release_note
    assert "production pivot code" in release_note
