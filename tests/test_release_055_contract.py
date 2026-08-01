"""Release contracts for the 0.55 Object Origin and projection work."""

from __future__ import annotations

from pathlib import Path
import tomllib


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    REPOSITORY_ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "blender_manifest.toml"
)
OBJECT_ORIGIN_TASK_PATH = (
    REPOSITORY_ROOT
    / "docs"
    / "tasks"
    / "normal_uv_segments_object_origin_pivot.md"
)
PROJECTION_TASK_PATH = (
    REPOSITORY_ROOT
    / "docs"
    / "tasks"
    / "normal_uv_segments_projection_space_and_draw_order.md"
)
OBJECT_ORIGIN_CORRECTION_RELEASE_PATH = (
    REPOSITORY_ROOT / "docs" / "releases" / "0.55.1.md"
)
AXIS_PROJECTION_RELEASE_PATH = (
    REPOSITORY_ROOT / "docs" / "releases" / "0.55.2.md"
)
AXIS_ACCEPTANCE_CORRECTION_RELEASE_PATH = (
    REPOSITORY_ROOT / "docs" / "releases" / "0.55.3.md"
)


def test_current_release_uses_version_0553() -> None:
    manifest = tomllib.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["id"] == "blender_to_spine2d_mesh_exporter"
    assert manifest["version"] == "0.55.3"


def test_object_origin_release_keeps_approved_task_document() -> None:
    task = OBJECT_ORIGIN_TASK_PATH.read_text(encoding="utf-8")

    assert "Approved for implementation" in task
    assert "`0.55.0`" in task
    assert "TWO_AXIS_ROTATION_SCALE" in task
    assert "OBJECT_ORIGIN" in task
    assert "Camera Projection" in task


def test_projection_release_keeps_approved_task_document() -> None:
    task = PROJECTION_TASK_PATH.read_text(encoding="utf-8")

    assert "Approved for implementation" in task
    assert "POSITIVE_X" in task
    assert "NEGATIVE_Z" in task
    assert "ACTIVE_CAMERA" in task
    assert "object-block" in task.lower()


def test_object_origin_acceptance_correction_has_release_note() -> None:
    release_note = OBJECT_ORIGIN_CORRECTION_RELEASE_PATH.read_text(encoding="utf-8")

    assert "0.55.1" in release_note
    assert "view_layer.update" in release_note
    assert "production pivot code" in release_note


def test_axis_projection_slice_has_release_note() -> None:
    release_note = AXIS_PROJECTION_RELEASE_PATH.read_text(encoding="utf-8")

    assert "0.55.2" in release_note
    assert "six signed-axis" in release_note
    assert "POSITIVE_Z" in release_note
    assert "Active Camera" in release_note


def test_axis_projection_acceptance_correction_has_release_note() -> None:
    release_note = AXIS_ACCEPTANCE_CORRECTION_RELEASE_PATH.read_text(encoding="utf-8")
    normalized_release_note = release_note.lower()

    assert "0.55.3" in release_note
    assert "mathutils.Matrix @ Vector" in release_note
    assert "2.38418579e-7" in release_note
    assert "production axis projection" in normalized_release_note
