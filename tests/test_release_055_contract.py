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
RELEASE_CONTRACT_CORRECTION_PATH = (
    REPOSITORY_ROOT / "docs" / "releases" / "0.55.4.md"
)
PROJECTION_CONTRACT_CORRECTION_PATH = (
    REPOSITORY_ROOT / "docs" / "releases" / "0.55.5.md"
)
STANDALONE_DRAW_ORDER_RELEASE_PATH = (
    REPOSITORY_ROOT / "docs" / "releases" / "0.55.6.md"
)
ACTIVE_CAMERA_RELEASE_PATH = (
    REPOSITORY_ROOT / "docs" / "releases" / "0.55.7.md"
)
ACTIVE_CAMERA_VALIDATION_CORRECTION_PATH = (
    REPOSITORY_ROOT / "docs" / "releases" / "0.55.8.md"
)
ACTIVE_CAMERA_INDEX_CORRECTION_PATH = (
    REPOSITORY_ROOT / "docs" / "releases" / "0.55.9.md"
)
ACTIVE_CAMERA_RELEASE_CONTRACT_CORRECTION_PATH = (
    REPOSITORY_ROOT / "docs" / "releases" / "0.55.10.md"
)


def _normalized_prose(value: str) -> str:
    """Normalize case, whitespace and hyphenation for semantic prose checks."""

    if not isinstance(value, str):
        raise TypeError("value must be str")
    return " ".join(value.lower().replace("-", " ").split())


def test_current_release_uses_version_05510() -> None:
    manifest = tomllib.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["id"] == "blender_to_spine2d_mesh_exporter"
    assert manifest["version"] == "0.55.10"


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
    normalized_release_note = _normalized_prose(release_note)

    assert "0.55.3" in release_note
    assert "mathutils.Matrix @ Vector" in release_note
    assert "2.38418579e-7" in release_note
    assert "production projection introduced" in normalized_release_note
    assert "is unchanged" in normalized_release_note


def test_release_contract_correction_has_release_note() -> None:
    release_note = RELEASE_CONTRACT_CORRECTION_PATH.read_text(encoding="utf-8")
    normalized_release_note = _normalized_prose(release_note)

    assert "0.55.4" in release_note
    assert "exact contiguous phrase" in normalized_release_note
    assert "normalizes case and whitespace" in normalized_release_note
    assert "production code" in normalized_release_note


def test_projection_contract_and_documentation_correction_has_release_note() -> None:
    release_note = PROJECTION_CONTRACT_CORRECTION_PATH.read_text(encoding="utf-8")
    normalized_release_note = _normalized_prose(release_note)

    assert "0.55.5" in release_note
    assert "projection_direction" in release_note
    assert "final seven appended" in normalized_release_note
    assert "public documentation" in normalized_release_note
    assert "production code changed" in normalized_release_note


def test_standalone_draw_order_slice_has_release_note() -> None:
    release_note = STANDALONE_DRAW_ORDER_RELEASE_PATH.read_text(encoding="utf-8")
    normalized_release_note = _normalized_prose(release_note)

    assert "0.55.6" in release_note
    assert "nearest projected vertex" in normalized_release_note
    assert "one contiguous block" in normalized_release_note
    assert "internal segment slot order" in normalized_release_note
    assert "camera projection setup ordering is unchanged" in normalized_release_note
    assert "connected and mixed composition remain unchanged" in normalized_release_note


def test_active_camera_slice_has_release_note() -> None:
    release_note = ACTIVE_CAMERA_RELEASE_PATH.read_text(encoding="utf-8")
    normalized_release_note = _normalized_prose(release_note)

    assert "0.55.7" in release_note
    assert "perspective and orthographic" in normalized_release_note
    assert "export texture dimensions" in normalized_release_note
    assert "camera local depth" in normalized_release_note
    assert "connected and mixed active camera requests remain blocked" in normalized_release_note
    assert "existing rendered camera projection" in normalized_release_note
    assert "public ui does not yet expose" in normalized_release_note


def test_active_camera_validation_correction_has_release_note() -> None:
    release_note = ACTIVE_CAMERA_VALIDATION_CORRECTION_PATH.read_text(
        encoding="utf-8"
    )
    normalized_release_note = _normalized_prose(release_note)

    assert "0.55.8" in release_note
    assert "legacyzgroup" in normalized_release_note
    assert "raw floats" in normalized_release_note
    assert "export texture dimensions" in normalized_release_note
    assert "production active camera projection" in normalized_release_note
    assert "are unchanged" in normalized_release_note


def test_active_camera_index_correction_has_release_note() -> None:
    release_note = ACTIVE_CAMERA_INDEX_CORRECTION_PATH.read_text(encoding="utf-8")
    normalized_release_note = _normalized_prose(release_note)

    assert "0.55.9" in release_note
    assert "profile owned z_index_base" in normalized_release_note
    assert "valid production bindings" in normalized_release_note
    assert "assuming a zero based index" in normalized_release_note
    assert "z_value == 0.0" in release_note
    assert "production active camera projection" in normalized_release_note
    assert "are unchanged" in normalized_release_note


def test_active_camera_release_contract_correction_has_release_note() -> None:
    release_note = ACTIVE_CAMERA_RELEASE_CONTRACT_CORRECTION_PATH.read_text(
        encoding="utf-8"
    )
    normalized_release_note = _normalized_prose(release_note)

    assert "0.55.10" in release_note
    assert "semantic release contract" in normalized_release_note
    assert "exact contiguous phrase" in normalized_release_note
    assert "profile owned z_index_base" in normalized_release_note
    assert "production code" in normalized_release_note
    assert "unchanged" in normalized_release_note
