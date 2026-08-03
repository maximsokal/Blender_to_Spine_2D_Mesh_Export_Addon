"""Release contracts for per-object mixed static/sequence export in 0.80.1."""

from __future__ import annotations

from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "Blender_to_Spine2D_Mesh_Exporter" / "blender_manifest.toml"
RELEASE_NOTE = ROOT / "docs" / "releases" / "0.80.1.md"
STANDALONE_RUNNER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_multi_object_mixed_static_sequence_matrix_integration.py"
)
CONNECTED_MIXED_RUNNER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_connected_mixed_static_sequence_matrix_integration.py"
)
ATOMIC_WORK_PATH = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "infrastructure"
    / "atomic_work_path.py"
)
ATOMIC_WORK_STATE = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "infrastructure"
    / "atomic_work_state.py"
)
DURABLE_TRANSACTION = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "infrastructure"
    / "durable_atomic_transaction.py"
)
CURRENT_DOCS = (
    ROOT / "README.md",
    ROOT / "docs" / "README.md",
    ROOT / "docs" / "CHANGELOG.md",
    ROOT / "docs" / "installation.md",
    ROOT / "docs" / "testing.md",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _normalized(value: str) -> str:
    return " ".join(value.lower().replace("-", " ").replace("`", " ").split())


def test_manifest_and_current_documentation_use_version_0801() -> None:
    manifest = tomllib.loads(_read(MANIFEST))

    assert manifest["id"] == "blender_to_spine2d_mesh_exporter"
    assert manifest["version"] == "0.80.1"
    assert manifest["blender_version_min"] == "5.2.0"
    for path in CURRENT_DOCS:
        assert "0.80.1" in _read(path), path.relative_to(ROOT)


def test_release_note_records_standalone_mixed_timing_matrix() -> None:
    note = _read(RELEASE_NOTE)
    normalized = _normalized(note)

    for version in ("3.8", "4.0", "4.1", "4.2", "4.3"):
        assert version in note
    assert "normal uv segments and camera projection" in normalized
    assert "object a: two frame sequence" in normalized
    assert "objects b and c: static textures" in normalized
    assert "static objects do not inherit sequence metadata" in normalized
    assert "four physical png files" in normalized
    assert "128x128" in note


def test_release_note_records_connected_and_mixed_sequence_ownership() -> None:
    note = _read(RELEASE_NOTE)
    normalized = _normalized(note)

    assert "spine 4.2 connected and mixed export" in normalized
    assert "3 axis rotation" in normalized
    assert "2 axis rotation + scale" in normalized
    assert "sequence owner inside the connected subgroup" in normalized
    assert "sequence owner inside the standalone subgroup" in normalized
    assert "every other object remains static" in normalized
    assert "native loop sequence ownership" in normalized


def test_release_note_records_ui_request_contract_and_package() -> None:
    note = _read(RELEASE_NOTE)
    normalized = _normalized(note)

    assert "frames and start" in normalized
    assert "selected mesh object" in normalized
    assert "frames = 0" in normalized
    assert "static current frame output" in normalized
    assert "positive value creates a loop texture sequence only for that object" in normalized
    assert "public export selected objects remains standalone only" in normalized
    assert "scene settings schema remains version 6" in normalized
    assert "blender_to_spine2d_mesh_exporter-0.80.1.zip" in note


def test_release_note_records_windows_safe_atomic_work_paths() -> None:
    note = _read(RELEASE_NOTE)
    normalized = _normalized(note)

    assert "windows safe atomic work paths" in normalized
    assert "240 utf 16 code unit budget" in normalized
    assert "only the repeated final stem is compacted" in normalized
    assert "final json and png filenames are never shortened" in normalized
    assert "choose a shorter directory" in normalized
    assert "compact version 3 ownership token" in normalized
    assert "complete 128 bit nonce" in normalized
    assert "blake2 digest" in normalized
    assert "version 2 work tokens remain readable" in normalized


def test_release_runners_use_real_public_export_boundaries() -> None:
    standalone = _read(STANDALONE_RUNNER)
    connected_mixed = _read(CONNECTED_MIXED_RUNNER)

    assert "export_a1_multi_object(" in standalone
    assert "SpineJsonTarget.SPINE_3_8" in standalone
    assert "SpineJsonTarget.SPINE_4_3" in standalone
    assert "sequence_frame_count=frame_count" in standalone
    assert "static slot inherited" in standalone

    assert "_execute_public_export(" in connected_mixed
    assert "A1MultiObjectMode.CONNECTED" in connected_mixed
    assert "A1MultiObjectMode.MIXED" in connected_mixed
    assert "_SEQUENCE_CONNECTED" in connected_mixed
    assert "_SEQUENCE_STANDALONE" in connected_mixed
    assert "static slot inherited sequence timeline" in connected_mixed


def test_durable_transaction_uses_path_budgeted_stage_and_backup_builders() -> None:
    work_path = _read(ATOMIC_WORK_PATH)
    work_state = _read(ATOMIC_WORK_STATE)
    durable = _read(DURABLE_TRANSACTION)

    assert "WINDOWS_EXTERNAL_IO_PATH_BUDGET = 240" in work_path
    assert "def build_atomic_stage_path(" in work_path
    assert "def build_atomic_backup_path(" in work_path
    assert "reservation_index" in work_path
    assert "Choose a shorter export directory" in work_path

    assert '_TOKEN_VERSION_V2 = "v2"' in work_state
    assert '_TOKEN_VERSION_V3 = "v3"' in work_state
    assert "hashlib.blake2s(" in work_state
    assert "digest_size=_MARKER_DIGEST_BYTES" in work_state
    assert 'token_version=_TOKEN_VERSION_V3' in work_state
    assert "def matches_process_start_marker(" in work_state

    assert "from .atomic_work_path import (" in durable
    assert "build_atomic_stage_path(" in durable
    assert "build_atomic_backup_path(" in durable
    assert "reservation_index=len(self._entries)" in durable
    assert "for reservation_index, entry in enumerate(self._entries)" in durable
