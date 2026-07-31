from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "tests" / "blender_headless" / "run_all_spine_versions_integration.py"


def test_worker_uses_only_public_production_export_services() -> None:
    source = WORKER.read_text(encoding="utf-8")

    for fragment in (
        "export_a1_single_object",
        "export_a1_multi_object",
        "export_a1_mixed_object",
        "require_spine_json_export_capability",
        '"accepted": accepted',
        '"blocked": blocked',
    ):
        assert fragment in source
    assert "serialize_spine_document" not in source
    assert "write_staged_utf8_text" not in source
    assert "prepare_a1_object(" not in source


def test_worker_covers_single_standalone_connected_and_mixed() -> None:
    source = WORKER.read_text(encoding="utf-8")

    for scope in (
        "SINGLE_OBJECT",
        "STANDALONE_MULTI_OBJECT",
        "CONNECTED_MULTI_OBJECT",
        "MIXED_MULTI_OBJECT",
    ):
        assert scope in source
    assert "[SPINE_ALL_VERSIONS] RUN 20 production export cases" in source
