"""Architecture contract for the real-Blender Spine 3.8 profile worker."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_spine38_standalone_profiles_integration.py"
)


def _source() -> str:
    return WORKER.read_text(encoding="utf-8")


def test_worker_is_valid_python_and_uses_production_multi_export() -> None:
    source = _source()
    tree = ast.parse(source)
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "export_a1_multi_object" in imported_names
    assert "export_a1_multi_object" in called_names
    assert "Spine38JsonCodec" not in imported_names
    assert "SpineSerializer" not in imported_names
    assert "compose_spine_documents" not in imported_names
    assert "build_connected_group_document" not in imported_names


def test_worker_covers_both_profiles_and_exact_target() -> None:
    source = _source()
    for fragment in (
        "SpineJsonTarget.SPINE_3_8.exact_version",
        "A1RigProfile.TWO_AXIS_ROTATION_SCALE",
        "A1RigProfile.THREE_AXIS_ROTATION",
        "mode=A1MultiObjectMode.STANDALONE",
        'output_stem="Spine38TwoAxisStandaloneMulti"',
        'output_stem="Spine38ThreeAxisStandaloneMulti"',
        "expected_bones=55",
        "expected_bones=52",
        "[SPINE38_STANDALONE] PASS production profile exports",
    ):
        assert fragment in source


def test_worker_enforces_legacy_schema_and_profile_topology() -> None:
    source = _source()
    for fragment in (
        '"constraints" not in document',
        '"inherit" not in raw_bone',
        '"referenceScale" not in raw_bone',
        "LEGACY_MIX_FIELDS",
        "FORBIDDEN_NEW_MIX_FIELDS",
        'f"{prefix}_1_scale_spine41_bridge"',
        '"connectedWrapperPresent": False',
        '"crossObjectReferencesPresent": False',
        '"sequencePresent": False',
    ):
        assert fragment in source
