"""Architecture contract for the real-Blender Spine 4.3 profile worker."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_spine43_standalone_profiles_integration.py"
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
    assert "build_connected_group_document" not in imported_names
    assert "compose_spine_documents" not in imported_names
    assert "Spine43JsonCodec" not in imported_names
    assert "SpineSerializer" not in imported_names
    assert "bpy.ops" not in source


def test_worker_covers_both_profiles_through_exact_spine43_target() -> None:
    source = _source()

    required_fragments = (
        "SpineJsonTarget.SPINE_4_3.exact_version",
        "A1RigProfile.TWO_AXIS_ROTATION_SCALE",
        "A1RigProfile.THREE_AXIS_ROTATION",
        "mode=A1MultiObjectMode.STANDALONE",
        'output_stem="Spine43TwoAxisStandaloneMulti"',
        'output_stem="Spine43ThreeAxisStandaloneMulti"',
        "expected_bones=52",
        "expected_bones=46",
        "[SPINE43_STANDALONE] PASS production profile exports",
    )
    for fragment in required_fragments:
        assert fragment in source


def test_worker_enforces_unified_constraint_schema_and_ownership() -> None:
    source = _source()

    for fragment in (
        'legacy_collection in ("ik", "transform", "path", "physics", "slider")',
        '_json_array(document, "constraints", required=True)',
        'constraint_type in {"ik", "transform"}',
        '"order" not in raw_constraint',
        '"local" not in raw_constraint',
        '"relative" not in raw_constraint',
        '"source" not in raw_constraint',
        '"target" not in raw_constraint',
        '_assert_transform_properties(raw_constraint, name=name)',
        '"connectedWrapperPresent": False',
        '"crossObjectReferencesPresent": False',
    ):
        assert fragment in source


def test_worker_does_not_address_or_claim_an_external_spine43_runtime() -> None:
    source = _source()

    assert "spine-webgl-43" not in source
    assert "Spine2D_curve_optimization" not in source
    assert "runtime-entry" not in source
    assert '"runtimeValidated": False' in source
    assert '"manualEditorImportRequired": True' in source
