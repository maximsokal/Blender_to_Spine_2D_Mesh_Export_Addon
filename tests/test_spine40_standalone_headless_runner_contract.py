"""Architecture contract for the real-Blender Spine 4.0 standalone worker."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_spine40_standalone_multi_object_integration.py"
)


def test_worker_is_valid_python_and_uses_production_multi_export() -> None:
    source = WORKER.read_text(encoding="utf-8")
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
    assert "Spine40JsonCodec" not in imported_names
    assert "SpineSerializer" not in imported_names


def test_worker_requests_exact_spine40_standalone_scope() -> None:
    source = WORKER.read_text(encoding="utf-8")

    required_fragments = (
        "SpineJsonTarget.SPINE_4_0.exact_version",
        "A1RigProfile.TWO_AXIS_ROTATION_SCALE.value",
        "mode=A1MultiObjectMode.STANDALONE",
        'OUTPUT_STEM = "Spine40StandaloneMulti"',
        "connectedWrapperPresent",
        "crossObjectReferencesPresent",
        "sequencePresent",
        "[SPINE40_STANDALONE] PASS production multi-object export",
    )
    for fragment in required_fragments:
        assert fragment in source


def test_worker_audits_legacy_schema_and_forbidden_sequence_fields() -> None:
    source = WORKER.read_text(encoding="utf-8")

    assert '"referenceScale" not in skeleton' in source
    assert '"physics" not in document' in source
    assert 'not path.endswith(".sequence")' in source
    assert '"inherit" not in item' in source
    assert '"constraints" not in skin' in source
    assert 'for field_name, collection in (("ik", ik), ("transform", transform))' in source


def test_worker_does_not_address_the_external_runtime_repository() -> None:
    source = WORKER.read_text(encoding="utf-8")

    assert "spine-webgl-40" not in source
    assert "Spine2D_curve_optimization" not in source
    assert "runtime-entry" not in source
