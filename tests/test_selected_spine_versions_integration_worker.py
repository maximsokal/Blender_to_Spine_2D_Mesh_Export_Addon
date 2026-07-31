from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "tests" / "blender_headless" / "run_selected_spine_versions_integration.py"


def _source() -> str:
    return WORKER.read_text(encoding="utf-8")


def test_selected_worker_is_valid_python_and_reuses_production_pipeline() -> None:
    source = _source()

    ast.parse(source, filename=str(WORKER))
    assert "from run_all_spine_versions_integration import" in source
    assert "_export_case" in source
    assert "_prepare_output_directory" in source
    assert 'action="append"' in source
    assert '"selected_blender_acceptance_report.json"' in source
    for forbidden in (
        "serialize_spine_document",
        "SpineSerializer",
        "json.dump(document",
        "write_text(json.dumps(document",
    ):
        assert forbidden not in source


def test_selected_worker_validates_exact_case_key_selection() -> None:
    source = _source()

    assert "def _select_cases(" in source
    assert "by_key = {case.key: case for case in POSITIVE_CASES}" in source
    assert "unknown = tuple(key for key in requested if key not in by_key)" in source
    assert "if len(requested) != len(set(requested)):" in source
    assert "return tuple(by_key[key] for key in requested)" in source


def test_selected_worker_writes_only_selected_case_report() -> None:
    source = _source()

    assert "selected = _select_cases(case_keys)" in source
    assert "cases = tuple(_export_case(output_root, case) for case in selected)" in source
    assert '"selectedCaseKeys": [case.key for case in selected]' in source
    assert '"caseCount": len(cases)' in source
