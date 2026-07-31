from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "tests" / "blender_headless" / "run_selected_spine_versions_integration.py"


def _load_worker_module():
    spec = importlib.util.spec_from_file_location("selected_spine_worker", WORKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_selected_worker_reuses_production_export_case_pipeline() -> None:
    source = WORKER.read_text(encoding="utf-8")

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


def test_selected_worker_resolves_exact_case_keys_in_requested_order() -> None:
    module = _load_worker_module()
    keys = (
        "spine_4_2__two_axis_rotation_scale__mixed_multi_object",
        "spine_4_2__two_axis_rotation_scale__connected_multi_object",
    )

    selected = module._select_cases(keys)

    assert tuple(case.key for case in selected) == keys


def test_selected_worker_rejects_unknown_and_duplicate_cases() -> None:
    module = _load_worker_module()

    with pytest.raises(ValueError, match="Unknown production acceptance case"):
        module._select_cases(("not_a_case",))

    key = "spine_4_2__two_axis_rotation_scale__connected_multi_object"
    with pytest.raises(ValueError, match="cannot be repeated"):
        module._select_cases((key, key))
