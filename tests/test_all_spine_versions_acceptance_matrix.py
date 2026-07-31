from __future__ import annotations

from dataclasses import FrozenInstanceError
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "spine_version_acceptance_matrix.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "spine_version_acceptance_matrix_under_test",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_positive_matrix_has_exact_twenty_case_contract() -> None:
    module = _load_module()

    assert len(module.POSITIVE_CASES) == 20
    assert dict(module.EXPECTED_CASE_COUNT_BY_TARGET) == {
        "SPINE_3_8": 4,
        "SPINE_4_0": 2,
        "SPINE_4_1": 2,
        "SPINE_4_2": 8,
        "SPINE_4_3": 4,
    }
    assert dict(module.EXACT_VERSION_BY_TARGET) == {
        "SPINE_3_8": "3.8.99",
        "SPINE_4_0": "4.0.64",
        "SPINE_4_1": "4.1.24",
        "SPINE_4_2": "4.2.43",
        "SPINE_4_3": "4.3.23",
    }
    assert len({case.key for case in module.POSITIVE_CASES}) == 20


def test_matrix_matches_current_fail_closed_capability_shape() -> None:
    module = _load_module()
    triples = {(case.target, case.profile, case.scope) for case in module.POSITIVE_CASES}

    assert (
        "SPINE_3_8",
        "THREE_AXIS_ROTATION",
        "CONNECTED_MULTI_OBJECT",
    ) not in triples
    assert (
        "SPINE_4_0",
        "THREE_AXIS_ROTATION",
        "SINGLE_OBJECT",
    ) not in triples
    assert (
        "SPINE_4_1",
        "TWO_AXIS_ROTATION_SCALE",
        "CONNECTED_MULTI_OBJECT",
    ) not in triples
    assert (
        "SPINE_4_2",
        "THREE_AXIS_ROTATION",
        "MIXED_MULTI_OBJECT",
    ) in triples
    assert (
        "SPINE_4_2",
        "TWO_AXIS_ROTATION_SCALE",
        "CONNECTED_MULTI_OBJECT",
    ) in triples
    assert (
        "SPINE_4_3",
        "THREE_AXIS_ROTATION",
        "MIXED_MULTI_OBJECT",
    ) not in triples


def test_matrix_cases_are_immutable_and_scope_counts_are_valid() -> None:
    module = _load_module()
    case = module.POSITIVE_CASES[0]

    with pytest.raises(FrozenInstanceError):
        case.object_count = 99

    expected_counts = {
        "SINGLE_OBJECT": 1,
        "STANDALONE_MULTI_OBJECT": 3,
        "CONNECTED_MULTI_OBJECT": 3,
        "MIXED_MULTI_OBJECT": 3,
    }
    assert all(case.object_count == expected_counts[case.scope] for case in module.POSITIVE_CASES)
