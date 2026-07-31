from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
MODULE_PATH = TOOLS / "run_all_spine_versions_acceptance.py"


def _load_module(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location(
        "run_all_spine_versions_acceptance_under_test",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runner_routes_every_target_to_an_exact_runtime_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(monkeypatch)

    assert set(module.ORACLE_BY_TARGET) == {
        "SPINE_3_8",
        "SPINE_4_0",
        "SPINE_4_1",
        "SPINE_4_2",
        "SPINE_4_3",
    }
    assert module.ORACLE_BY_TARGET["SPINE_4_0"].name == "spine40_runtime_oracle.mjs"
    assert module.ORACLE_BY_TARGET["SPINE_4_2"].name == "spine42_runtime_oracle.mjs"


def test_parse_json_stdout_accepts_one_prefixed_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(monkeypatch)
    payload = {"ok": True, "version": "4.2.43"}

    assert module._parse_json_stdout(
        "runtime diagnostic\n" + json.dumps(payload),
        label="oracle",
    ) == payload


def test_runtime_report_rejects_non_positive_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(monkeypatch)
    report = {
        "ok": True,
        "version": "4.2.43",
        "counts": {"bones": 2, "setupRenderableAttachments": 1},
        "updateCache": {
            "expectedConstraints": 1,
            "scheduledConstraints": 1,
            "everyConstraintScheduledExactlyOnce": True,
        },
        "matrices": {"finiteBones": 2, "allFinite": True},
        "bounds": {"x": 0.0, "y": 0.0, "width": 0.0, "height": 1.0},
    }

    with pytest.raises(module.AllSpineVersionsAcceptanceError, match="bounds.width"):
        module._validate_runtime_report(
            report,
            target="SPINE_4_2",
            profile="TWO_AXIS_ROTATION_SCALE",
            expected_version="4.2.43",
            object_count=1,
        )


def test_runner_source_keeps_external_runtimes_read_only() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "externalRuntimesReadOnly" in source
    assert "runtime_entry.write" not in source
    assert "runtime_43_root.write" not in source
    assert "copytree(" not in source
