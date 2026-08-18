"""Focused tests for synchronous, diagnostic-only Rewrite readiness."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from Blender_to_Spine2D_Mesh_Exporter import auto_readiness
from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1ExportReadinessReport,
    A1ObjectReadiness,
    A1ReadinessState,
)


ROOT = Path(__file__).resolve().parents[1]
AUTO_READINESS_PATH = ROOT / "Blender_to_Spine2D_Mesh_Exporter" / "auto_readiness.py"


def _mesh_object(name: str):
    return SimpleNamespace(
        type="MESH",
        name=name,
        name_full=name,
        data=SimpleNamespace(name=f"{name}Mesh"),
    )


def _context():
    source = _mesh_object("Hero")
    return SimpleNamespace(
        scene=SimpleNamespace(name="Scene"),
        active_object=source,
        selected_objects=(source,),
    )


def _report() -> A1ExportReadinessReport:
    return A1ExportReadinessReport(
        signature="ready",
        objects=(A1ObjectReadiness(object_id="Hero"),),
    )


@pytest.fixture(autouse=True)
def _clean_process_state(monkeypatch):
    monkeypatch.setattr(auto_readiness, "_REGISTERED", False)
    monkeypatch.setattr(auto_readiness, "_ANALYSIS_RUNNING", False)
    monkeypatch.setattr(auto_readiness, "_UI_MODULE", None)
    auto_readiness._BASE_METHODS.clear()
    monkeypatch.setattr(auto_readiness, "_redraw", lambda: None)
    yield
    auto_readiness._BASE_METHODS.clear()


def test_manual_analysis_runs_once_and_stores_valid_report(monkeypatch):
    context = _context()
    report = _report()
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        auto_readiness._readiness,
        "analyse_a1_export_readiness",
        lambda value: calls.append(("analyse", value)) or report,
    )
    monkeypatch.setattr(
        auto_readiness._readiness,
        "store_a1_export_readiness",
        lambda value, result: calls.append(("store", (value, result))),
    )

    result = auto_readiness.run_a1_readiness_analysis(context, origin="manual")

    assert result is report
    assert calls == [
        ("analyse", context),
        ("store", (context, report)),
    ]
    assert auto_readiness._ANALYSIS_RUNNING is False


def test_manual_analysis_rejects_reentry_without_background_synchronization(monkeypatch):
    context = _context()
    monkeypatch.setattr(auto_readiness, "_ANALYSIS_RUNNING", True)

    with pytest.raises(RuntimeError, match="already running"):
        auto_readiness.run_a1_readiness_analysis(context)


def test_manual_operator_reports_blockers_but_keeps_export_available(monkeypatch):
    context = _context()
    report = SimpleNamespace(
        state=A1ReadinessState.BLOCKED,
        blocker_count=2,
        warning_count=1,
    )
    operator = SimpleNamespace(report=MagicMock())
    monkeypatch.setattr(
        auto_readiness,
        "run_a1_readiness_analysis",
        lambda _context, origin: report,
    )

    result = auto_readiness._manual_execute(operator, context)

    assert result == {"FINISHED"}
    operator.report.assert_called_once_with(
        {"WARNING"},
        "Analysis found 2 blocker(s) and 1 warning(s); export remains available",
    )


class _RecordingLayout:
    def __init__(self) -> None:
        self.labels: list[dict[str, object]] = []
        self.operators: list[tuple[str, dict[str, object]]] = []

    def box(self):
        return self

    def row(self, **_kwargs):
        return self

    def label(self, **kwargs) -> None:
        self.labels.append(dict(kwargs))

    def operator(self, operator_id: str, **kwargs):
        self.operators.append((operator_id, dict(kwargs)))
        return SimpleNamespace()


def test_not_analyzed_ui_requires_explicit_analyze_and_never_blocks_export(monkeypatch):
    context = _context()
    monkeypatch.setattr(
        auto_readiness._readiness,
        "current_a1_export_readiness",
        lambda _context: (A1ReadinessState.NOT_ANALYSED, None),
    )
    panel = SimpleNamespace()
    layout = _RecordingLayout()

    allowed = auto_readiness._draw_nonblocking(panel, layout, context)

    assert allowed is True
    texts = tuple(str(item.get("text", "")) for item in layout.labels)
    assert "Run Analyze for diagnostics" in texts
    assert "Export remains available" in texts
    assert not any("automatic" in text.casefold() for text in texts)


def test_blocked_diagnostics_never_disable_export(monkeypatch):
    context = _context()
    report = SimpleNamespace(
        blocker_count=1,
        warning_count=0,
        issues=(),
        objects=(),
    )
    monkeypatch.setattr(
        auto_readiness._readiness,
        "current_a1_export_readiness",
        lambda _context: (A1ReadinessState.BLOCKED, report),
    )
    panel = SimpleNamespace(
        _state_icon=lambda _state: "CANCEL",
        _issue_icon=lambda _severity: "INFO",
        _draw_object_readiness=lambda _layout, _item: None,
    )
    layout = _RecordingLayout()

    allowed = auto_readiness._draw_nonblocking(panel, layout, context)

    assert allowed is True
    assert any(
        item.get("text") == "Diagnostics do not disable production export"
        for item in layout.labels
    )


def test_single_export_calls_production_export_directly(monkeypatch):
    context = _context()
    expected = object()
    auto_readiness._UI_MODULE = SimpleNamespace(
        export_active_object_a1=lambda value: expected if value is context else None,
    )
    operator = SimpleNamespace(
        report=MagicMock(),
        _report_result=MagicMock(return_value={"FINISHED"}),
    )

    result = auto_readiness._single_execute(operator, context)

    assert result == {"FINISHED"}
    operator._report_result.assert_called_once_with(expected)


def test_bridge_register_and_unregister_restore_exact_ui_methods(monkeypatch):
    owners = {
        "panel": SimpleNamespace(_draw_readiness=object()),
        "manual": SimpleNamespace(execute=object()),
        "single": SimpleNamespace(execute=object()),
        "multi": SimpleNamespace(execute=object()),
        "guard": SimpleNamespace(_require_readiness=object()),
    }
    originals = {
        "draw": owners["panel"]._draw_readiness,
        "manual": owners["manual"].execute,
        "single": owners["single"].execute,
        "multi": owners["multi"].execute,
        "guard": owners["guard"]._require_readiness,
    }
    fake_ui = SimpleNamespace(
        OBJECT_PT_Spine2DMeshPanel=owners["panel"],
        OBJECT_OT_Spine2DRefreshInfo=owners["manual"],
        OBJECT_OT_Spine2DSingleExport=owners["single"],
        OBJECT_OT_Spine2DMultiExport=owners["multi"],
        _Spine2DExportOperatorMixin=owners["guard"],
    )

    auto_readiness._patch_ui(fake_ui)
    assert owners["panel"]._draw_readiness is auto_readiness._draw_nonblocking
    assert owners["manual"].execute is auto_readiness._manual_execute
    assert owners["single"].execute is auto_readiness._single_execute
    assert owners["multi"].execute is auto_readiness._multi_execute
    assert owners["guard"]._require_readiness is auto_readiness._never_blocks

    auto_readiness._restore_ui(fake_ui)
    assert owners["panel"]._draw_readiness is originals["draw"]
    assert owners["manual"].execute is originals["manual"]
    assert owners["single"].execute is originals["single"]
    assert owners["multi"].execute is originals["multi"]
    assert owners["guard"]._require_readiness is originals["guard"]
    assert auto_readiness._BASE_METHODS == {}


def test_shipped_bridge_contains_no_scheduler_or_automatic_callbacks():
    source = AUTO_READINESS_PATH.read_text(encoding="utf-8")

    for forbidden in (
        "bpy.app.timers",
        "monotonic",
        "_automatic_timer",
        "request_auto_analysis",
        "a1_auto_readiness_depsgraph_update_post",
        "a1_auto_readiness_load_pre",
        "a1_auto_readiness_load_post",
        "_install_handlers",
        "_register_timer",
    ):
        assert forbidden not in source
