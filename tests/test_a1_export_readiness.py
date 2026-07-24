"""Focused tests for typed A1 readiness state and Blender-session caching."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1ExportReadinessReport,
    A1ObjectReadiness,
    A1ReadinessState,
    ExportIssue,
    IssueSeverity,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import a1_export_readiness


def _issue(
    severity: IssueSeverity,
    *,
    code: str,
    object_id: str | None = None,
) -> ExportIssue:
    return ExportIssue(
        severity=severity,
        stage="TEST",
        code=code,
        message=code,
        object_id=object_id,
    )


def _report(
    *,
    signature: str = "signature",
    object_issues=(),
    global_issues=(),
) -> A1ExportReadinessReport:
    return A1ExportReadinessReport(
        signature=signature,
        objects=(
            A1ObjectReadiness(
                object_id="Mesh",
                issues=tuple(object_issues),
                statistics={"source_vertices": 4},
            ),
        ),
        issues=tuple(global_issues),
        statistics={"object_count": 1},
    )


def test_readiness_state_uses_errors_as_blockers_and_warnings_as_non_blocking():
    ready = _report()
    warning = _report(
        object_issues=(_issue(IssueSeverity.WARNING, code="WARN", object_id="Mesh"),)
    )
    blocked = _report(
        global_issues=(_issue(IssueSeverity.ERROR, code="BLOCK"),)
    )

    assert ready.state is A1ReadinessState.READY
    assert ready.can_export
    assert warning.state is A1ReadinessState.WARNING
    assert warning.can_export
    assert warning.warning_count == 1
    assert blocked.state is A1ReadinessState.BLOCKED
    assert not blocked.can_export
    assert blocked.blocker_count == 1


def test_object_readiness_rejects_foreign_issue_and_invalid_statistics():
    with pytest.raises(ValueError, match="match object_id"):
        A1ObjectReadiness(
            object_id="Mesh",
            issues=(_issue(IssueSeverity.WARNING, code="FOREIGN", object_id="Other"),),
        )
    with pytest.raises(TypeError, match="finite float"):
        A1ObjectReadiness(object_id="Mesh", statistics={"flag": True})
    with pytest.raises(ValueError, match="finite"):
        A1ObjectReadiness(object_id="Mesh", statistics={"value": float("nan")})


def test_readiness_report_rejects_duplicate_object_ids():
    item = A1ObjectReadiness(object_id="Mesh")

    with pytest.raises(ValueError, match="unique object_id"):
        A1ExportReadinessReport(
            signature="signature",
            objects=(item, item),
        )


def test_cached_report_becomes_stale_when_request_signature_changes(monkeypatch):
    scene = SimpleNamespace(as_pointer=lambda: 101)
    context = SimpleNamespace(scene=scene)
    report = _report(signature="first")
    monkeypatch.setattr(
        a1_export_readiness,
        "build_a1_readiness_signature",
        lambda _context: "first",
    )
    a1_export_readiness.clear_a1_export_readiness()
    try:
        a1_export_readiness.store_a1_export_readiness(context, report)
        state, current = a1_export_readiness.current_a1_export_readiness(context)
        assert state is A1ReadinessState.READY
        assert current is report

        monkeypatch.setattr(
            a1_export_readiness,
            "build_a1_readiness_signature",
            lambda _context: "changed",
        )
        state, current = a1_export_readiness.current_a1_export_readiness(context)
        assert state is A1ReadinessState.STALE
        assert current is report
    finally:
        a1_export_readiness.clear_a1_export_readiness()


def test_depsgraph_update_invalidates_cached_report(monkeypatch):
    scene = SimpleNamespace(as_pointer=lambda: 202)
    context = SimpleNamespace(scene=scene)
    report = _report(signature="same")
    monkeypatch.setattr(
        a1_export_readiness,
        "build_a1_readiness_signature",
        lambda _context: "same",
    )
    a1_export_readiness.clear_a1_export_readiness()
    try:
        a1_export_readiness.store_a1_export_readiness(context, report)
        depsgraph = SimpleNamespace(
            updates=(SimpleNamespace(id=SimpleNamespace(id_type="MESH")),)
        )
        a1_export_readiness.a1_readiness_depsgraph_update_post(scene, depsgraph)

        state, current = a1_export_readiness.current_a1_export_readiness(context)
        assert state is A1ReadinessState.STALE
        assert current is report
    finally:
        a1_export_readiness.clear_a1_export_readiness()


def test_export_guard_requires_current_non_blocked_report(monkeypatch):
    scene = SimpleNamespace(as_pointer=lambda: 303)
    context = SimpleNamespace(scene=scene)
    monkeypatch.setattr(
        a1_export_readiness,
        "build_a1_readiness_signature",
        lambda _context: "blocked",
    )
    blocked = _report(
        signature="blocked",
        global_issues=(_issue(IssueSeverity.ERROR, code="BLOCK"),),
    )
    a1_export_readiness.clear_a1_export_readiness()
    try:
        allowed, message = a1_export_readiness.require_current_a1_export_readiness(
            context
        )
        assert not allowed
        assert "Analyze" in message

        a1_export_readiness.store_a1_export_readiness(context, blocked)
        allowed, message = a1_export_readiness.require_current_a1_export_readiness(
            context
        )
        assert not allowed
        assert "blocked by 1" in message
    finally:
        a1_export_readiness.clear_a1_export_readiness()
