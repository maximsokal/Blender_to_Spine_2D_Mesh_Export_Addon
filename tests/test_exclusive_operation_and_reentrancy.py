from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import ExportResult
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import a1_ui_router
from Blender_to_Spine2D_Mesh_Exporter.infrastructure import (
    A1_EXPORT_OPERATION_KEY,
    OperationAlreadyRunningError,
    active_exclusive_operations,
    exclusive_operation,
)


def test_exclusive_operation_rejects_nested_owner_and_releases_afterwards():
    assert active_exclusive_operations() == ()

    with exclusive_operation("test-operation", label="outer"):
        active = active_exclusive_operations()
        assert len(active) == 1
        assert active[0].key == "test-operation"
        with pytest.raises(OperationAlreadyRunningError) as captured:
            with exclusive_operation("test-operation", label="inner"):
                pytest.fail("nested operation must not acquire the same key")
        assert captured.value.active_label == "outer"
        assert captured.value.requested_label == "inner"

    assert active_exclusive_operations() == ()


def test_exclusive_operation_releases_after_exception_and_allows_retry():
    with pytest.raises(RuntimeError, match="forced operation failure"):
        with exclusive_operation("retry-operation", label="first"):
            raise RuntimeError("forced operation failure")

    assert active_exclusive_operations() == ()
    with exclusive_operation("retry-operation", label="retry"):
        assert active_exclusive_operations()[0].label == "retry"
    assert active_exclusive_operations() == ()


def test_different_operation_keys_can_coexist_without_false_conflict():
    with exclusive_operation("operation-a", label="A"):
        with exclusive_operation("operation-b", label="B"):
            assert tuple(item.key for item in active_exclusive_operations()) == (
                "operation-a",
                "operation-b",
            )
    assert active_exclusive_operations() == ()


def test_ui_router_blocks_progress_callback_reentry_before_second_plan(monkeypatch):
    context = SimpleNamespace(scene=object())
    plan = SimpleNamespace(source_object=object(), settings=object())
    monkeypatch.setattr(a1_ui_router, "build_active_ui_export_plan", lambda _context: plan)

    selected_plan_calls = []

    def unexpected_selected_plan(_context):
        selected_plan_calls.append(True)
        raise AssertionError("reentrant call must fail before request planning")

    monkeypatch.setattr(
        a1_ui_router,
        "build_selected_ui_export_plan",
        unexpected_selected_plan,
    )

    conflicts = []

    def export_single(*_args, **_kwargs):
        with pytest.raises(OperationAlreadyRunningError) as captured:
            a1_ui_router.export_selected_objects_a1(
                context,
                progress_callback=lambda _update: None,
            )
        conflicts.append(captured.value)
        return ExportResult(success=True)

    monkeypatch.setattr(a1_ui_router, "export_a1_single_object", export_single)

    result = a1_ui_router.export_active_object_a1(
        context,
        progress_callback=lambda _update: None,
    )

    assert result.success
    assert len(conflicts) == 1
    assert conflicts[0].key == A1_EXPORT_OPERATION_KEY
    assert selected_plan_calls == []
    assert active_exclusive_operations() == ()


def test_ui_router_releases_guard_when_planning_raises(monkeypatch):
    context = SimpleNamespace(scene=object())

    def fail_plan(_context):
        raise RuntimeError("forced request-plan failure")

    monkeypatch.setattr(a1_ui_router, "build_active_ui_export_plan", fail_plan)
    with pytest.raises(RuntimeError, match="forced request-plan failure"):
        a1_ui_router.export_active_object_a1(
            context,
            progress_callback=lambda _update: None,
        )
    assert active_exclusive_operations() == ()

    plan = SimpleNamespace(source_object=object(), settings=object())
    monkeypatch.setattr(a1_ui_router, "build_active_ui_export_plan", lambda _context: plan)
    monkeypatch.setattr(
        a1_ui_router,
        "export_a1_single_object",
        lambda *_args, **_kwargs: ExportResult(success=True),
    )
    assert a1_ui_router.export_active_object_a1(
        context,
        progress_callback=lambda _update: None,
    ).success
    assert active_exclusive_operations() == ()
