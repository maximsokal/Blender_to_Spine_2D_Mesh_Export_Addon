"""Focused tests for automatic, diagnostic-only Rewrite readiness."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from Blender_to_Spine2D_Mesh_Exporter import auto_readiness
from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1ExportReadinessReport,
    A1ObjectReadiness,
    A1ReadinessState,
)


def _mesh_object(name: str, pointer: int):
    data = SimpleNamespace(
        id_type="MESH",
        name=f"{name}Mesh",
        name_full=f"{name}Mesh",
        as_pointer=lambda: pointer + 10_000,
    )
    return SimpleNamespace(
        id_type="OBJECT",
        type="MESH",
        name=name,
        name_full=name,
        as_pointer=lambda: pointer,
        data=data,
        spine2d_bake_settings=SimpleNamespace(
            bake_frame_start=0,
            frames_for_render=0,
        ),
        spine2d_connect_settings=SimpleNamespace(enabled=False),
    )


def _context(pointer: int = 100):
    source = _mesh_object("Hero", pointer)
    scene = SimpleNamespace(
        id_type="SCENE",
        name="Scene",
        name_full="Scene",
        as_pointer=lambda: pointer + 20_000,
        frame_current=0,
        camera=None,
        render=SimpleNamespace(engine="BLENDER_EEVEE"),
        spine2d_texture_size=1024,
        spine2d_json_path="//exports",
        spine2d_images_path="images",
        spine2d_control_icons=True,
        spine2d_export_preview_animation=True,
        spine2d_seam_maker_mode="AUTO",
        spine2d_angle_limit=30.0,
        spine2d_angular_mode="SEED_CONE",
        spine2d_local_angle_limit=30.0,
        spine2d_frames_for_render=0,
        spine2d_bake_frame_start=0,
        spine2d_material_source_policy="REQUIRE_SOURCE",
        spine2d_generated_material_pattern="SOLID_GRAY",
        spine2d_generated_gray_color=(0.5, 0.5, 0.5),
        spine2d_projection_alpha_threshold=1.0 / 255.0,
    )
    return SimpleNamespace(
        scene=scene,
        active_object=source,
        selected_objects=(source,),
        mode="OBJECT",
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
    monkeypatch.setattr(auto_readiness, "_ANALYSIS_ORIGIN", None)
    monkeypatch.setattr(auto_readiness, "_EXPORT_DEPTH", 0)
    monkeypatch.setattr(auto_readiness, "_FILE_LOADING", False)
    monkeypatch.setattr(auto_readiness, "_LAST_KEY", None)
    monkeypatch.setattr(auto_readiness, "_FAILED_KEY", None)
    monkeypatch.setattr(auto_readiness, "_LAST_ERROR", None)
    monkeypatch.setattr(auto_readiness, "_UI_MODULE", None)
    monkeypatch.setattr(auto_readiness, "_redraw", lambda: None)
    auto_readiness._cancel_pending()
    yield
    auto_readiness._cancel_pending()


def test_request_key_tracks_selection_and_rewrite_settings():
    context = _context(101)
    first = auto_readiness._request_key(context)

    context.scene.spine2d_texture_size = 2048
    second = auto_readiness._request_key(context)

    other = _mesh_object("Other", 202)
    context.active_object = other
    context.selected_objects = (other,)
    third = auto_readiness._request_key(context)

    assert first is not None
    assert second is not None
    assert third is not None
    assert first != second
    assert second != third


def test_debounce_coalesces_repeated_requests(monkeypatch):
    context = _context(201)
    times = iter((10.0, 10.2))
    monkeypatch.setattr(auto_readiness, "monotonic", lambda: next(times))

    assert auto_readiness.request_auto_analysis(context, reason="first") is True
    first_deadline = auto_readiness._PENDING_DEADLINE
    assert auto_readiness.request_auto_analysis(context, reason="second") is True

    assert auto_readiness._PENDING is True
    assert auto_readiness._PENDING_REASON == "second"
    assert auto_readiness._PENDING_DEADLINE > first_deadline


def test_manual_and_automatic_paths_share_one_analysis_service(monkeypatch):
    context = _context(301)
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


def test_manual_operator_reports_blockers_but_keeps_export_available(monkeypatch):
    context = _context(401)
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


def test_blocked_diagnostics_never_disable_export(monkeypatch):
    context = _context(501)
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
    monkeypatch.setattr(
        auto_readiness,
        "current_auto_readiness_status",
        lambda _context: auto_readiness.AutoReadinessStatus("IDLE", ""),
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


def test_single_export_bypasses_readiness_guard(monkeypatch):
    context = _context(601)
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


def test_timer_registration_is_persistent_and_symmetric(monkeypatch):
    registered: set[object] = set()
    calls: list[tuple[str, object]] = []

    class _Timers:
        @staticmethod
        def is_registered(callback):
            return callback in registered

        @staticmethod
        def register(callback, **kwargs):
            registered.add(callback)
            calls.append(("register", kwargs))

        @staticmethod
        def unregister(callback):
            registered.remove(callback)
            calls.append(("unregister", callback))

    monkeypatch.setattr(
        auto_readiness,
        "bpy",
        SimpleNamespace(app=SimpleNamespace(timers=_Timers())),
    )

    auto_readiness._register_timer()
    auto_readiness._register_timer()
    auto_readiness._unregister_timer()

    assert calls[0] == (
        "register",
        {
            "first_interval": auto_readiness._AUTO_POLL_SECONDS,
            "persistent": True,
        },
    )
    assert calls[1][0] == "unregister"
    assert len(calls) == 2


def test_handler_registration_deduplicates_and_removes(monkeypatch):
    handlers = SimpleNamespace(
        depsgraph_update_post=[],
        load_pre=[],
        load_post=[],
    )
    monkeypatch.setattr(
        auto_readiness,
        "bpy",
        SimpleNamespace(app=SimpleNamespace(handlers=handlers)),
    )

    auto_readiness._install_handlers()
    auto_readiness._install_handlers()

    assert handlers.depsgraph_update_post == [
        auto_readiness.a1_auto_readiness_depsgraph_update_post
    ]
    assert handlers.load_pre == [auto_readiness.a1_auto_readiness_load_pre]
    assert handlers.load_post == [auto_readiness.a1_auto_readiness_load_post]

    auto_readiness._remove_handlers()

    assert handlers.depsgraph_update_post == []
    assert handlers.load_pre == []
    assert handlers.load_post == []
