"""Focused tests for typed A1 progress events and Blender status-bar ownership."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1ExportProgressUpdate,
    a1_frame_progress_percent,
    emit_a1_export_progress,
    emit_a1_frame_progress,
    scale_a1_export_progress_callback,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import a1_ui_router
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_blender_progress import (
    BlenderA1ProgressSession,
)


def test_progress_update_validates_percentage_and_object_position():
    with pytest.raises(TypeError, match="percent must be int"):
        A1ExportProgressUpdate(percent=True, stage="TEST", message="Testing")
    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        A1ExportProgressUpdate(percent=101, stage="TEST", message="Testing")
    with pytest.raises(ValueError, match="provided together"):
        A1ExportProgressUpdate(
            percent=10,
            stage="TEST",
            message="Testing",
            object_index=1,
        )
    with pytest.raises(ValueError, match="boundary whitespace"):
        A1ExportProgressUpdate(percent=10, stage=" TEST", message="Testing")


def test_frame_progress_counts_only_completed_physical_frames():
    assert a1_frame_progress_percent(1, 60, completed=False) == 0
    assert a1_frame_progress_percent(17, 60, completed=False) == 27
    assert a1_frame_progress_percent(17, 60, completed=True) == 28
    assert a1_frame_progress_percent(60, 60, completed=False) == 98
    assert a1_frame_progress_percent(60, 60, completed=True) == 100

    with pytest.raises(TypeError, match="frame_index must be int"):
        a1_frame_progress_percent(True, 60, completed=False)
    with pytest.raises(ValueError, match="frame_count must be positive"):
        a1_frame_progress_percent(1, 0, completed=False)
    with pytest.raises(ValueError, match=r"\[1, frame_count\]"):
        a1_frame_progress_percent(61, 60, completed=False)
    with pytest.raises(TypeError, match="completed must be bool"):
        a1_frame_progress_percent(1, 60, completed=1)


def test_frame_progress_emits_exact_ui_message_and_object_id():
    received: list[A1ExportProgressUpdate] = []

    emit_a1_frame_progress(
        received.append,
        stage="BAKE_FRAME",
        action="Baking",
        frame_index=17,
        frame_count=60,
        completed=False,
        object_id="Body",
    )

    assert received == [
        A1ExportProgressUpdate(
            percent=27,
            stage="BAKE_FRAME",
            message="Baking frame 17/60",
            object_id="Body",
        )
    ]


def test_scaled_progress_maps_child_range_and_metadata():
    received: list[A1ExportProgressUpdate] = []
    scaled = scale_a1_export_progress_callback(
        received.append,
        start_percent=20.0,
        end_percent=60.0,
        object_id="Component",
        object_index=2,
        object_count=4,
        message_prefix="Preparing: ",
    )
    assert scaled is not None

    scaled(
        A1ExportProgressUpdate(
            percent=25,
            stage="READ_GEOMETRY",
            message="Reading geometry",
        )
    )

    assert received == [
        A1ExportProgressUpdate(
            percent=30,
            stage="READ_GEOMETRY",
            message="Preparing: Reading geometry",
            object_id="Component",
            object_index=2,
            object_count=4,
        )
    ]


def test_scaled_frame_progress_preserves_message_and_maps_percentage():
    received: list[A1ExportProgressUpdate] = []
    scaled = scale_a1_export_progress_callback(
        received.append,
        start_percent=65.0,
        end_percent=80.0,
        object_index=2,
        object_count=4,
    )
    assert scaled is not None

    emit_a1_frame_progress(
        scaled,
        stage="BAKE_FRAME",
        action="Baking",
        frame_index=17,
        frame_count=60,
        completed=False,
        object_id="Body",
    )

    assert received == [
        A1ExportProgressUpdate(
            percent=69,
            stage="BAKE_FRAME",
            message="Baking frame 17/60",
            object_id="Body",
            object_index=2,
            object_count=4,
        )
    ]


def test_progress_observer_failure_never_escapes_export_path():
    calls: list[A1ExportProgressUpdate] = []

    def broken(update: A1ExportProgressUpdate) -> None:
        calls.append(update)
        raise RuntimeError("UI failed")

    emit_a1_export_progress(
        broken,
        percent=50,
        stage="STAGE_OUTPUTS",
        message="Staging textures",
    )

    assert len(calls) == 1
    assert calls[0].percent == 50


def test_blender_progress_session_balances_lifecycle_and_clamps_regressions():
    begin_calls: list[tuple[float, float]] = []
    update_calls: list[float] = []
    end_calls: list[bool] = []
    status_calls: list[str | None] = []
    redraw_calls: list[bool] = []

    window_manager = SimpleNamespace(
        progress_begin=lambda minimum, maximum: begin_calls.append((minimum, maximum)),
        progress_update=lambda value: update_calls.append(value),
        progress_end=lambda: end_calls.append(True),
    )
    workspace = SimpleNamespace(status_text_set=lambda value: status_calls.append(value))
    area = SimpleNamespace(tag_redraw=lambda: redraw_calls.append(True))
    context = SimpleNamespace(
        window_manager=window_manager,
        workspace=workspace,
        screen=SimpleNamespace(areas=(area,)),
    )

    with BlenderA1ProgressSession(context, operation_name="Spine2D export") as callback:
        callback(
            A1ExportProgressUpdate(
                percent=40,
                stage="PREPARE_GEOMETRY",
                message="Preparing geometry",
            )
        )
        callback(
            A1ExportProgressUpdate(
                percent=30,
                stage="PREPARE_GEOMETRY",
                message="Late nested event",
            )
        )
        callback(
            A1ExportProgressUpdate(
                percent=71,
                stage="BAKE_FRAME",
                message="Baking frame 17/60",
                object_id="Body",
                object_index=1,
                object_count=2,
            )
        )

    assert begin_calls == [(0.0, 100.0)]
    assert update_calls == [40.0, 40.0, 71.0]
    assert end_calls == [True]
    assert status_calls[0].startswith("Spine2D export: 0%")
    assert "71% [1/2] — Baking frame 17/60" in status_calls[-2]
    assert status_calls[-1] is None
    assert len(redraw_calls) >= 2


def test_ui_router_forwards_custom_callback_without_owning_blender_session(monkeypatch):
    source_object = object()
    settings = object()
    expected_result = object()
    callback = lambda _update: None
    captured: dict[str, object] = {}
    context = SimpleNamespace(scene=object())

    monkeypatch.setattr(
        a1_ui_router,
        "build_active_ui_export_plan",
        lambda _context: SimpleNamespace(
            source_object=source_object,
            settings=settings,
        ),
    )

    def export_single(resolved_object, resolved_settings, **kwargs):
        captured.update(
            source_object=resolved_object,
            settings=resolved_settings,
            kwargs=kwargs,
        )
        return expected_result

    monkeypatch.setattr(a1_ui_router, "export_a1_single_object", export_single)
    monkeypatch.setattr(
        a1_ui_router,
        "blender_a1_progress_session",
        lambda *_args, **_kwargs: pytest.fail(
            "custom callback must bypass Blender progress ownership"
        ),
    )

    result = a1_ui_router.export_active_object_a1(
        context,
        progress_callback=callback,
    )

    assert result is expected_result
    assert captured["source_object"] is source_object
    assert captured["settings"] is settings
    assert captured["kwargs"] == {
        "context": context,
        "scene": context.scene,
        "progress_callback": callback,
    }
