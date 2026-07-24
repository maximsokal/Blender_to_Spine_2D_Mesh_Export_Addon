"""Focused tests for typed A1 progress events and Blender status-bar ownership."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1ExportProgressUpdate,
    emit_a1_export_progress,
    scale_a1_export_progress_callback,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_blender_progress import (
    BlenderA1ProgressSession,
)


def test_progress_update_validates_percentage_and_object_position():
    with pytest.raises(TypeError, match="percent must be int"):
        A1ExportProgressUpdate(percent=True, stage="TEST", message="Testing")
    with pytest.raises(ValueError, match="\[0, 100\]"):
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
                percent=80,
                stage="STAGE_OUTPUTS",
                message="Staging textures",
                object_id="Body",
                object_index=1,
                object_count=2,
            )
        )

    assert begin_calls == [(0.0, 100.0)]
    assert update_calls == [40.0, 40.0, 80.0]
    assert end_calls == [True]
    assert status_calls[0].startswith("Spine2D export: 0%")
    assert "80% [1/2]" in status_calls[-2]
    assert status_calls[-1] is None
    assert len(redraw_calls) >= 2
