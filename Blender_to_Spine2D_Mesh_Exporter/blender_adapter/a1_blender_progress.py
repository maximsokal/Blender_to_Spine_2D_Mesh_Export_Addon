# pylint: disable=import-error
"""Blender UI adapter for typed A1 export progress events.

The export remains synchronous, but Blender's WindowManager progress API updates the
status-bar progress indicator at every real pipeline boundary. The adapter is deliberately
best-effort: UI failures are logged and never alter export correctness.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
import logging
from typing import Any

from ..application import A1ExportProgressCallback, A1ExportProgressUpdate


logger = logging.getLogger(__name__)


def _tag_redraw(context: Any) -> None:
    screen = getattr(context, "screen", None)
    areas = tuple(getattr(screen, "areas", ())) if screen is not None else ()
    if not areas:
        area = getattr(context, "area", None)
        areas = () if area is None else (area,)
    for area in areas:
        tag_redraw = getattr(area, "tag_redraw", None)
        if callable(tag_redraw):
            try:
                tag_redraw()
            except Exception:
                logger.debug("Unable to tag Blender area for redraw", exc_info=True)


def _workspace(context: Any) -> Any | None:
    workspace = getattr(context, "workspace", None)
    if workspace is not None:
        return workspace
    window = getattr(context, "window", None)
    return None if window is None else getattr(window, "workspace", None)


class BlenderA1ProgressSession(AbstractContextManager[A1ExportProgressCallback]):
    """Own one balanced ``progress_begin/update/end`` lifecycle."""

    def __init__(self, context: Any, *, operation_name: str) -> None:
        if context is None:
            raise ValueError("context cannot be None")
        normalized_name = str(operation_name).strip()
        if not normalized_name:
            raise ValueError("operation_name must be a non-empty string")
        self._context = context
        self._operation_name = normalized_name
        self._window_manager = getattr(context, "window_manager", None)
        self._workspace = _workspace(context)
        self._begun = False
        self._last_percent = 0

    def _set_status_text(self, text: str | None) -> None:
        setter = getattr(self._workspace, "status_text_set", None)
        if callable(setter):
            try:
                setter(text)
            except Exception:
                logger.debug("Unable to update Blender workspace status text", exc_info=True)

    def _begin(self) -> None:
        begin = getattr(self._window_manager, "progress_begin", None)
        if callable(begin):
            try:
                begin(0.0, 100.0)
                self._begun = True
            except Exception:
                logger.debug("Unable to begin Blender progress display", exc_info=True)
        self._set_status_text(f"{self._operation_name}: 0% — Starting")
        _tag_redraw(self._context)

    def _finish(self) -> None:
        try:
            end = getattr(self._window_manager, "progress_end", None)
            if self._begun and callable(end):
                end()
        except Exception:
            logger.debug("Unable to end Blender progress display", exc_info=True)
        finally:
            self._begun = False
            self._set_status_text(None)
            _tag_redraw(self._context)

    def __enter__(self) -> A1ExportProgressCallback:
        self._begin()
        return self.update

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self._finish()
        return False

    def update(self, update: A1ExportProgressUpdate) -> None:
        if not isinstance(update, A1ExportProgressUpdate):
            raise TypeError("update must be A1ExportProgressUpdate")
        # Guard the status bar against accidental regressions from nested pipelines.
        percent = max(self._last_percent, int(update.percent))
        self._last_percent = min(100, percent)
        progress_update = getattr(self._window_manager, "progress_update", None)
        if callable(progress_update):
            try:
                progress_update(float(self._last_percent))
            except Exception:
                logger.debug("Unable to update Blender progress display", exc_info=True)

        object_text = ""
        if update.object_index is not None and update.object_count is not None:
            object_text = f" [{update.object_index}/{update.object_count}]"
        elif update.object_id:
            object_text = f" [{update.object_id}]"
        self._set_status_text(
            f"{self._operation_name}: {self._last_percent}%{object_text} — {update.message}"
        )
        _tag_redraw(self._context)


def blender_a1_progress_session(
    context: Any,
    *,
    operation_name: str,
) -> BlenderA1ProgressSession:
    """Construct a balanced progress session for one UI-triggered export."""

    return BlenderA1ProgressSession(context, operation_name=operation_name)


__all__ = ["BlenderA1ProgressSession", "blender_a1_progress_session"]
