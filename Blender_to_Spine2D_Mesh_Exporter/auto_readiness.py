# pylint: disable=import-error
"""Manual Rewrite readiness diagnostics with non-blocking export UI.

This module intentionally owns no Blender timer, depsgraph handler, load handler, Python
worker, debounce queue, or deferred analysis state. Analyze is user-triggered and runs
synchronously on Blender's main thread. The only lifecycle resource owned here is a
small set of reversible method overrides that keep readiness diagnostics advisory rather
than making them a prerequisite for Export.
"""

from __future__ import annotations

import logging
from typing import Any

import bpy

from .application import A1ExportReadinessReport, A1ReadinessState
from .blender_adapter import a1_export_readiness as _readiness
from .blender_adapter import a1_readiness_invalidation as _invalidation


logger = logging.getLogger(__name__)

_REGISTERED = False
_ANALYSIS_RUNNING = False
_UI_MODULE: Any | None = None
_BASE_METHODS: dict[str, Any] = {}


def _tuple(value: Any) -> tuple[Any, ...]:
    """Return a stable tuple for Blender RNA collections and test doubles."""

    try:
        return tuple(value or ())
    except Exception:
        return ()


def _has_mesh_request(context: Any) -> bool:
    """Return whether the current request contains at least one usable Mesh object."""

    if context is None:
        return False
    active = getattr(context, "active_object", None)
    if (
        active is not None
        and getattr(active, "type", None) == "MESH"
        and getattr(active, "data", None) is not None
    ):
        return True
    return any(
        getattr(obj, "type", None) == "MESH"
        and getattr(obj, "data", None) is not None
        for obj in _tuple(getattr(context, "selected_objects", ()))
    )


def _redraw() -> None:
    """Request a redraw of current Blender areas after synchronous diagnostics."""

    try:
        manager = getattr(bpy.context, "window_manager", None)
        for window in _tuple(getattr(manager, "windows", ())):
            screen = getattr(window, "screen", None)
            for area in _tuple(getattr(screen, "areas", ())):
                tag_redraw = getattr(area, "tag_redraw", None)
                if callable(tag_redraw):
                    tag_redraw()
    except Exception:
        logger.debug("Unable to redraw Blender areas", exc_info=True)


def run_a1_readiness_analysis(
    context: Any,
    *,
    origin: str = "manual",
) -> A1ExportReadinessReport:
    """Run one synchronous readiness analysis and store its report.

    Input:
        ``context`` must expose the Blender Scene/selection expected by the normal
        readiness pipeline and contain at least one usable Mesh object.

    Output:
        A validated :class:`A1ExportReadinessReport` already stored in the normal
        readiness cache.

    The guard protects only accidental re-entry in the same Python call stack. It is not
    a worker/thread synchronization primitive; Blender executes this function directly on
    its main thread from the Analyze operator.
    """

    global _ANALYSIS_RUNNING

    if context is None:
        raise ValueError("context cannot be None")
    if not _has_mesh_request(context):
        raise ValueError("Select at least one Mesh object to analyze")
    if _ANALYSIS_RUNNING:
        raise RuntimeError("Spine2D readiness analysis is already running")

    resolved_origin = str(origin or "manual").strip().lower() or "manual"
    _ANALYSIS_RUNNING = True
    _redraw()
    try:
        report = _readiness.analyse_a1_export_readiness(context)
        if not isinstance(report, A1ExportReadinessReport):
            raise TypeError("readiness analysis returned an invalid report")
        _readiness.store_a1_export_readiness(context, report)
        logger.info("%s readiness analysis: %s", resolved_origin, report.state.value)
        return report
    except Exception:
        logger.exception("%s readiness analysis failed", resolved_origin)
        raise
    finally:
        _ANALYSIS_RUNNING = False
        _redraw()


def _manual_execute(self: Any, context: Any) -> set[str]:
    """Run Analyze explicitly while keeping diagnostics advisory for Export."""

    try:
        report = run_a1_readiness_analysis(context, origin="manual")
        if report.state is A1ReadinessState.BLOCKED:
            self.report(
                {"WARNING"},
                f"Analysis found {report.blocker_count} blocker(s) and "
                f"{report.warning_count} warning(s); export remains available",
            )
        elif report.state is A1ReadinessState.WARNING:
            self.report(
                {"WARNING"},
                f"Export ready with {report.warning_count} warning(s)",
            )
        else:
            self.report({"INFO"}, "Export readiness analysis passed")
        return {"FINISHED"}
    except Exception as exc:
        self.report({"ERROR"}, f"Analyze error: {exc}")
        return {"CANCELLED"}


def _never_blocks(_self: Any, _context: Any) -> bool:
    """Keep readiness reports diagnostic-only at the UI export boundary."""

    return True


def _export_execute(self: Any, context: Any, *, multiple: bool) -> set[str]:
    """Run production export directly without scheduling readiness work."""

    try:
        if _UI_MODULE is None:
            raise RuntimeError("Rewrite UI module is unavailable")
        function = (
            _UI_MODULE.export_selected_objects_a1
            if multiple
            else _UI_MODULE.export_active_object_a1
        )
        return self._report_result(function(context))
    except Exception as exc:
        label = "Multi-object" if multiple else "Single-object"
        logger.exception("Rewrite %s export failed", label.lower())
        self.report({"ERROR"}, f"{label} export failed: {exc}")
        return {"CANCELLED"}


def _single_execute(self: Any, context: Any) -> set[str]:
    return _export_execute(self, context, multiple=False)


def _multi_execute(self: Any, context: Any) -> set[str]:
    return _export_execute(self, context, multiple=True)


def _draw_nonblocking(self: Any, layout: Any, context: Any) -> bool:
    """Draw the cached manual diagnostics while always leaving Export available."""

    state, report = _readiness.current_a1_export_readiness(context)
    box = layout.box()
    row = box.row(align=True)
    row.label(text="Export readiness:")
    row.operator("object.spine2d_refresh_info", text="Analyze", icon="VIEWZOOM")

    if state is A1ReadinessState.NOT_ANALYSED:
        box.label(text="Not analyzed yet", icon="QUESTION")
        box.label(text="Run Analyze for diagnostics")
        box.label(text="Export remains available", icon="INFO")
        return True

    if state is A1ReadinessState.STALE:
        box.label(text="Analysis outdated", icon="FILE_REFRESH")
        box.label(text=_invalidation.current_a1_readiness_reason(context))
        box.label(text="Run Analyze again to refresh diagnostics")
        box.label(text="Export remains available", icon="INFO")
        return True

    if report is None:
        box.label(text="Analysis cache unavailable", icon="CANCEL")
        box.label(text="Export remains available", icon="INFO")
        return True

    box.label(
        text=(
            f"{state.value}: {report.blocker_count} blocker(s), "
            f"{report.warning_count} warning(s)"
        ),
        icon=self._state_icon(state),
    )
    for issue in report.issues[:6]:
        box.label(
            text=f"{issue.code}: {issue.message}",
            icon=self._issue_icon(issue.severity),
        )
    for item in report.objects:
        self._draw_object_readiness(box, item)
    if state is A1ReadinessState.BLOCKED:
        box.label(text="Diagnostics do not disable production export", icon="INFO")
    return True


def _patch_ui(ui_module: Any) -> None:
    """Install the advisory readiness methods and remember exact originals."""

    targets = {
        "draw": (
            ui_module.OBJECT_PT_Spine2DMeshPanel,
            "_draw_readiness",
            _draw_nonblocking,
        ),
        "manual": (
            ui_module.OBJECT_OT_Spine2DRefreshInfo,
            "execute",
            _manual_execute,
        ),
        "single": (
            ui_module.OBJECT_OT_Spine2DSingleExport,
            "execute",
            _single_execute,
        ),
        "multi": (
            ui_module.OBJECT_OT_Spine2DMultiExport,
            "execute",
            _multi_execute,
        ),
        "guard": (
            ui_module._Spine2DExportOperatorMixin,
            "_require_readiness",
            _never_blocks,
        ),
    }

    if not _BASE_METHODS:
        for name, (owner, attribute, _replacement) in targets.items():
            _BASE_METHODS[name] = getattr(owner, attribute)

    for _name, (owner, attribute, replacement) in targets.items():
        setattr(owner, attribute, replacement)


def _restore_ui(ui_module: Any) -> None:
    """Restore every method captured by :func:`_patch_ui`."""

    targets = {
        "draw": (ui_module.OBJECT_PT_Spine2DMeshPanel, "_draw_readiness"),
        "manual": (ui_module.OBJECT_OT_Spine2DRefreshInfo, "execute"),
        "single": (ui_module.OBJECT_OT_Spine2DSingleExport, "execute"),
        "multi": (ui_module.OBJECT_OT_Spine2DMultiExport, "execute"),
        "guard": (ui_module._Spine2DExportOperatorMixin, "_require_readiness"),
    }
    for name, (owner, attribute) in targets.items():
        if name in _BASE_METHODS:
            setattr(owner, attribute, _BASE_METHODS[name])
    _BASE_METHODS.clear()


def register() -> None:
    """Install the synchronous, manual readiness compatibility bridge."""

    global _REGISTERED, _UI_MODULE

    if _REGISTERED:
        return

    from . import ui

    _UI_MODULE = ui
    try:
        _patch_ui(ui)
    except Exception:
        logger.exception("Unable to register manual readiness bridge")
        _restore_ui(ui)
        _UI_MODULE = None
        raise

    _REGISTERED = True
    logger.debug("Manual readiness bridge registered")


def unregister() -> None:
    """Restore UI methods and release all process-local bridge state."""

    global _REGISTERED, _ANALYSIS_RUNNING, _UI_MODULE

    ui_module = _UI_MODULE
    _REGISTERED = False
    if ui_module is not None:
        _restore_ui(ui_module)
    else:
        _BASE_METHODS.clear()
    _UI_MODULE = None
    _ANALYSIS_RUNNING = False
    logger.debug("Manual readiness bridge unregistered")


__all__ = [
    "register",
    "run_a1_readiness_analysis",
    "unregister",
]
