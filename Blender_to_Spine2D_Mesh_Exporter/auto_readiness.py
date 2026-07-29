# pylint: disable=import-error
"""Debounced automatic Rewrite readiness with diagnostic-only export gating."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
from time import monotonic
from typing import Any, Iterator

import bpy

from .application import A1ExportReadinessReport, A1ReadinessState
from .blender_adapter import a1_export_readiness as _readiness
from .blender_adapter import a1_readiness_invalidation as _invalidation
from .blender_adapter.a1_ui_selection import _rna_identity

try:
    from bpy.app.handlers import persistent as _persistent
except Exception:  # pragma: no cover
    def _persistent(function):
        return function


logger = logging.getLogger(__name__)
_AUTO_DEBOUNCE_SECONDS = 0.5
_AUTO_POLL_SECONDS = 0.25
_BUSY_POLL_SECONDS = 0.5


@dataclass(frozen=True, slots=True)
class AutoReadinessStatus:
    mode: str
    message: str


_REGISTERED = False
_ANALYSIS_RUNNING = False
_ANALYSIS_ORIGIN: str | None = None
_EXPORT_DEPTH = 0
_FILE_LOADING = False
_PENDING = False
_PENDING_DEADLINE = 0.0
_PENDING_KEY: tuple[object, ...] | None = None
_PENDING_REASON = ""
_LAST_KEY: tuple[object, ...] | None = None
_FAILED_KEY: tuple[object, ...] | None = None
_LAST_ERROR: str | None = None
_UI_MODULE: Any | None = None
_BASE_METHODS: dict[str, Any] = {}


def _tuple(value: Any) -> tuple[Any, ...]:
    try:
        return tuple(value or ())
    except Exception:
        return ()


def _int(value: Any, default: int = 0) -> int | str:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default if value is None else str(value)


def _float(value: Any, default: float = 0.0) -> float | str:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default if value is None else str(value)


def _color(value: Any) -> tuple[float, ...] | str:
    try:
        return tuple(float(value[index]) for index in range(3))
    except Exception:
        return str(value)


def _has_mesh_request(context: Any) -> bool:
    if context is None:
        return False
    active = getattr(context, "active_object", None)
    if active is not None and getattr(active, "type", None) == "MESH":
        return getattr(active, "data", None) is not None
    return any(
        getattr(obj, "type", None) == "MESH" and getattr(obj, "data", None) is not None
        for obj in _tuple(getattr(context, "selected_objects", ()))
    )


def _object_key(obj: Any) -> tuple[object, ...]:
    data = getattr(obj, "data", None)
    bake = getattr(obj, "spine2d_bake_settings", None)
    connect = getattr(obj, "spine2d_connect_settings", None)
    return (
        _rna_identity(obj),
        str(getattr(obj, "name_full", None) or getattr(obj, "name", "")),
        None if data is None else _rna_identity(data),
        bool(getattr(connect, "enabled", False)),
        _int(getattr(bake, "bake_frame_start", 0)),
        _int(getattr(bake, "frames_for_render", 0)),
    )


def _request_key(context: Any) -> tuple[object, ...] | None:
    """Cheap poll key; depsgraph owns geometry/material/render dependency edits."""

    if not _has_mesh_request(context):
        return None
    scene = getattr(context, "scene", None)
    if scene is None:
        return None
    try:
        active = getattr(context, "active_object", None)
        render = getattr(scene, "render", None)
        camera = getattr(scene, "camera", None)
        return (
            _readiness._scene_key(scene),
            None if active is None else _rna_identity(active),
            tuple(_object_key(obj) for obj in _readiness._request_mesh_objects(context)),
            _int(getattr(scene, "frame_current", 0)),
            None if camera is None else _rna_identity(camera),
            str(getattr(render, "engine", "")),
            _int(getattr(scene, "spine2d_texture_size", 1024)),
            str(getattr(scene, "spine2d_json_path", "")),
            str(getattr(scene, "spine2d_images_path", "")),
            bool(getattr(scene, "spine2d_control_icons", False)),
            bool(getattr(scene, "spine2d_export_preview_animation", False)),
            str(getattr(scene, "spine2d_seam_maker_mode", "AUTO")),
            _float(getattr(scene, "spine2d_angle_limit", 30.0)),
            str(getattr(scene, "spine2d_angular_mode", "SEED_CONE")),
            _float(getattr(scene, "spine2d_local_angle_limit", 30.0)),
            _int(getattr(scene, "spine2d_frames_for_render", 0)),
            _int(getattr(scene, "spine2d_bake_frame_start", 0)),
            str(getattr(scene, "spine2d_material_source_policy", "REQUIRE_SOURCE")),
            str(getattr(scene, "spine2d_generated_material_pattern", "SOLID_GRAY")),
            _color(getattr(scene, "spine2d_generated_gray_color", (0.5,) * 3)),
            _float(getattr(scene, "spine2d_projection_alpha_threshold", 1.0 / 255.0)),
        )
    except Exception:
        logger.debug("Unable to build automatic readiness request key", exc_info=True)
        return None


def _redraw() -> None:
    try:
        manager = getattr(bpy.context, "window_manager", None)
        for window in _tuple(getattr(manager, "windows", ())):
            for area in _tuple(getattr(getattr(window, "screen", None), "areas", ())):
                callback = getattr(area, "tag_redraw", None)
                if callable(callback):
                    callback()
    except Exception:
        logger.debug("Unable to redraw Blender areas", exc_info=True)


def _cancel_pending() -> None:
    global _PENDING, _PENDING_DEADLINE, _PENDING_KEY, _PENDING_REASON
    _PENDING = False
    _PENDING_DEADLINE = 0.0
    _PENDING_KEY = None
    _PENDING_REASON = ""


def request_auto_analysis(
    context: Any | None = None,
    *,
    reason: str,
    immediate: bool = False,
    retry_failed: bool = False,
) -> bool:
    global _PENDING, _PENDING_DEADLINE, _PENDING_KEY, _PENDING_REASON
    global _FAILED_KEY, _LAST_ERROR

    resolved = context or getattr(bpy, "context", None)
    key = _request_key(resolved)
    if key is None:
        _cancel_pending()
        return False
    if retry_failed:
        _FAILED_KEY = None
        _LAST_ERROR = None
    _PENDING = True
    _PENDING_KEY = key
    _PENDING_REASON = str(reason or "export request changed")
    _PENDING_DEADLINE = monotonic() + (0.0 if immediate else _AUTO_DEBOUNCE_SECONDS)
    _redraw()
    return True


def run_a1_readiness_analysis(context: Any, *, origin: str) -> A1ExportReadinessReport:
    global _ANALYSIS_RUNNING, _ANALYSIS_ORIGIN, _LAST_KEY
    global _FAILED_KEY, _LAST_ERROR

    if context is None:
        raise ValueError("context cannot be None")
    if not _has_mesh_request(context):
        raise ValueError("Select at least one Mesh object to analyze")
    if _ANALYSIS_RUNNING:
        raise RuntimeError("Spine2D readiness analysis is already running")
    if _EXPORT_DEPTH > 0:
        raise RuntimeError("Spine2D export is currently running")

    _cancel_pending()
    _ANALYSIS_RUNNING = True
    _ANALYSIS_ORIGIN = str(origin or "manual").strip().lower() or "manual"
    key = _request_key(context)
    _redraw()
    try:
        report = _readiness.analyse_a1_export_readiness(context)
        if not isinstance(report, A1ExportReadinessReport):
            raise TypeError("readiness analysis returned an invalid report")
        _readiness.store_a1_export_readiness(context, report)
        _LAST_KEY = _request_key(context)
        _FAILED_KEY = None
        _LAST_ERROR = None
        logger.info("%s readiness analysis: %s", _ANALYSIS_ORIGIN, report.state.value)
        return report
    except Exception as exc:
        _FAILED_KEY = key
        _LAST_ERROR = str(exc).strip() or type(exc).__name__
        logger.exception("%s readiness analysis failed", _ANALYSIS_ORIGIN)
        raise
    finally:
        _ANALYSIS_RUNNING = False
        _ANALYSIS_ORIGIN = None
        _redraw()


def current_auto_readiness_status(context: Any | None = None) -> AutoReadinessStatus:
    if _ANALYSIS_RUNNING:
        text = "Analyzing automatically..." if _ANALYSIS_ORIGIN == "automatic" else "Analyzing..."
        return AutoReadinessStatus("RUNNING", text)
    if _PENDING:
        return AutoReadinessStatus("PENDING", "Automatic refresh pending")
    key = _request_key(context or getattr(bpy, "context", None))
    if _LAST_ERROR and key is not None and key == _FAILED_KEY:
        return AutoReadinessStatus("ERROR", f"Automatic analysis failed: {_LAST_ERROR}")
    return AutoReadinessStatus("IDLE", "")


def _cache_entry(context: Any) -> Any | None:
    try:
        scene = getattr(context, "scene", None)
        return _readiness._READINESS_CACHE.get(_readiness._scene_key(scene))
    except Exception:
        return None


def _schedule(context: Any) -> None:
    global _LAST_KEY, _FAILED_KEY, _LAST_ERROR

    key = _request_key(context)
    if key is None:
        _LAST_KEY = None
        _cancel_pending()
        return
    if key != _LAST_KEY:
        _LAST_KEY = key
        _FAILED_KEY = None
        _LAST_ERROR = None
        request_auto_analysis(context, reason="export request changed")
        return
    entry = _cache_entry(context)
    if entry is not None and not bool(getattr(entry, "stale", False)):
        _cancel_pending()
        return
    if key != _FAILED_KEY and (not _PENDING or _PENDING_KEY != key):
        request_auto_analysis(context, reason="readiness report needs refresh")


def _automatic_timer() -> float | None:
    if not _REGISTERED:
        return None
    if _FILE_LOADING or _ANALYSIS_RUNNING or _EXPORT_DEPTH > 0:
        return _BUSY_POLL_SECONDS
    context = getattr(bpy, "context", None)
    if context is None or str(getattr(context, "mode", "OBJECT")).upper() != "OBJECT":
        return _BUSY_POLL_SECONDS
    try:
        _schedule(context)
        if not _PENDING or monotonic() < _PENDING_DEADLINE:
            return _AUTO_POLL_SECONDS
        key = _request_key(context)
        if key is None:
            _cancel_pending()
        elif key != _PENDING_KEY:
            request_auto_analysis(context, reason="export request changed")
        else:
            entry = _cache_entry(context)
            if entry is not None and not bool(getattr(entry, "stale", False)):
                _cancel_pending()
            else:
                try:
                    run_a1_readiness_analysis(context, origin="automatic")
                except Exception:
                    pass
    except Exception:
        logger.exception("Automatic Spine2D readiness timer failed")
    return _AUTO_POLL_SECONDS


def _real_tracked_update(scene: Any, depsgraph: Any) -> bool:
    try:
        snapshot = _invalidation._DEPENDENCIES_BY_SCENE.get(_readiness._scene_key(scene))
        updates = tuple(getattr(depsgraph, "updates", ()))
    except Exception:
        return False
    if snapshot is None:
        return False
    for update in updates:
        updated_id = getattr(update, "id", None)
        identity = _invalidation._dependency_identity(updated_id)
        if identity is None or identity not in snapshot.identities:
            continue
        if _invalidation._is_temporary_datablock(updated_id):
            continue
        flags = _invalidation._update_flags(update)
        if identity[0] == "OBJECT" and flags and not any(flags.values()):
            continue
        if _invalidation._semantic_state_unchanged(identity, updated_id, flags, snapshot):
            continue
        return True
    return False


def _same_scene(first: Any, second: Any) -> bool:
    try:
        return _readiness._scene_key(first) == _readiness._scene_key(second)
    except Exception:
        return first is not None and first is second


@_persistent
def a1_auto_readiness_depsgraph_update_post(scene: Any, depsgraph: Any) -> None:
    if not _REGISTERED or _FILE_LOADING or _ANALYSIS_RUNNING or _EXPORT_DEPTH > 0:
        return
    context = getattr(bpy, "context", None)
    if not _same_scene(getattr(context, "scene", None), scene):
        return
    if _real_tracked_update(scene, depsgraph):
        request_auto_analysis(
            context,
            reason="tracked export dependency changed",
            retry_failed=True,
        )


@_persistent
def a1_auto_readiness_load_pre(_dummy: Any) -> None:
    global _FILE_LOADING
    _FILE_LOADING = True
    _cancel_pending()


@_persistent
def a1_auto_readiness_load_post(_dummy: Any) -> None:
    global _FILE_LOADING, _LAST_KEY, _FAILED_KEY, _LAST_ERROR
    _FILE_LOADING = False
    _LAST_KEY = None
    _FAILED_KEY = None
    _LAST_ERROR = None
    request_auto_analysis(getattr(bpy, "context", None), reason="Blend file loaded")


@contextmanager
def suspend_auto_readiness_for_export() -> Iterator[None]:
    global _EXPORT_DEPTH
    _EXPORT_DEPTH += 1
    try:
        yield
    finally:
        _EXPORT_DEPTH = max(0, _EXPORT_DEPTH - 1)


def _manual_execute(self: Any, context: Any) -> set[str]:
    try:
        report = run_a1_readiness_analysis(context, origin="manual")
        if report.state is A1ReadinessState.BLOCKED:
            self.report(
                {"WARNING"},
                f"Analysis found {report.blocker_count} blocker(s) and "
                f"{report.warning_count} warning(s); export remains available",
            )
        elif report.state is A1ReadinessState.WARNING:
            self.report({"WARNING"}, f"Export ready with {report.warning_count} warning(s)")
        else:
            self.report({"INFO"}, "Export readiness analysis passed")
        return {"FINISHED"}
    except Exception as exc:
        self.report({"ERROR"}, f"Analyze error: {exc}")
        return {"CANCELLED"}


def _never_blocks(_self: Any, _context: Any) -> bool:
    return True


def _export_execute(self: Any, context: Any, *, multiple: bool) -> set[str]:
    try:
        if _UI_MODULE is None:
            raise RuntimeError("Rewrite UI module is unavailable")
        function = (
            _UI_MODULE.export_selected_objects_a1
            if multiple
            else _UI_MODULE.export_active_object_a1
        )
        with suspend_auto_readiness_for_export():
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
    state, report = _readiness.current_a1_export_readiness(context)
    status = current_auto_readiness_status(context)
    box = layout.box()
    row = box.row(align=True)
    row.label(text="Export readiness:")
    row.operator("object.spine2d_refresh_info", text="Analyze", icon="VIEWZOOM")
    if status.mode != "IDLE":
        icon = {"RUNNING": "TIME", "PENDING": "FILE_REFRESH"}.get(status.mode, "ERROR")
        box.label(text=status.message, icon=icon)
    if state is A1ReadinessState.NOT_ANALYSED:
        box.label(text="Not analyzed yet", icon="QUESTION")
        box.label(text="Automatic analysis will run after the selection stabilizes")
        box.label(text="Export remains available", icon="INFO")
        return True
    if state is A1ReadinessState.STALE:
        box.label(text="Analysis outdated", icon="FILE_REFRESH")
        box.label(text=_invalidation.current_a1_readiness_reason(context))
        box.label(text="Export remains available while the report refreshes", icon="INFO")
        return True
    if report is None:
        box.label(text="Analysis cache unavailable", icon="CANCEL")
        box.label(text="Export remains available", icon="INFO")
        return True
    box.label(
        text=f"{state.value}: {report.blocker_count} blocker(s), {report.warning_count} warning(s)",
        icon=self._state_icon(state),
    )
    for issue in report.issues[:6]:
        box.label(text=f"{issue.code}: {issue.message}", icon=self._issue_icon(issue.severity))
    for item in report.objects:
        self._draw_object_readiness(box, item)
    if state is A1ReadinessState.BLOCKED:
        box.label(text="Diagnostics do not disable production export", icon="INFO")
    return True


def _remove_all(collection: Any, callback: Any) -> None:
    while callback in collection:
        collection.remove(callback)


def _timer_registered() -> bool:
    checker = getattr(bpy.app.timers, "is_registered", None)
    return bool(callable(checker) and checker(_automatic_timer))


def _register_timer() -> None:
    if not _timer_registered():
        bpy.app.timers.register(
            _automatic_timer,
            first_interval=_AUTO_POLL_SECONDS,
            persistent=True,
        )


def _unregister_timer() -> None:
    if _timer_registered():
        bpy.app.timers.unregister(_automatic_timer)


def _patch_ui(ui_module: Any) -> None:
    targets = {
        "draw": (ui_module.OBJECT_PT_Spine2DMeshPanel, "_draw_readiness", _draw_nonblocking),
        "manual": (ui_module.OBJECT_OT_Spine2DRefreshInfo, "execute", _manual_execute),
        "single": (ui_module.OBJECT_OT_Spine2DSingleExport, "execute", _single_execute),
        "multi": (ui_module.OBJECT_OT_Spine2DMultiExport, "execute", _multi_execute),
        "guard": (ui_module._Spine2DExportOperatorMixin, "_require_readiness", _never_blocks),
    }
    if not _BASE_METHODS:
        for name, (owner, attribute, _replacement) in targets.items():
            _BASE_METHODS[name] = getattr(owner, attribute)
    for _name, (owner, attribute, replacement) in targets.items():
        setattr(owner, attribute, replacement)


def _restore_ui(ui_module: Any) -> None:
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


def _install_handlers() -> None:
    handlers = bpy.app.handlers
    for collection, callback in (
        (handlers.depsgraph_update_post, a1_auto_readiness_depsgraph_update_post),
        (handlers.load_pre, a1_auto_readiness_load_pre),
        (handlers.load_post, a1_auto_readiness_load_post),
    ):
        _remove_all(collection, callback)
        collection.append(callback)


def _remove_handlers() -> None:
    handlers = bpy.app.handlers
    for collection, callback in (
        (handlers.depsgraph_update_post, a1_auto_readiness_depsgraph_update_post),
        (handlers.load_pre, a1_auto_readiness_load_pre),
        (handlers.load_post, a1_auto_readiness_load_post),
    ):
        _remove_all(collection, callback)


def register() -> None:
    global _REGISTERED, _UI_MODULE, _FILE_LOADING, _LAST_KEY
    if _REGISTERED:
        return
    from . import ui

    _UI_MODULE = ui
    _FILE_LOADING = False
    _LAST_KEY = None
    try:
        _patch_ui(ui)
        # Readiness is a user-triggered diagnostic. Keep the compatibility patch
        # for non-blocking export, but do not install timers or file/depsgraph
        # callbacks that would run analysis in the background.
        _REGISTERED = True
    except Exception:
        logger.exception("Unable to register automatic readiness service")
        _REGISTERED = False
        _restore_ui(ui)
        _UI_MODULE = None
        raise


def unregister() -> None:
    global _REGISTERED, _UI_MODULE, _ANALYSIS_RUNNING, _ANALYSIS_ORIGIN
    global _EXPORT_DEPTH, _FILE_LOADING, _LAST_KEY, _FAILED_KEY, _LAST_ERROR

    ui_module = _UI_MODULE
    _REGISTERED = False
    if ui_module is not None:
        _restore_ui(ui_module)
    _cancel_pending()
    _UI_MODULE = None
    _ANALYSIS_RUNNING = False
    _ANALYSIS_ORIGIN = None
    _EXPORT_DEPTH = 0
    _FILE_LOADING = False
    _LAST_KEY = None
    _FAILED_KEY = None
    _LAST_ERROR = None


__all__ = [
    "AutoReadinessStatus",
    "a1_auto_readiness_depsgraph_update_post",
    "a1_auto_readiness_load_post",
    "a1_auto_readiness_load_pre",
    "current_auto_readiness_status",
    "register",
    "request_auto_analysis",
    "run_a1_readiness_analysis",
    "suspend_auto_readiness_for_export",
    "unregister",
]
