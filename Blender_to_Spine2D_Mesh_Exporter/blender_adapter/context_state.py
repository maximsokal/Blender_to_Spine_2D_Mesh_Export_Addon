"""Capture, switch, and restore Blender operator context safely."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
from typing import Any, Iterator

logger = logging.getLogger(__name__)


class BlenderContextError(RuntimeError):
    """Raised when a required Blender context transition cannot be completed."""


def _load_bpy() -> Any:
    try:
        import bpy
    except Exception as exc:
        raise BlenderContextError("Blender bpy module is unavailable") from exc
    return bpy


def _object_is_alive(bpy_module: Any, obj: Any | None) -> bool:
    if obj is None:
        return False
    try:
        stored = bpy_module.data.objects.get(obj.name)
        return stored is obj or stored == obj
    except (AttributeError, ReferenceError, RuntimeError):
        return False


def _set_mode(bpy_module: Any, mode: str) -> None:
    if not isinstance(mode, str) or not mode:
        raise ValueError("mode must be a non-empty string")
    operator = bpy_module.ops.object.mode_set
    poll = getattr(operator, "poll", None)
    if callable(poll) and not poll():
        raise BlenderContextError(f"bpy.ops.object.mode_set cannot enter {mode}")
    try:
        operator(mode=mode)
    except Exception as exc:
        raise BlenderContextError(f"Unable to switch Blender mode to {mode}") from exc


@dataclass(frozen=True, slots=True)
class BlenderContextState:
    active_object: Any | None
    selected_objects: tuple[Any, ...]
    active_mode: str | None

    @classmethod
    def capture(cls, context: Any | None = None) -> "BlenderContextState":
        bpy_module = _load_bpy()
        resolved_context = context or bpy_module.context
        try:
            active = resolved_context.view_layer.objects.active
        except Exception:
            active = getattr(resolved_context, "active_object", None)
        try:
            selected = tuple(resolved_context.selected_objects)
        except Exception:
            selected = ()
        try:
            active_mode = active.mode if active is not None else None
        except Exception:
            active_mode = None
        return cls(
            active_object=active,
            selected_objects=selected,
            active_mode=active_mode,
        )

    def restore(self, context: Any | None = None) -> None:
        """Restore context without hiding the primary export exception."""

        bpy_module = _load_bpy()
        resolved_context = context or bpy_module.context
        try:
            current_active = resolved_context.view_layer.objects.active
            current_mode = (
                getattr(current_active, "mode", "OBJECT")
                if current_active is not None
                else "OBJECT"
            )
            if current_active is not None and current_mode != "OBJECT":
                try:
                    _set_mode(bpy_module, "OBJECT")
                except BlenderContextError:
                    logger.warning("Unable to leave current mode during context restore")

            try:
                currently_selected = tuple(resolved_context.selected_objects)
            except Exception:
                currently_selected = ()
            for selected_object in currently_selected:
                if _object_is_alive(bpy_module, selected_object):
                    try:
                        selected_object.select_set(False)
                    except Exception:
                        logger.debug("Unable to deselect object during restore", exc_info=True)

            for selected_object in self.selected_objects:
                if _object_is_alive(bpy_module, selected_object):
                    try:
                        selected_object.select_set(True)
                    except Exception:
                        logger.debug("Unable to restore object selection", exc_info=True)

            if _object_is_alive(bpy_module, self.active_object):
                resolved_context.view_layer.objects.active = self.active_object
                if self.active_mode and self.active_mode != "OBJECT":
                    try:
                        _set_mode(bpy_module, self.active_mode)
                    except BlenderContextError:
                        logger.warning(
                            "Unable to restore original Blender mode '%s'",
                            self.active_mode,
                        )
            else:
                resolved_context.view_layer.objects.active = None
        except Exception:
            logger.exception("Failed to restore Blender context")


@contextmanager
def activate_object_for_operator(
    obj: Any,
    *,
    context: Any | None = None,
) -> Iterator[BlenderContextState]:
    """Make one linked object exclusively active and restore prior state."""

    bpy_module = _load_bpy()
    resolved_context = context or bpy_module.context
    if not _object_is_alive(bpy_module, obj):
        raise BlenderContextError("Cannot activate an unlinked Blender object")

    state = BlenderContextState.capture(resolved_context)
    try:
        current_active = resolved_context.view_layer.objects.active
        if current_active is not None and getattr(current_active, "mode", "OBJECT") != "OBJECT":
            _set_mode(bpy_module, "OBJECT")

        try:
            selected_objects = tuple(resolved_context.selected_objects)
        except Exception:
            selected_objects = ()
        for selected_object in selected_objects:
            if _object_is_alive(bpy_module, selected_object):
                selected_object.select_set(False)

        obj.select_set(True)
        resolved_context.view_layer.objects.active = obj
        yield state
    finally:
        state.restore(resolved_context)
