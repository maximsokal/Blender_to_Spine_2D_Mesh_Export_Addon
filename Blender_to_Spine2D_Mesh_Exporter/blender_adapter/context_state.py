"""Capture, switch, and restore Blender 5.2 operator context safely."""

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


def _require_operator_finished(result: Any, *, label: str) -> None:
    try:
        finished = "FINISHED" in result
    except Exception as exc:
        raise BlenderContextError(
            f"{label} returned an invalid operator result: {result!r}"
        ) from exc
    if not finished:
        raise BlenderContextError(f"{label} did not finish: {result!r}")


def _set_mode(bpy_module: Any, mode: str) -> None:
    if not isinstance(mode, str) or not mode.strip():
        raise ValueError("mode must be a non-empty string")
    resolved_mode = mode.strip().upper()
    operator = bpy_module.ops.object.mode_set
    poll = getattr(operator, "poll", None)
    if callable(poll) and not poll():
        raise BlenderContextError(
            f"bpy.ops.object.mode_set cannot enter {resolved_mode}"
        )
    try:
        result = operator(mode=resolved_mode)
    except Exception as exc:
        raise BlenderContextError(
            f"Unable to switch Blender mode to {resolved_mode}"
        ) from exc
    _require_operator_finished(
        result,
        label=f"bpy.ops.object.mode_set(mode={resolved_mode!r})",
    )


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
            active_mode = (
                str(active.mode).upper() if active is not None else None
            )
        except Exception:
            active_mode = None
        return cls(
            active_object=active,
            selected_objects=selected,
            active_mode=active_mode,
        )

    def restore(self, context: Any | None = None) -> None:
        """Restore context best-effort without hiding a primary export exception."""

        bpy_module = _load_bpy()
        resolved_context = context or bpy_module.context
        failures: list[str] = []

        try:
            current_active = resolved_context.view_layer.objects.active
        except Exception as exc:
            current_active = None
            failures.append(f"active object inspection: {exc}")

        current_mode = "OBJECT"
        if current_active is not None:
            try:
                current_mode = str(current_active.mode).upper()
            except Exception as exc:
                failures.append(f"current mode inspection: {exc}")
        if current_active is not None and current_mode != "OBJECT":
            try:
                _set_mode(bpy_module, "OBJECT")
            except BlenderContextError as exc:
                failures.append(str(exc))

        try:
            currently_selected = tuple(resolved_context.selected_objects)
        except Exception as exc:
            currently_selected = ()
            failures.append(f"selection inspection: {exc}")
        for selected_object in currently_selected:
            if _object_is_alive(bpy_module, selected_object):
                try:
                    selected_object.select_set(False)
                except Exception as exc:
                    failures.append(
                        f"deselect {getattr(selected_object, 'name', selected_object)!r}: {exc}"
                    )

        for selected_object in self.selected_objects:
            if _object_is_alive(bpy_module, selected_object):
                try:
                    selected_object.select_set(True)
                except Exception as exc:
                    failures.append(
                        f"select {getattr(selected_object, 'name', selected_object)!r}: {exc}"
                    )

        try:
            if _object_is_alive(bpy_module, self.active_object):
                resolved_context.view_layer.objects.active = self.active_object
                if self.active_mode and self.active_mode != "OBJECT":
                    _set_mode(bpy_module, self.active_mode)
            else:
                resolved_context.view_layer.objects.active = None
        except Exception as exc:
            failures.append(f"active object/mode restore: {exc}")

        if failures:
            raise BlenderContextError(
                "Unable to restore Blender context completely: " + "; ".join(failures)
            )


@contextmanager
def activate_object_for_operator(
    obj: Any,
    *,
    context: Any | None = None,
) -> Iterator[BlenderContextState]:
    """Make one linked Object exclusively active and restore prior state."""

    bpy_module = _load_bpy()
    resolved_context = context or bpy_module.context
    if not _object_is_alive(bpy_module, obj):
        raise BlenderContextError("Cannot activate an unlinked Blender object")

    state = BlenderContextState.capture(resolved_context)
    primary_error: BaseException | None = None
    try:
        current_active = resolved_context.view_layer.objects.active
        if (
            current_active is not None
            and str(getattr(current_active, "mode", "OBJECT")).upper() != "OBJECT"
        ):
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
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            state.restore(resolved_context)
        except Exception:
            if primary_error is None:
                raise
            logger.exception(
                "Failed to restore Blender context while handling another exception"
            )


__all__ = [
    "BlenderContextError",
    "BlenderContextState",
    "activate_object_for_operator",
]
