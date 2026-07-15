# pylint: disable=import-error
"""Safe Blender context and BMesh resource helpers.

The exporter performs many stateful Blender operations.  This module keeps
ownership rules explicit:

* every BMesh created with :func:`bmesh.new` is freed exactly once;
* edit BMeshes returned by ``bmesh.from_edit_mesh`` are never handled here;
* active object, selection and object mode can be restored after a pipeline;
* temporary objects are removed through the data API, not selection operators.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
from typing import Iterator

import bpy
import bmesh

logger = logging.getLogger(__name__)


def _object_is_alive(obj: object | None) -> bool:
    """Return ``True`` when *obj* still refers to a linked Blender object."""
    if obj is None:
        return False
    try:
        name = obj.name
        stored = bpy.data.objects.get(name)
        return stored is obj or stored == obj
    except (AttributeError, ReferenceError, RuntimeError):
        return False


def _safe_mode_set(mode: str) -> bool:
    """Set object mode only when Blender's operator context allows it."""
    try:
        poll = getattr(bpy.ops.object.mode_set, "poll", None)
        if callable(poll) and not poll():
            return False
        bpy.ops.object.mode_set(mode=mode)
        return True
    except (AttributeError, RuntimeError, TypeError):
        logger.debug("Unable to switch Blender mode to %s", mode, exc_info=True)
        return False


@dataclass(slots=True)
class BlenderContextSnapshot:
    """Minimal scene state required to restore the user's working context."""

    active_object: object | None
    selected_objects: tuple[object, ...]
    active_mode: str | None

    @classmethod
    def capture(cls) -> "BlenderContextSnapshot":
        try:
            active = bpy.context.view_layer.objects.active
        except (AttributeError, RuntimeError):
            active = getattr(bpy.context, "active_object", None)

        try:
            selected = tuple(bpy.context.selected_objects)
        except (AttributeError, RuntimeError, TypeError):
            selected = ()

        try:
            mode = active.mode if active is not None else None
        except (AttributeError, ReferenceError, RuntimeError):
            mode = None

        return cls(active_object=active, selected_objects=selected, active_mode=mode)

    def restore(self) -> None:
        """Best-effort restoration that never masks the export exception."""
        try:
            current_active = getattr(bpy.context.view_layer.objects, "active", None)
            current_mode = getattr(current_active, "mode", "OBJECT")
            if current_active is not None and current_mode != "OBJECT":
                _safe_mode_set("OBJECT")

            try:
                currently_selected = tuple(bpy.context.selected_objects)
            except (AttributeError, RuntimeError, TypeError):
                currently_selected = ()

            for obj in currently_selected:
                if _object_is_alive(obj):
                    try:
                        obj.select_set(False)
                    except (AttributeError, ReferenceError, RuntimeError):
                        logger.debug("Failed to deselect %r", obj, exc_info=True)

            for obj in self.selected_objects:
                if _object_is_alive(obj):
                    try:
                        obj.select_set(True)
                    except (AttributeError, ReferenceError, RuntimeError):
                        logger.debug("Failed to reselect %r", obj, exc_info=True)

            if _object_is_alive(self.active_object):
                bpy.context.view_layer.objects.active = self.active_object
                if self.active_mode and self.active_mode != "OBJECT":
                    _safe_mode_set(self.active_mode)
            else:
                bpy.context.view_layer.objects.active = None
        except Exception:  # Context restoration must never hide primary failures.
            logger.exception("Failed to restore Blender context")


def activate_object(obj: object, *, ensure_object_mode: bool = True) -> None:
    """Make a linked object active and exclusively selected."""
    if not _object_is_alive(obj):
        raise ReferenceError("Cannot activate an object that is not linked to bpy.data")

    current_active = getattr(bpy.context.view_layer.objects, "active", None)
    current_mode = getattr(current_active, "mode", "OBJECT")
    if ensure_object_mode and current_active is not None and current_mode != "OBJECT":
        if not _safe_mode_set("OBJECT"):
            raise RuntimeError("Blender could not switch to OBJECT mode")

    try:
        selected = tuple(bpy.context.selected_objects)
    except (AttributeError, RuntimeError, TypeError):
        selected = ()

    for selected_obj in selected:
        if _object_is_alive(selected_obj):
            selected_obj.select_set(False)

    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


@contextmanager
def managed_bmesh(mesh: object, *, write_back: bool = False) -> Iterator[object]:
    """Yield a new BMesh and free it exactly once.

    This helper is only for BMeshes created through ``bmesh.new()``.  It must
    never be used with ``bmesh.from_edit_mesh()``.
    """
    if mesh is None:
        raise TypeError("mesh must not be None")

    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        yield bm
        if write_back:
            bm.to_mesh(mesh)
            update = getattr(mesh, "update", None)
            if callable(update):
                update()
    finally:
        bm.free()


def triangulate_mesh_data(mesh: object) -> None:
    """Triangulate a Mesh datablock and persist the result safely."""
    with managed_bmesh(mesh, write_back=True) as bm:
        faces = list(bm.faces)
        if not faces:
            return
        bmesh.ops.triangulate(
            bm,
            faces=faces,
            quad_method="BEAUTY",
            ngon_method="BEAUTY",
        )


def remove_object_if_alive(obj: object | None) -> bool:
    """Remove a linked object and return whether removal was performed."""
    if not _object_is_alive(obj):
        return False
    try:
        bpy.data.objects.remove(obj, do_unlink=True)
        return True
    except (ReferenceError, RuntimeError):
        logger.warning("Failed to remove temporary object %r", obj, exc_info=True)
        return False


def scene_bool(name: str, default: bool = False) -> bool:
    """Read a registered property or custom scene property without assumptions."""
    scene = getattr(bpy.context, "scene", None)
    if scene is None:
        return default

    try:
        value = getattr(scene, name)
    except (AttributeError, RuntimeError):
        try:
            value = scene.get(name, default)
        except (AttributeError, RuntimeError, TypeError):
            return default
    return bool(value)
