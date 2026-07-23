"""Grouped Blender 5.2 camera visibility inside one reversible render scope."""

from __future__ import annotations

from typing import Any, Tuple

from .camera_projection_error import CameraProjectionExecutionError
from .grouped_camera_projection_validation import object_name, rna_identity


_RENDERABLE_TYPES = frozenset(
    {"MESH", "CURVE", "SURFACE", "META", "FONT", "VOLUME"}
)


def configure_group_camera_visibility(
    source_objects: Tuple[Any, ...],
    scene: Any,
) -> None:
    """Expose grouped sources and hide direct camera rays of other renderables."""

    if (
        not isinstance(source_objects, tuple)
        or len(source_objects) < 2
        or any(item is None for item in source_objects)
    ):
        raise ValueError(
            "source_objects must contain at least two Blender objects"
        )
    if scene is None:
        raise CameraProjectionExecutionError("scene cannot be None")

    try:
        scene_objects = tuple(scene.objects)
    except Exception as exc:
        raise CameraProjectionExecutionError(
            "Unable to inspect scene objects for grouped projection visibility"
        ) from exc

    source_identities = {rna_identity(obj) for obj in source_objects}
    if len(source_identities) != len(source_objects):
        raise CameraProjectionExecutionError(
            "grouped projection source_objects contain duplicate Blender objects"
        )

    scene_identities = {rna_identity(obj) for obj in scene_objects}
    missing = tuple(
        obj
        for obj in source_objects
        if rna_identity(obj) not in scene_identities
    )
    if missing:
        raise CameraProjectionExecutionError(
            "grouped projection source objects are not linked to the render scene: "
            + str(tuple(object_name(obj) for obj in missing))
        )

    for obj in scene_objects:
        identity = rna_identity(obj)
        if identity in source_identities:
            try:
                obj.hide_render = False
                obj.visible_camera = True
            except Exception as exc:
                raise CameraProjectionExecutionError(
                    "Unable to make grouped source "
                    f"'{object_name(obj)}' camera-visible"
                ) from exc
            continue

        if str(getattr(obj, "type", "") or "") in _RENDERABLE_TYPES:
            try:
                obj.visible_camera = False
            except Exception as exc:
                raise CameraProjectionExecutionError(
                    "Unable to isolate grouped projection camera layer from "
                    f"'{object_name(obj)}'"
                ) from exc


__all__ = [
    "CameraProjectionExecutionError",
    "configure_group_camera_visibility",
]
