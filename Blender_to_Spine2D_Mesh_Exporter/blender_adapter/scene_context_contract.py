"""Blender RNA identity checks for scene-bound Rewrite operations."""

from __future__ import annotations

from typing import Any


class BlenderSceneContextError(ValueError):
    """Raised when independently supplied Blender context owners disagree."""


def rna_identity(value: Any) -> int:
    """Return a stable identity for Blender RNA values and ordinary test doubles."""

    if value is None:
        raise ValueError("value cannot be None")
    pointer = getattr(value, "as_pointer", None)
    if callable(pointer):
        try:
            resolved = int(pointer())
        except Exception as exc:
            raise BlenderSceneContextError(
                f"Unable to resolve RNA identity for {value!r}"
            ) from exc
        if resolved <= 0:
            raise BlenderSceneContextError(
                f"RNA value {value!r} returned an invalid pointer {resolved}"
            )
        return resolved
    return id(value)


def require_context_scene_consistency(
    context: Any | None,
    scene: Any | None,
) -> None:
    """Require ``context.scene`` and an explicit scene to reference one Scene."""

    if context is None or scene is None:
        return
    context_scene = getattr(context, "scene", None)
    if context_scene is None:
        raise BlenderSceneContextError(
            "Explicit Blender context has no scene while a scene argument was supplied"
        )
    if rna_identity(context_scene) != rna_identity(scene):
        context_name = str(getattr(context_scene, "name", "") or "<unnamed>")
        scene_name = str(getattr(scene, "name", "") or "<unnamed>")
        raise BlenderSceneContextError(
            "Blender context.scene and explicit scene must reference the same Scene; "
            f"context_scene={context_name!r}, scene={scene_name!r}"
        )


__all__ = [
    "BlenderSceneContextError",
    "require_context_scene_consistency",
    "rna_identity",
]
