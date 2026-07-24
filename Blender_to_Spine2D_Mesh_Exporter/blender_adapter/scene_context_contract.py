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


def _depsgraph_original_scene(depsgraph: Any) -> Any:
    """Resolve the original Scene that owns a Blender 5.2 dependency graph."""

    scene = getattr(depsgraph, "scene", None)
    if scene is not None:
        return scene

    evaluated_scene = getattr(depsgraph, "scene_eval", None)
    if evaluated_scene is None:
        raise BlenderSceneContextError(
            "Blender dependency graph exposes neither scene nor scene_eval"
        )
    original = getattr(evaluated_scene, "original", None)
    if original is None:
        raise BlenderSceneContextError(
            "Blender dependency graph scene_eval has no original Scene owner"
        )
    return original


def require_depsgraph_scene_consistency(
    depsgraph: Any | None,
    scene: Any | None,
) -> None:
    """Require a supplied dependency graph to belong to the explicit Scene."""

    if depsgraph is None or scene is None:
        return
    depsgraph_scene = _depsgraph_original_scene(depsgraph)
    if rna_identity(depsgraph_scene) != rna_identity(scene):
        depsgraph_scene_name = str(
            getattr(depsgraph_scene, "name", "") or "<unnamed>"
        )
        scene_name = str(getattr(scene, "name", "") or "<unnamed>")
        raise BlenderSceneContextError(
            "Blender dependency graph and explicit scene must reference the same "
            f"Scene; depsgraph_scene={depsgraph_scene_name!r}, scene={scene_name!r}"
        )


__all__ = [
    "BlenderSceneContextError",
    "require_context_scene_consistency",
    "require_depsgraph_scene_consistency",
    "rna_identity",
]
