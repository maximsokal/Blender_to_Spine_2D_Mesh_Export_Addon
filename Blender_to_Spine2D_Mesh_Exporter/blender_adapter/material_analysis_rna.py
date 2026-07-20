"""Blender-compatible RNA reads used by material analysis."""

from __future__ import annotations

import logging
from typing import Any

from .material_analysis_error import MaterialAnalysisError
from .shader_graph_rna import (
    is_temporary_node,
    normalise_render_target,
)


logger = logging.getLogger(__name__)


def material_name(material: Any) -> str:
    """Return a stable non-empty Blender material name."""

    value = str(
        getattr(material, "name_full", None)
        or getattr(material, "name", None)
        or ""
    )
    if not value:
        raise MaterialAnalysisError("Material name is empty")
    return value


def object_name(obj: Any) -> str:
    """Return a stable non-empty Blender object name."""

    value = str(
        getattr(obj, "name_full", None)
        or getattr(obj, "name", None)
        or ""
    )
    if not value:
        raise MaterialAnalysisError("object name is empty")
    return value


def node_type(node: Any) -> str:
    """Return Blender's node type identifier or the historical UNKNOWN fallback."""

    value = str(getattr(node, "type", "") or "")
    return value or "UNKNOWN"


def material_root_nodes(material: Any, *, resolved_name: str) -> tuple[Any, ...]:
    """Freeze one material's root node collection with the historical error contract."""

    try:
        return tuple(material.node_tree.nodes)
    except Exception as exc:
        raise MaterialAnalysisError(
            f"Unable to iterate nodes of material '{resolved_name}'"
        ) from exc


def object_material_slots(obj: Any) -> tuple[Any, ...]:
    """Freeze material slots once so dense slot order cannot change mid-analysis."""

    return tuple(getattr(obj, "material_slots", ()))


def render_target_from_engine(render_engine: str | None) -> str:
    """Translate Blender render-engine identifiers to ShaderNodeTree targets."""

    return normalise_render_target(render_engine)


def resolve_render_target(render_target: str | None) -> str:
    """Resolve an explicit target or the active Blender Scene renderer."""

    if render_target is not None:
        return normalise_render_target(render_target)
    try:
        import bpy  # pylint: disable=import-error,import-outside-toplevel

        scene = getattr(getattr(bpy, "context", None), "scene", None)
        engine = getattr(getattr(scene, "render", None), "engine", None)
        return render_target_from_engine(engine)
    except Exception:
        logger.debug(
            "Unable to resolve active Blender render target; using ALL",
            exc_info=True,
        )
        return "ALL"


__all__ = [
    "is_temporary_node",
    "material_name",
    "material_root_nodes",
    "node_type",
    "normalise_render_target",
    "object_material_slots",
    "object_name",
    "render_target_from_engine",
    "resolve_render_target",
]
