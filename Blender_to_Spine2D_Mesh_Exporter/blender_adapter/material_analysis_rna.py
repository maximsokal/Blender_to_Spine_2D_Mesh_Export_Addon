"""Strict Blender 5.2 RNA reads used by material analysis."""

from __future__ import annotations

from typing import Any

from .material_analysis_error import MaterialAnalysisError
from .shader_graph_error import MaterialGraphAnalysisError
from .shader_graph_rna import (
    is_temporary_node,
    normalise_render_target,
)


def material_name(material: Any) -> str:
    """Return a stable non-empty Blender material name."""

    value = str(
        getattr(material, "name_full", None)
        or getattr(material, "name", None)
        or ""
    ).strip()
    if not value:
        raise MaterialAnalysisError("Material name is empty")
    return value


def object_name(obj: Any) -> str:
    """Return a stable non-empty Blender object name."""

    value = str(
        getattr(obj, "name_full", None)
        or getattr(obj, "name", None)
        or ""
    ).strip()
    if not value:
        raise MaterialAnalysisError("object name is empty")
    return value


def node_type(node: Any) -> str:
    """Return Blender's node type identifier or UNKNOWN for malformed nodes."""

    value = str(getattr(node, "type", "") or "").strip()
    return value or "UNKNOWN"


def material_root_nodes(material: Any, *, resolved_name: str) -> tuple[Any, ...]:
    """Freeze one Blender 5.2 material root node collection exactly once."""

    if not isinstance(resolved_name, str) or not resolved_name.strip():
        raise ValueError("resolved_name must be a non-empty string")
    node_tree = getattr(material, "node_tree", None)
    if node_tree is None:
        raise MaterialAnalysisError(
            f"Material '{resolved_name}' has no node tree"
        )
    try:
        return tuple(node_tree.nodes)
    except Exception as exc:
        raise MaterialAnalysisError(
            f"Unable to iterate nodes of material '{resolved_name}'"
        ) from exc


def object_material_slots(obj: Any) -> tuple[Any, ...]:
    """Freeze material slots once so dense slot order cannot change mid-analysis."""

    try:
        return tuple(obj.material_slots)
    except Exception as exc:
        raise MaterialAnalysisError(
            f"Unable to iterate material slots of object '{object_name(obj)}'"
        ) from exc


def require_render_target(render_target: str) -> str:
    """Require one explicit ShaderNodeTree target from the renderer contract.

    Material analysis no longer reads the mutable active Scene and never falls
    back to ``ALL`` when renderer resolution fails. The caller must pass the
    immutable target selected during source preparation.
    """

    if not isinstance(render_target, str) or not render_target.strip():
        raise MaterialAnalysisError(
            "render_target must be an explicit non-empty Blender 5.2 shader target"
        )
    try:
        return normalise_render_target(render_target)
    except MaterialGraphAnalysisError as exc:
        raise MaterialAnalysisError(
            f"Invalid Blender 5.2 material render target {render_target!r}: {exc}"
        ) from exc


__all__ = [
    "is_temporary_node",
    "material_name",
    "material_root_nodes",
    "node_type",
    "normalise_render_target",
    "object_material_slots",
    "object_name",
    "require_render_target",
]
