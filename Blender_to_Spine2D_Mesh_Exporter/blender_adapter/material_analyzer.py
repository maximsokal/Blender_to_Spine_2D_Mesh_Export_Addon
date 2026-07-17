"""Read Blender material slots into immutable semantic baking analysis.

Legacy material kinds remain available for compatibility, while every node material
also receives a graph snapshot rooted at the renderer-effective Material Output.
Strategy selection therefore depends on connected shader semantics instead of all
nodes that merely happen to exist in the editor.
"""

from __future__ import annotations

import logging
from typing import Any

from ..domain.baking import (
    ImageDependency,
    MaterialAnalysis,
    MaterialKind,
    ObjectMaterialAnalysis,
)
from .shader_graph_analyzer import (
    MaterialGraphAnalysisError,
    analyse_material_graph_detailed,
)

logger = logging.getLogger(__name__)


class MaterialAnalysisError(RuntimeError):
    """Raised when Blender material data cannot be inspected safely."""


_PROCEDURAL_NODE_TYPES = frozenset(
    {
        "TEX_BRICK",
        "TEX_CHECKER",
        "TEX_GABOR",
        "TEX_GRADIENT",
        "TEX_MAGIC",
        "TEX_MUSGRAVE",
        "TEX_NOISE",
        "TEX_SKY",
        "TEX_VORONOI",
        "TEX_WAVE",
        "TEX_WHITE_NOISE",
        "SCRIPT",
    }
)


def _material_name(material: Any) -> str:
    name = str(
        getattr(material, "name_full", None)
        or getattr(material, "name", None)
        or ""
    )
    if not name:
        raise MaterialAnalysisError("Material name is empty")
    return name


def _node_type(node: Any) -> str:
    value = str(getattr(node, "type", "") or "")
    return value or "UNKNOWN"


def _is_temporary_bake_node(node: Any) -> bool:
    name = str(getattr(node, "name", "") or "")
    return name.startswith(
        (
            "TEMP_BAKE_",
            "TEMP_UV_",
            "__Spine2D_BakeTarget_",
            "__Spine2D_Proxy_",
        )
    )


def _normalise_render_target(value: str | None) -> str:
    target = str(value or "ALL").strip().upper()
    if target in {"ALL", "CYCLES", "EEVEE"}:
        return target
    if "CYCLE" in target:
        return "CYCLES"
    if "EEVEE" in target:
        return "EEVEE"
    return "ALL"


def render_target_from_engine(render_engine: str | None) -> str:
    """Translate Blender render-engine identifiers to ShaderNodeTree targets."""

    return _normalise_render_target(render_engine)


def _resolve_render_target(render_target: str | None) -> str:
    """Resolve an explicit target or the active Blender scene renderer.

    Adapter callers may pass a target directly for alternate-scene/headless workflows.
    The context lookup preserves compatibility with existing production call sites that
    analyze the active export scene without passing a renderer explicitly.
    """

    if render_target is not None:
        return _normalise_render_target(render_target)
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


def _image_dependency(node: Any) -> tuple[ImageDependency | None, str | None]:
    image = getattr(node, "image", None)
    node_name = str(getattr(node, "name", "") or "TEX_IMAGE")
    if image is None:
        return None, f"Image Texture node '{node_name}' has no image"

    image_name = str(
        getattr(image, "name_full", None)
        or getattr(image, "name", None)
        or ""
    )
    if not image_name:
        return None, f"Image Texture node '{node_name}' references an unnamed image"
    source = str(getattr(image, "source", "FILE") or "FILE")
    filepath_value = (
        getattr(image, "filepath_raw", None)
        or getattr(image, "filepath", None)
    )
    filepath = None if filepath_value in (None, "") else str(filepath_value)
    frame_duration = int(getattr(image, "frame_duration", 1) or 1)
    return (
        ImageDependency(
            image_name=image_name,
            source=source,
            filepath=filepath,
            frame_duration=max(1, frame_duration),
            generated=source.upper() == "GENERATED",
        ),
        None,
    )


def _classify_nodes(
    nodes: tuple[Any, ...],
) -> tuple[
    MaterialKind,
    tuple[str, ...],
    tuple[ImageDependency, ...],
    tuple[str, ...],
]:
    """Classify one already-resolved reachable node set deterministically."""

    relevant_nodes = tuple(
        node
        for node in nodes
        if not _is_temporary_bake_node(node)
        and not bool(getattr(node, "mute", False))
    )
    node_types = tuple(sorted({_node_type(node) for node in relevant_nodes}))
    procedural = any(
        _node_type(node) in _PROCEDURAL_NODE_TYPES for node in relevant_nodes
    )

    dependencies_by_key: dict[
        tuple[str, str, str | None, int], ImageDependency
    ] = {}
    issues: list[str] = []
    invalid_image_count = 0
    for node in relevant_nodes:
        if _node_type(node) != "TEX_IMAGE":
            continue
        dependency, issue = _image_dependency(node)
        if issue is not None:
            invalid_image_count += 1
            issues.append(issue)
            continue
        assert dependency is not None
        key = (
            dependency.image_name,
            dependency.source,
            dependency.filepath,
            dependency.frame_duration,
        )
        dependencies_by_key[key] = dependency

    dependencies = tuple(
        dependency
        for _, dependency in sorted(
            dependencies_by_key.items(),
            key=lambda item: item[0],
        )
    )

    if invalid_image_count:
        kind = MaterialKind.UNSUPPORTED
    elif dependencies and procedural:
        kind = MaterialKind.MIXED
    elif dependencies:
        kind = MaterialKind.IMAGE
    elif procedural:
        kind = MaterialKind.PROCEDURAL
    else:
        kind = MaterialKind.SOLID_COLOR

    return kind, node_types, dependencies, tuple(issues)


def analyse_material_slot(
    slot_index: int,
    material: Any | None,
    *,
    render_target: str | None = None,
) -> MaterialAnalysis:
    """Analyze one Blender material slot without modifying its material."""

    if not isinstance(slot_index, int) or slot_index < 0:
        raise ValueError("slot_index must be a non-negative integer")
    if material is None:
        return MaterialAnalysis(
            slot_index=slot_index,
            material_name=None,
            kind=MaterialKind.EMPTY,
            issues=("Material slot is empty",),
        )

    material_name = _material_name(material)
    node_tree = getattr(material, "node_tree", None)
    use_nodes = bool(getattr(material, "use_nodes", node_tree is not None))
    if not use_nodes or node_tree is None:
        return MaterialAnalysis(
            slot_index=slot_index,
            material_name=material_name,
            kind=MaterialKind.SOLID_COLOR,
            issues=("Material has no node tree; diffuse_color fallback is required",),
        )

    try:
        root_nodes = tuple(node_tree.nodes)
    except Exception as exc:
        raise MaterialAnalysisError(
            f"Unable to iterate nodes of material '{material_name}'"
        ) from exc

    target = _resolve_render_target(render_target)
    graph = None
    graph_nodes: tuple[Any, ...] | None = None
    graph_issues: list[str] = []
    try:
        detailed = analyse_material_graph_detailed(
            material,
            render_target=target,
        )
        graph = detailed.snapshot
        graph_issues.extend(graph.issues)
        if graph.active_output_node_id is None and not graph.semantic_channels:
            # Preserve historical no-output behavior for synthetic or damaged materials.
            # The diagnostic graph is still useful in issues, but classification must use
            # the old root-node fallback instead of treating orphaned group content as an
            # active material program.
            graph = None
        else:
            graph_nodes = detailed.reachable_nodes
    except MaterialGraphAnalysisError as exc:
        graph_issues.append(f"Shader graph analysis failed: {exc}")
        logger.warning(
            "Shader graph analysis failed for material '%s'",
            material_name,
            exc_info=True,
        )

    classification_nodes = (
        graph_nodes
        if graph_nodes is not None
        else tuple(
            node for node in root_nodes if not _is_temporary_bake_node(node)
        )
    )
    kind, node_types, dependencies, classification_issues = _classify_nodes(
        classification_nodes
    )
    issues = tuple(
        dict.fromkeys(tuple(classification_issues) + tuple(graph_issues))
    )

    return MaterialAnalysis(
        slot_index=slot_index,
        material_name=material_name,
        kind=kind,
        node_types=node_types,
        image_dependencies=dependencies,
        issues=issues,
        graph=graph,
    )


def analyse_object_materials(
    obj: Any,
    *,
    source_object_id: str | None = None,
    render_target: str | None = None,
) -> ObjectMaterialAnalysis:
    """Analyze all material slots of one Blender mesh object in stable order."""

    if obj is None:
        raise MaterialAnalysisError("obj cannot be None")
    if getattr(obj, "type", None) != "MESH":
        raise MaterialAnalysisError("obj must be a Blender MESH object")
    object_name = str(
        getattr(obj, "name_full", None)
        or getattr(obj, "name", None)
        or ""
    )
    if not object_name:
        raise MaterialAnalysisError("object name is empty")
    resolved_source_object_id = source_object_id or object_name
    target = _resolve_render_target(render_target)

    try:
        material_slots = tuple(getattr(obj, "material_slots", ()))
        analyses = tuple(
            analyse_material_slot(
                slot_index,
                getattr(slot, "material", None),
                render_target=target,
            )
            for slot_index, slot in enumerate(material_slots)
        )
        result = ObjectMaterialAnalysis(
            source_object_id=resolved_source_object_id,
            slots=analyses,
        )
        logger.debug(
            "Analyzed %d material slots for '%s' target=%s: kinds=%s channels=%s",
            len(result.slots),
            object_name,
            target,
            tuple(slot.kind.value for slot in result.slots),
            tuple(
                tuple(channel.value for channel in slot.semantic_channels)
                for slot in result.slots
            ),
        )
        return result
    except MaterialAnalysisError:
        raise
    except Exception as exc:
        logger.exception("Failed to analyze materials for '%s'", object_name)
        raise MaterialAnalysisError(
            f"Failed to analyze materials for '{object_name}': {exc}"
        ) from exc


__all__ = [
    "MaterialAnalysisError",
    "analyse_material_slot",
    "analyse_object_materials",
    "render_target_from_engine",
]
