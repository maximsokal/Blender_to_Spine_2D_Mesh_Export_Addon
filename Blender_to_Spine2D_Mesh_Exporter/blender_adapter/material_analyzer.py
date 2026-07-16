"""Read Blender material slots into immutable semantic baking analysis.

Legacy material kinds remain available for compatibility, while every node material
also receives a graph snapshot rooted at the active Material Output. Strategy
selection therefore depends on connected shader semantics instead of all nodes that
happen to exist in the editor.
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
    analyse_material_graph,
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
        "GROUP",
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
    return name.startswith(("TEMP_BAKE_", "TEMP_UV_", "__Spine2D_BakeTarget_"))


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


def analyse_material_slot(slot_index: int, material: Any | None) -> MaterialAnalysis:
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
        nodes = tuple(node_tree.nodes)
    except Exception as exc:
        raise MaterialAnalysisError(
            f"Unable to iterate nodes of material '{material_name}'"
        ) from exc

    relevant_nodes = tuple(node for node in nodes if not _is_temporary_bake_node(node))
    node_types = tuple(sorted({_node_type(node) for node in relevant_nodes}))
    procedural = any(
        _node_type(node) in _PROCEDURAL_NODE_TYPES for node in relevant_nodes
    )

    dependencies_by_key: dict[tuple[str, str, str | None, int], ImageDependency] = {}
    issues: list[str] = []
    image_node_count = 0
    for node in relevant_nodes:
        if _node_type(node) != "TEX_IMAGE":
            continue
        image_node_count += 1
        dependency, issue = _image_dependency(node)
        if issue is not None:
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

    if image_node_count and len(dependencies) != image_node_count:
        kind = MaterialKind.UNSUPPORTED
    elif dependencies and procedural:
        kind = MaterialKind.MIXED
    elif dependencies:
        kind = MaterialKind.IMAGE
    elif procedural:
        kind = MaterialKind.PROCEDURAL
    else:
        kind = MaterialKind.SOLID_COLOR

    graph = None
    try:
        graph = analyse_material_graph(material)
        issues.extend(graph.issues)
        if graph.active_output_node_id is None and not graph.semantic_channels:
            # Preserve the historical no-output behavior. The recursive snapshot remains
            # useful as a diagnostic, but an empty graph channel set must not suppress the
            # legacy node-type fallback used by synthetic files and damaged materials.
            graph = None
    except MaterialGraphAnalysisError as exc:
        # Graph analysis is richer than the legacy kind classifier, but a failure to
        # produce it must be visible. Do not silently invent semantics.
        issues.append(f"Shader graph analysis failed: {exc}")
        logger.warning(
            "Shader graph analysis failed for material '%s'",
            material_name,
            exc_info=True,
        )

    return MaterialAnalysis(
        slot_index=slot_index,
        material_name=material_name,
        kind=kind,
        node_types=node_types,
        image_dependencies=dependencies,
        issues=tuple(issues),
        graph=graph,
    )


def analyse_object_materials(
    obj: Any,
    *,
    source_object_id: str | None = None,
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

    try:
        material_slots = tuple(getattr(obj, "material_slots", ()))
        analyses = tuple(
            analyse_material_slot(slot_index, getattr(slot, "material", None))
            for slot_index, slot in enumerate(material_slots)
        )
        result = ObjectMaterialAnalysis(
            source_object_id=resolved_source_object_id,
            slots=analyses,
        )
        logger.debug(
            "Analyzed %d material slots for '%s': kinds=%s channels=%s",
            len(result.slots),
            object_name,
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
