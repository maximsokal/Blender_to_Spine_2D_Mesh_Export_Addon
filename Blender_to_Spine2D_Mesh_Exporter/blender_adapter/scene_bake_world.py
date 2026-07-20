"""Read Blender World state into an immutable scene-bake snapshot."""

from __future__ import annotations

import logging
from typing import Any

from ..domain.baking.context import WorldBakeSnapshot
from .scene_bake_error import SceneBakeAnalysisError
from .scene_bake_rna import animated, color_tuple, finite_float, name

logger = logging.getLogger(__name__)


def active_world_output(node_tree: Any) -> Any | None:
    try:
        outputs = tuple(
            node for node in node_tree.nodes
            if str(getattr(node, "type", "")) == "OUTPUT_WORLD"
        )
    except Exception:
        return None
    active = tuple(node for node in outputs if bool(getattr(node, "is_active_output", False)))
    return active[0] if active else (outputs[0] if outputs else None)


def input_socket(node: Any, socket_name: str) -> Any | None:
    inputs = getattr(node, "inputs", None)
    if inputs is None:
        return None
    getter = getattr(inputs, "get", None)
    if callable(getter):
        try:
            result = getter(socket_name)
            if result is not None:
                return result
        except Exception:
            logger.debug("World socket lookup failed", exc_info=True)
    try:
        return next(socket for socket in inputs if str(getattr(socket, "name", "")) == socket_name)
    except Exception:
        return None


def background_strength(world: Any) -> float | None:
    node_tree = getattr(world, "node_tree", None)
    if node_tree is None:
        return 1.0
    output = active_world_output(node_tree)
    if output is None:
        return None
    surface = input_socket(output, "Surface")
    try:
        links = tuple(getattr(surface, "links", ())) if surface is not None else ()
    except Exception:
        links = ()
    if not links:
        return 0.0
    source_node = getattr(links[0], "from_node", None)
    if str(getattr(source_node, "type", "")) != "BACKGROUND":
        return None
    strength = input_socket(source_node, "Strength")
    if strength is None or bool(getattr(strength, "is_linked", False)):
        return None
    try:
        return max(0.0, finite_float(getattr(strength, "default_value", 0.0), label="World Background Strength"))
    except SceneBakeAnalysisError:
        logger.debug("Unable to read World Background Strength", exc_info=True)
        return None


def analyse_world(scene: Any) -> WorldBakeSnapshot | None:
    world = getattr(scene, "world", None)
    if world is None:
        return None
    world_name = name(world)
    if not world_name:
        raise SceneBakeAnalysisError("Scene World has an empty name")
    use_nodes = bool(getattr(world, "use_nodes", False))
    node_tree = getattr(world, "node_tree", None)
    try:
        node_types = tuple(sorted({str(getattr(node, "type", "") or "UNKNOWN") for node in getattr(node_tree, "nodes", ())}))
    except Exception:
        node_types = ()
    try:
        return WorldBakeSnapshot(
            world_name=world_name,
            color=color_tuple(getattr(world, "color", (0.0, 0.0, 0.0)), default=(0.0, 0.0, 0.0), label=f"World '{world_name}' color"),
            use_nodes=use_nodes,
            node_types=node_types,
            background_strength=background_strength(world),
            animated=animated(world, node_tree),
        )
    except SceneBakeAnalysisError:
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise SceneBakeAnalysisError(f"Unable to build World snapshot for '{world_name}': {exc}") from exc


__all__ = ["active_world_output", "analyse_world", "background_strength", "input_socket"]
