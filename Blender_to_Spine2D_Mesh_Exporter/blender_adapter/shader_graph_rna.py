"""Strict Blender 5.2 RNA primitives for shader-graph analysis."""

from __future__ import annotations

import logging
from typing import Any

from .shader_graph_error import MaterialGraphAnalysisError


logger = logging.getLogger(__name__)

TEMPORARY_PREFIXES = (
    "TEMP_BAKE_",
    "TEMP_UV_",
    "__Spine2D_BakeTarget_",
    "__Spine2D_Proxy_",
)
VALID_RENDER_TARGETS = frozenset({"ALL", "CYCLES", "EEVEE"})


def material_name(material: Any) -> str:
    value = str(
        getattr(material, "name_full", None)
        or getattr(material, "name", None)
        or ""
    ).strip()
    if not value:
        raise MaterialGraphAnalysisError("Material name is empty")
    return value


def node_type(node: Any) -> str:
    value = str(getattr(node, "type", "") or "").strip()
    return value or "UNKNOWN"


def rna_identity(value: Any) -> int:
    """Return a stable Blender RNA identity with a test-double fallback."""

    if value is None:
        return 0
    pointer = getattr(value, "as_pointer", None)
    if callable(pointer):
        try:
            resolved = int(pointer())
            if resolved:
                return resolved
        except Exception:
            logger.debug("RNA pointer lookup failed", exc_info=True)
    return id(value)


def node_name(node: Any) -> str:
    value = str(getattr(node, "name", "") or "").strip()
    if value:
        return value
    return f"{node_type(node)}_{rna_identity(node)}"


def tree_name(node_tree: Any) -> str:
    value = str(
        getattr(node_tree, "name_full", None)
        or getattr(node_tree, "name", None)
        or ""
    ).strip()
    return value or f"NodeTree_{rna_identity(node_tree)}"


def is_temporary_node(node: Any) -> bool:
    return node is not None and node_name(node).startswith(TEMPORARY_PREFIXES)


def socket_name(socket: Any) -> str:
    value = str(getattr(socket, "name", "") or "").strip()
    return value or "Socket"


def socket_identifier(socket: Any) -> str:
    return str(getattr(socket, "identifier", "") or "").strip()


def normalise_render_target(value: str | None) -> str:
    """Normalize only ShaderNodeTree targets valid in Blender 5.2+.

    ``BLENDER_EEVEE`` is Blender's Scene render-engine identifier and maps to
    the ShaderNodeTree target ``EEVEE``. Removed identifiers and partial names
    are rejected rather than guessed.
    """

    target = str(value or "ALL").strip().upper()
    if target in VALID_RENDER_TARGETS:
        return target
    if target == "BLENDER_EEVEE":
        return "EEVEE"
    raise MaterialGraphAnalysisError(
        f"Unsupported Blender 5.2 shader render target: {value!r}"
    )


def iter_collection(value: Any, *, label: str) -> tuple[Any, ...]:
    try:
        return tuple(value or ())
    except Exception as exc:
        raise MaterialGraphAnalysisError(f"Unable to iterate {label}") from exc


def iter_nodes(node_tree: Any) -> tuple[Any, ...]:
    return tuple(
        node
        for node in iter_collection(getattr(node_tree, "nodes", ()), label="nodes")
        if not is_temporary_node(node)
    )


def iter_links(node_tree: Any) -> tuple[Any, ...]:
    return tuple(
        link
        for link in iter_collection(getattr(node_tree, "links", ()), label="links")
        if not is_temporary_node(getattr(link, "from_node", None))
        and not is_temporary_node(getattr(link, "to_node", None))
    )


def find_active_node(nodes: tuple[Any, ...], required_node_type: str) -> Any | None:
    matches = tuple(node for node in nodes if node_type(node) == required_node_type)
    if not matches:
        return None
    active = tuple(
        node for node in matches if bool(getattr(node, "is_active_output", False))
    )
    return active[0] if active else matches[0]


def node_output_target(node: Any) -> str:
    return normalise_render_target(getattr(node, "target", "ALL"))


def find_material_output(
    node_tree: Any,
    nodes: tuple[Any, ...],
    render_target: str,
) -> Any | None:
    """Resolve the effective Material Output for one Blender 5.2 target."""

    target = normalise_render_target(render_target)
    getter = getattr(node_tree, "get_output_node", None)
    if callable(getter):
        try:
            candidate = getter(target)
        except Exception as exc:
            raise MaterialGraphAnalysisError(
                f"ShaderNodeTree.get_output_node({target!r}) failed"
            ) from exc
        if candidate is not None:
            if node_type(candidate) != "OUTPUT_MATERIAL":
                raise MaterialGraphAnalysisError(
                    "ShaderNodeTree.get_output_node returned a non-Material Output node"
                )
            if is_temporary_node(candidate):
                raise MaterialGraphAnalysisError(
                    "ShaderNodeTree.get_output_node returned a temporary bake node"
                )
            return candidate

    outputs = tuple(node for node in nodes if node_type(node) == "OUTPUT_MATERIAL")
    if not outputs:
        return None
    exact = tuple(node for node in outputs if node_output_target(node) == target)
    generic = tuple(node for node in outputs if node_output_target(node) == "ALL")
    candidates = exact or generic
    if not candidates:
        return None
    active = tuple(
        node for node in candidates if bool(getattr(node, "is_active_output", False))
    )
    return active[0] if active else candidates[0]


def socket_by_name(collection: Any, name: str) -> Any | None:
    if collection is None:
        return None
    getter = getattr(collection, "get", None)
    if callable(getter):
        try:
            socket = getter(name)
            if socket is not None:
                return socket
        except Exception:
            logger.debug("Socket lookup by name failed", exc_info=True)
    try:
        for socket in collection:
            if socket_name(socket) == name:
                return socket
    except Exception:
        return None
    return None


def input_socket(node: Any, name: str) -> Any | None:
    return socket_by_name(getattr(node, "inputs", None), name)


def first_input_socket(node: Any, names: tuple[str, ...]) -> Any | None:
    for name in names:
        socket = input_socket(node, name)
        if socket is not None:
            return socket
    return None


def socket_index(collection: Any, target: Any) -> int | None:
    if collection is None or target is None:
        return None
    try:
        for index, socket in enumerate(collection):
            if socket is target:
                return index
    except Exception:
        return None
    return None


def same_socket(first: Any | None, second: Any | None) -> bool:
    if first is None or second is None:
        return False
    if first is second:
        return True
    first_direction = getattr(first, "is_output", None)
    second_direction = getattr(second, "is_output", None)
    if (
        first_direction is not None
        and second_direction is not None
        and bool(first_direction) != bool(second_direction)
    ):
        return False
    first_node = getattr(first, "node", None)
    second_node = getattr(second, "node", None)
    if (
        first_node is not None
        and second_node is not None
        and rna_identity(first_node) != rna_identity(second_node)
    ):
        return False
    first_identifier = socket_identifier(first)
    second_identifier = socket_identifier(second)
    if first_identifier and second_identifier:
        return first_identifier == second_identifier
    return socket_name(first) == socket_name(second)


def matching_socket(collection: Any, reference: Any) -> Any | None:
    """Resolve a group instance socket to its node-tree interface socket."""

    if collection is None or reference is None:
        return None
    sockets = iter_collection(collection, label="group sockets")
    identifier = socket_identifier(reference)
    if identifier:
        matches = tuple(
            item for item in sockets if socket_identifier(item) == identifier
        )
        if len(matches) == 1:
            return matches[0]
    name = socket_name(reference)
    matches = tuple(item for item in sockets if socket_name(item) == name)
    if len(matches) == 1:
        return matches[0]
    reference_node = getattr(reference, "node", None)
    if reference_node is not None:
        source_collection = (
            getattr(reference_node, "outputs", None)
            if bool(getattr(reference, "is_output", False))
            else getattr(reference_node, "inputs", None)
        )
        index = socket_index(source_collection, reference)
        if index is not None and index < len(sockets):
            return sockets[index]
    return None


def numeric_default(socket: Any | None, default: float) -> float:
    if socket is None:
        return default
    value = getattr(socket, "default_value", default)
    try:
        if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
            return float(value[0])
        return float(value)
    except Exception:
        return default


def socket_enabled(socket: Any | None, *, default: float = 0.0) -> bool:
    if socket is None:
        return False
    if bool(getattr(socket, "is_linked", False)):
        return True
    return abs(numeric_default(socket, default)) > 1e-8


def color_nonzero(socket: Any | None) -> bool:
    if socket is None:
        return False
    if bool(getattr(socket, "is_linked", False)):
        return True
    value = getattr(socket, "default_value", None)
    try:
        return any(abs(float(value[index])) > 1e-8 for index in range(3))
    except Exception:
        return False


__all__ = [
    "TEMPORARY_PREFIXES",
    "VALID_RENDER_TARGETS",
    "color_nonzero",
    "find_active_node",
    "find_material_output",
    "first_input_socket",
    "input_socket",
    "is_temporary_node",
    "iter_collection",
    "iter_links",
    "iter_nodes",
    "matching_socket",
    "material_name",
    "node_name",
    "node_output_target",
    "node_type",
    "normalise_render_target",
    "numeric_default",
    "rna_identity",
    "same_socket",
    "socket_by_name",
    "socket_enabled",
    "socket_identifier",
    "socket_index",
    "socket_name",
    "tree_name",
]
