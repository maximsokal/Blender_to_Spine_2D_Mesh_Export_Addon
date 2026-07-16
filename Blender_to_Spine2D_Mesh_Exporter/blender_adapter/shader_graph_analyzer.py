"""Read active Blender material outputs into immutable semantic graph snapshots."""

from __future__ import annotations

import logging
from typing import Any, Iterable

from ..domain.baking.graph import (
    MaterialDependencyKind,
    MaterialGraphSnapshot,
    MaterialSemanticChannel,
    ShaderLinkSnapshot,
    ShaderNodeSnapshot,
)

logger = logging.getLogger(__name__)


class MaterialGraphAnalysisError(RuntimeError):
    """Raised when a Blender node tree cannot be inspected deterministically."""


_TEMPORARY_PREFIXES = ("TEMP_BAKE_", "TEMP_UV_", "__Spine2D_BakeTarget_", "__Spine2D_Proxy_")
_VIEW_NODE_TYPES = frozenset({"FRESNEL", "LAYER_WEIGHT", "LIGHT_PATH"})
_OBJECT_NODE_TYPES = frozenset({"OBJECT_INFO", "TEX_COORD"})
_GEOMETRY_NODE_TYPES = frozenset(
    {"NEW_GEOMETRY", "NORMAL", "NORMAL_MAP", "BUMP", "TANGENT", "BEVEL"}
)
_CAMERA_SHADER_TYPES = frozenset({"BSDF_GLASS", "BSDF_REFRACTION", "BSDF_GLOSSY"})
_SCENE_LIGHTING_SHADER_TYPES = frozenset(
    {
        "BSDF_TRANSLUCENT",
        "BSDF_TOON",
        "SUBSURFACE_SCATTERING",
        "BSDF_HAIR",
        "BSDF_HAIR_PRINCIPLED",
    }
)
_SURFACE_SHADER_TYPES = frozenset(
    {
        "BSDF_PRINCIPLED",
        "BSDF_DIFFUSE",
        "BSDF_GLOSSY",
        "BSDF_GLASS",
        "BSDF_REFRACTION",
        "BSDF_TRANSLUCENT",
        "BSDF_TOON",
        "SUBSURFACE_SCATTERING",
        "BSDF_HAIR",
        "BSDF_HAIR_PRINCIPLED",
        "HOLDOUT",
    }
)


def _material_name(material: Any) -> str:
    value = str(
        getattr(material, "name_full", None)
        or getattr(material, "name", None)
        or ""
    ).strip()
    if not value:
        raise MaterialGraphAnalysisError("Material name is empty")
    return value


def _node_type(node: Any) -> str:
    value = str(getattr(node, "type", "") or "").strip()
    return value or "UNKNOWN"


def _node_name(node: Any) -> str:
    value = str(getattr(node, "name", "") or "").strip()
    if value:
        return value
    return f"{_node_type(node)}_{id(node)}"


def _is_temporary_node(node: Any) -> bool:
    return node is not None and _node_name(node).startswith(_TEMPORARY_PREFIXES)


def _socket_name(socket: Any) -> str:
    value = str(getattr(socket, "name", "") or "").strip()
    return value or "Socket"


def _iter_nodes(node_tree: Any) -> tuple[Any, ...]:
    try:
        return tuple(node for node in node_tree.nodes if not _is_temporary_node(node))
    except Exception as exc:
        raise MaterialGraphAnalysisError("Unable to iterate material nodes") from exc


def _iter_links(node_tree: Any) -> tuple[Any, ...]:
    try:
        links = tuple(getattr(node_tree, "links", ()))
    except Exception as exc:
        raise MaterialGraphAnalysisError("Unable to iterate material links") from exc
    return tuple(
        link
        for link in links
        if not _is_temporary_node(getattr(link, "from_node", None))
        and not _is_temporary_node(getattr(link, "to_node", None))
    )


def _find_active_output(nodes: tuple[Any, ...]) -> Any | None:
    outputs = tuple(node for node in nodes if _node_type(node) == "OUTPUT_MATERIAL")
    if not outputs:
        return None
    active = tuple(node for node in outputs if bool(getattr(node, "is_active_output", False)))
    return active[0] if active else outputs[0]


def _input_socket(node: Any, name: str) -> Any | None:
    inputs = getattr(node, "inputs", None)
    if inputs is None:
        return None
    getter = getattr(inputs, "get", None)
    if callable(getter):
        try:
            socket = getter(name)
            if socket is not None:
                return socket
        except Exception:
            logger.debug("Socket lookup by name failed", exc_info=True)
    try:
        for socket in inputs:
            if _socket_name(socket) == name:
                return socket
    except Exception:
        return None
    return None


def _first_input_socket(node: Any, names: tuple[str, ...]) -> Any | None:
    for name in names:
        socket = _input_socket(node, name)
        if socket is not None:
            return socket
    return None


def _incoming_by_node_name(links: tuple[Any, ...]) -> dict[str, tuple[Any, ...]]:
    grouped: dict[str, list[Any]] = {}
    for link in links:
        to_node = getattr(link, "to_node", None)
        if to_node is None:
            continue
        grouped.setdefault(_node_name(to_node), []).append(link)
    return {
        key: tuple(
            sorted(
                values,
                key=lambda link: (
                    _node_name(getattr(link, "from_node", None)).casefold(),
                    _socket_name(getattr(link, "from_socket", None)).casefold(),
                    _socket_name(getattr(link, "to_socket", None)).casefold(),
                ),
            )
        )
        for key, values in grouped.items()
    }


def _reachable_from_nodes(
    roots: Iterable[Any],
    incoming: dict[str, tuple[Any, ...]],
) -> tuple[set[str], tuple[Any, ...]]:
    pending = list(roots)
    seen: set[str] = set()
    used_links: list[Any] = []
    while pending:
        node = pending.pop()
        if node is None:
            continue
        node_name = _node_name(node)
        if node_name in seen:
            continue
        seen.add(node_name)
        for link in incoming.get(node_name, ()):
            used_links.append(link)
            pending.append(getattr(link, "from_node", None))
    return seen, tuple(used_links)


def _socket_roots(socket: Any | None, links: tuple[Any, ...]) -> tuple[Any, ...]:
    if socket is None:
        return ()
    try:
        socket_links = tuple(getattr(socket, "links", ()))
    except Exception:
        socket_links = ()
    if socket_links:
        return tuple(
            getattr(link, "from_node", None)
            for link in socket_links
            if getattr(link, "from_node", None) is not None
        )

    socket_name = _socket_name(socket)
    return tuple(
        getattr(link, "from_node", None)
        for link in links
        if _socket_name(getattr(link, "to_socket", None)) == socket_name
    )


def _numeric_default(socket: Any | None, default: float) -> float:
    if socket is None:
        return default
    value = getattr(socket, "default_value", default)
    try:
        if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
            return float(value[0])
        return float(value)
    except Exception:
        return default


def _socket_enabled(socket: Any | None, *, default: float = 0.0) -> bool:
    if socket is None:
        return False
    if bool(getattr(socket, "is_linked", False)):
        return True
    return abs(_numeric_default(socket, default)) > 1e-8


def _color_nonzero(socket: Any | None) -> bool:
    if socket is None:
        return False
    if bool(getattr(socket, "is_linked", False)):
        return True
    value = getattr(socket, "default_value", None)
    try:
        return any(abs(float(value[index])) > 1e-8 for index in range(3))
    except Exception:
        return False


def _principled_emission_enabled(node: Any) -> bool:
    if _node_type(node) != "BSDF_PRINCIPLED":
        return False
    color = _first_input_socket(node, ("Emission Color", "Emission"))
    strength = _input_socket(node, "Emission Strength")
    return _color_nonzero(color) and _numeric_default(strength, 1.0) > 1e-8


def _principled_alpha_enabled(node: Any) -> bool:
    if _node_type(node) != "BSDF_PRINCIPLED":
        return False
    alpha = _input_socket(node, "Alpha")
    if alpha is None:
        return False
    return bool(getattr(alpha, "is_linked", False)) or _numeric_default(alpha, 1.0) < 0.999999


def _principled_dependencies(node: Any) -> set[MaterialDependencyKind]:
    result: set[MaterialDependencyKind] = set()
    if _node_type(node) != "BSDF_PRINCIPLED":
        return result

    transmission = _first_input_socket(node, ("Transmission Weight", "Transmission"))
    metallic = _input_socket(node, "Metallic")
    coat = _first_input_socket(node, ("Coat Weight", "Clearcoat"))
    subsurface = _first_input_socket(node, ("Subsurface Weight", "Subsurface"))
    sheen = _first_input_socket(node, ("Sheen Weight", "Sheen"))

    if _socket_enabled(transmission):
        result.update(
            {
                MaterialDependencyKind.CAMERA,
                MaterialDependencyKind.VIEW,
                MaterialDependencyKind.WORLD,
                MaterialDependencyKind.SCENE_OBJECTS,
                MaterialDependencyKind.REFLECTION,
                MaterialDependencyKind.TRANSMISSION,
            }
        )
    if _socket_enabled(metallic) or _socket_enabled(coat):
        result.update(
            {
                MaterialDependencyKind.CAMERA,
                MaterialDependencyKind.VIEW,
                MaterialDependencyKind.WORLD,
                MaterialDependencyKind.SCENE_OBJECTS,
                MaterialDependencyKind.REFLECTION,
            }
        )
    if _socket_enabled(subsurface) or _socket_enabled(sheen):
        result.add(MaterialDependencyKind.LIGHTING)
    return result


def _semantic_channels(
    output: Any | None,
    nodes: tuple[Any, ...],
    links: tuple[Any, ...],
    incoming: dict[str, tuple[Any, ...]],
) -> tuple[MaterialSemanticChannel, ...]:
    if output is None:
        surface_nodes = nodes
        volume_nodes: tuple[Any, ...] = ()
        displacement_nodes: tuple[Any, ...] = ()
    else:
        surface_names, _ = _reachable_from_nodes(
            _socket_roots(_input_socket(output, "Surface"), links),
            incoming,
        )
        volume_names, _ = _reachable_from_nodes(
            _socket_roots(_input_socket(output, "Volume"), links),
            incoming,
        )
        displacement_names, _ = _reachable_from_nodes(
            _socket_roots(_input_socket(output, "Displacement"), links),
            incoming,
        )
        surface_nodes = tuple(node for node in nodes if _node_name(node) in surface_names)
        volume_nodes = tuple(node for node in nodes if _node_name(node) in volume_names)
        displacement_nodes = tuple(
            node for node in nodes if _node_name(node) in displacement_names
        )

    surface_types = {_node_type(node) for node in surface_nodes}
    channels: set[MaterialSemanticChannel] = set()
    if surface_types & _SURFACE_SHADER_TYPES:
        channels.add(MaterialSemanticChannel.SURFACE_COLOR)
    if "EMISSION" in surface_types or any(
        _principled_emission_enabled(node) for node in surface_nodes
    ):
        channels.add(MaterialSemanticChannel.SURFACE_EMISSION)
    if "BSDF_TRANSPARENT" in surface_types or any(
        _principled_alpha_enabled(node) for node in surface_nodes
    ):
        channels.add(MaterialSemanticChannel.ALPHA)
    if volume_nodes:
        channels.add(MaterialSemanticChannel.VOLUME)
    if displacement_nodes:
        channels.add(MaterialSemanticChannel.DISPLACEMENT)

    known_non_surface = {
        "EMISSION",
        "OUTPUT_MATERIAL",
        "MIX_SHADER",
        "ADD_SHADER",
        "BSDF_TRANSPARENT",
    }
    if surface_nodes and not channels and not surface_types.issubset(known_non_surface):
        channels.add(MaterialSemanticChannel.SURFACE_COLOR)
    return tuple(sorted(channels, key=lambda value: value.value))


def _dependencies(
    material: Any,
    node_tree: Any,
    reachable_nodes: tuple[Any, ...],
) -> tuple[MaterialDependencyKind, ...]:
    result: set[MaterialDependencyKind] = set()
    node_types = {_node_type(node) for node in reachable_nodes}
    if "TEX_IMAGE" in node_types or "TEX_ENVIRONMENT" in node_types:
        result.add(MaterialDependencyKind.IMAGE)
    if node_types & _VIEW_NODE_TYPES:
        result.update({MaterialDependencyKind.VIEW, MaterialDependencyKind.CAMERA})
    if "FRESNEL" in node_types or "LAYER_WEIGHT" in node_types:
        result.add(MaterialDependencyKind.REFLECTION)
    if "LIGHT_PATH" in node_types:
        result.add(MaterialDependencyKind.LIGHTING)
    if node_types & _OBJECT_NODE_TYPES:
        result.add(MaterialDependencyKind.OBJECT)
    if node_types & _GEOMETRY_NODE_TYPES:
        result.add(MaterialDependencyKind.GEOMETRY)
    if "AMBIENT_OCCLUSION" in node_types:
        result.update(
            {
                MaterialDependencyKind.OCCLUSION,
                MaterialDependencyKind.SCENE_OBJECTS,
            }
        )
    if node_types & _SCENE_LIGHTING_SHADER_TYPES:
        result.add(MaterialDependencyKind.LIGHTING)
    if node_types & _CAMERA_SHADER_TYPES:
        result.update(
            {
                MaterialDependencyKind.CAMERA,
                MaterialDependencyKind.VIEW,
                MaterialDependencyKind.WORLD,
                MaterialDependencyKind.SCENE_OBJECTS,
                MaterialDependencyKind.REFLECTION,
            }
        )
    if "BSDF_GLASS" in node_types or "BSDF_REFRACTION" in node_types:
        result.add(MaterialDependencyKind.TRANSMISSION)
    if "TEX_ENVIRONMENT" in node_types:
        result.update({MaterialDependencyKind.VIEW, MaterialDependencyKind.CAMERA})
    if "GROUP" in node_types:
        result.add(MaterialDependencyKind.NODE_GROUP)

    for node in reachable_nodes:
        result.update(_principled_dependencies(node))
        if _node_type(node) != "TEX_IMAGE":
            continue
        image = getattr(node, "image", None)
        source = str(getattr(image, "source", "") or "").upper()
        duration = int(getattr(image, "frame_duration", 1) or 1) if image else 1
        if source in {"SEQUENCE", "MOVIE"} or duration > 1:
            result.add(MaterialDependencyKind.TIME)
    if getattr(material, "animation_data", None) is not None or getattr(
        node_tree, "animation_data", None
    ) is not None:
        result.add(MaterialDependencyKind.TIME)
    return tuple(sorted(result, key=lambda value: value.value))


def analyse_material_graph(material: Any) -> MaterialGraphSnapshot:
    """Analyze only nodes reachable from the active Material Output when possible."""

    if material is None:
        raise MaterialGraphAnalysisError("material cannot be None")
    material_name = _material_name(material)
    node_tree = getattr(material, "node_tree", None)
    if node_tree is None:
        raise MaterialGraphAnalysisError(f"Material '{material_name}' has no node tree")

    nodes = _iter_nodes(node_tree)
    links = _iter_links(node_tree)
    output = _find_active_output(nodes)
    issues: list[str] = []
    incoming = _incoming_by_node_name(links)
    if output is None:
        issues.append(
            "Active Material Output was not found; semantic analysis used all nodes"
        )
        reachable_nodes = nodes
        reachable_links = links
    else:
        reachable_names, reachable_links = _reachable_from_nodes((output,), incoming)
        reachable_nodes = tuple(
            node for node in nodes if _node_name(node) in reachable_names
        )

    node_snapshots = tuple(
        ShaderNodeSnapshot(
            node_id=_node_name(node),
            node_type=_node_type(node),
            node_name=_node_name(node),
        )
        for node in sorted(reachable_nodes, key=lambda item: _node_name(item).casefold())
    )
    link_snapshots = tuple(
        ShaderLinkSnapshot(
            from_node_id=_node_name(getattr(link, "from_node", None)),
            from_socket=_socket_name(getattr(link, "from_socket", None)),
            to_node_id=_node_name(getattr(link, "to_node", None)),
            to_socket=_socket_name(getattr(link, "to_socket", None)),
        )
        for link in sorted(
            reachable_links,
            key=lambda item: (
                _node_name(getattr(item, "from_node", None)).casefold(),
                _socket_name(getattr(item, "from_socket", None)).casefold(),
                _node_name(getattr(item, "to_node", None)).casefold(),
                _socket_name(getattr(item, "to_socket", None)).casefold(),
            ),
        )
    )
    snapshot = MaterialGraphSnapshot(
        material_name=material_name,
        active_output_node_id=None if output is None else _node_name(output),
        reachable_nodes=node_snapshots,
        reachable_links=link_snapshots,
        semantic_channels=_semantic_channels(output, reachable_nodes, links, incoming),
        dependencies=_dependencies(material, node_tree, reachable_nodes),
        issues=tuple(issues),
    )
    logger.debug(
        "Analyzed reachable shader graph '%s': nodes=%d channels=%s dependencies=%s",
        material_name,
        len(snapshot.reachable_nodes),
        tuple(value.value for value in snapshot.semantic_channels),
        tuple(value.value for value in snapshot.dependencies),
    )
    return snapshot
