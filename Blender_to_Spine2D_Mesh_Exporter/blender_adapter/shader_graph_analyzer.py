"""Read active Blender material outputs into immutable semantic graph snapshots.

The analyzer expands reachable Shader Node Groups recursively. Traversal follows the
actual socket path through Group Output and Group Input nodes, so unused group inputs do
not leak dependencies into the material plan. Every nested node receives a deterministic
instance-qualified node ID and an explicit ``group_path`` in the immutable snapshot.

Renderer-specific Material Output nodes and muted-node ``internal_links`` are respected.
The detailed adapter result keeps live Blender nodes private to the adapter layer; public
domain snapshots remain immutable and ``bpy``-free.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

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


_TEMPORARY_PREFIXES = (
    "TEMP_BAKE_",
    "TEMP_UV_",
    "__Spine2D_BakeTarget_",
    "__Spine2D_Proxy_",
)
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
_GROUP_NODE_TYPES = frozenset({"GROUP"})
_GROUP_INPUT_TYPES = frozenset({"GROUP_INPUT"})
_GROUP_OUTPUT_TYPES = frozenset({"GROUP_OUTPUT"})
_VALID_RENDER_TARGETS = frozenset({"ALL", "CYCLES", "EEVEE"})
_MAX_GROUP_DEPTH = 64


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


def _rna_identity(value: Any) -> int:
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


def _node_name(node: Any) -> str:
    value = str(getattr(node, "name", "") or "").strip()
    if value:
        return value
    return f"{_node_type(node)}_{_rna_identity(node)}"


def _tree_name(node_tree: Any) -> str:
    value = str(
        getattr(node_tree, "name_full", None)
        or getattr(node_tree, "name", None)
        or ""
    ).strip()
    return value or f"NodeTree_{_rna_identity(node_tree)}"


def _is_temporary_node(node: Any) -> bool:
    return node is not None and _node_name(node).startswith(_TEMPORARY_PREFIXES)


def _socket_name(socket: Any) -> str:
    value = str(getattr(socket, "name", "") or "").strip()
    return value or "Socket"


def _socket_identifier(socket: Any) -> str:
    return str(getattr(socket, "identifier", "") or "").strip()


def _normalise_render_target(value: str | None) -> str:
    target = str(value or "ALL").strip().upper()
    if target in _VALID_RENDER_TARGETS:
        return target
    if "CYCLE" in target:
        return "CYCLES"
    if "EEVEE" in target:
        return "EEVEE"
    return "ALL"


def _iter_collection(value: Any, *, label: str) -> tuple[Any, ...]:
    try:
        return tuple(value or ())
    except Exception as exc:
        raise MaterialGraphAnalysisError(f"Unable to iterate {label}") from exc


def _iter_nodes(node_tree: Any) -> tuple[Any, ...]:
    return tuple(
        node
        for node in _iter_collection(getattr(node_tree, "nodes", ()), label="nodes")
        if not _is_temporary_node(node)
    )


def _iter_links(node_tree: Any) -> tuple[Any, ...]:
    return tuple(
        link
        for link in _iter_collection(getattr(node_tree, "links", ()), label="links")
        if not _is_temporary_node(getattr(link, "from_node", None))
        and not _is_temporary_node(getattr(link, "to_node", None))
    )


def _find_active_node(nodes: tuple[Any, ...], node_type: str) -> Any | None:
    matches = tuple(node for node in nodes if _node_type(node) == node_type)
    if not matches:
        return None
    active = tuple(
        node for node in matches if bool(getattr(node, "is_active_output", False))
    )
    return active[0] if active else matches[0]


def _node_output_target(node: Any) -> str:
    return _normalise_render_target(getattr(node, "target", "ALL"))


def _find_material_output(
    node_tree: Any,
    nodes: tuple[Any, ...],
    render_target: str,
) -> Any | None:
    """Resolve the effective Material Output for one render backend."""

    target = _normalise_render_target(render_target)
    getter = getattr(node_tree, "get_output_node", None)
    if callable(getter):
        candidates = (target, "ALL") if target != "ALL" else ("ALL",)
        for candidate_target in candidates:
            try:
                candidate = getter(candidate_target)
            except TypeError:
                try:
                    candidate = getter(target=candidate_target)
                except Exception:
                    logger.debug(
                        "Material Output lookup failed for target %s",
                        candidate_target,
                        exc_info=True,
                    )
                    continue
            except Exception:
                logger.debug(
                    "Material Output lookup failed for target %s",
                    candidate_target,
                    exc_info=True,
                )
                continue
            if (
                candidate is not None
                and _node_type(candidate) == "OUTPUT_MATERIAL"
                and not _is_temporary_node(candidate)
            ):
                return candidate

    outputs = tuple(node for node in nodes if _node_type(node) == "OUTPUT_MATERIAL")
    if not outputs:
        return None
    exact = tuple(node for node in outputs if _node_output_target(node) == target)
    generic = tuple(node for node in outputs if _node_output_target(node) == "ALL")
    candidates = exact or generic or outputs
    active = tuple(
        node for node in candidates if bool(getattr(node, "is_active_output", False))
    )
    return active[0] if active else candidates[0]


def _socket_by_name(collection: Any, name: str) -> Any | None:
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
            if _socket_name(socket) == name:
                return socket
    except Exception:
        return None
    return None


def _input_socket(node: Any, name: str) -> Any | None:
    return _socket_by_name(getattr(node, "inputs", None), name)


def _first_input_socket(node: Any, names: tuple[str, ...]) -> Any | None:
    for name in names:
        socket = _input_socket(node, name)
        if socket is not None:
            return socket
    return None


def _socket_index(collection: Any, target: Any) -> int | None:
    if collection is None or target is None:
        return None
    try:
        for index, socket in enumerate(collection):
            if socket is target:
                return index
    except Exception:
        return None
    return None


def _same_socket(first: Any | None, second: Any | None) -> bool:
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
        and _rna_identity(first_node) != _rna_identity(second_node)
    ):
        return False
    first_identifier = _socket_identifier(first)
    second_identifier = _socket_identifier(second)
    if first_identifier and second_identifier:
        return first_identifier == second_identifier
    return _socket_name(first) == _socket_name(second)


def _matching_socket(collection: Any, reference: Any) -> Any | None:
    """Resolve a group interface socket across Blender API versions."""

    if collection is None or reference is None:
        return None
    sockets = _iter_collection(collection, label="group sockets")
    identifier = _socket_identifier(reference)
    if identifier:
        matches = tuple(
            item for item in sockets if _socket_identifier(item) == identifier
        )
        if len(matches) == 1:
            return matches[0]
    name = _socket_name(reference)
    matches = tuple(item for item in sockets if _socket_name(item) == name)
    if len(matches) == 1:
        return matches[0]
    reference_node = getattr(reference, "node", None)
    if reference_node is not None:
        source_collection = (
            getattr(reference_node, "outputs", None)
            if bool(getattr(reference, "is_output", False))
            else getattr(reference_node, "inputs", None)
        )
        index = _socket_index(source_collection, reference)
        if index is not None and index < len(sockets):
            return sockets[index]
    return None


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
    return bool(getattr(alpha, "is_linked", False)) or _numeric_default(
        alpha, 1.0
    ) < 0.999999


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


@dataclass(frozen=True, slots=True)
class MaterialGraphAnalysisResult:
    """Adapter-only graph result containing the snapshot and live nodes."""

    snapshot: MaterialGraphSnapshot
    reachable_nodes: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, MaterialGraphSnapshot):
            raise TypeError("snapshot must be MaterialGraphSnapshot")
        if not isinstance(self.reachable_nodes, tuple):
            raise TypeError("reachable_nodes must be tuple")


@dataclass(frozen=True, slots=True)
class _GraphFrame:
    node_tree: Any
    group_path: tuple[str, ...]
    parent: "_GraphFrame | None" = None
    parent_group_node: Any | None = None
    tree_stack: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class _ReachableNode:
    node: Any
    frame: _GraphFrame

    @property
    def node_id(self) -> str:
        return "::".join(self.frame.group_path + (_node_name(self.node),))


class _RecursiveGraphWalker:
    def __init__(self, material_name: str, root_tree: Any) -> None:
        self.material_name = material_name
        self.root_tree = root_tree
        self.nodes: dict[str, _ReachableNode] = {}
        self.links: dict[tuple[str, str, str, str], ShaderLinkSnapshot] = {}
        self.channel_nodes: dict[str, set[str]] = {
            "SURFACE": set(),
            "VOLUME": set(),
            "DISPLACEMENT": set(),
            "ALL": set(),
        }
        root_identity = _rna_identity(root_tree)
        self.node_trees: dict[int, Any] = {root_identity: root_tree}
        self.issues: list[str] = []
        self._visited_outputs: set[
            tuple[int, tuple[str, ...], str, str, str]
        ] = set()
        self._visited_inputs: set[
            tuple[int, tuple[str, ...], str, str, str]
        ] = set()

    def root_frame(self) -> _GraphFrame:
        identity = _rna_identity(self.root_tree)
        return _GraphFrame(
            node_tree=self.root_tree,
            group_path=(),
            tree_stack=(identity,),
        )

    def _node_id(self, frame: _GraphFrame, node: Any) -> str:
        return "::".join(frame.group_path + (_node_name(node),))

    def _record_node(self, frame: _GraphFrame, node: Any, channel: str) -> str:
        node_id = self._node_id(frame, node)
        existing = self.nodes.get(node_id)
        if (
            existing is not None
            and _rna_identity(existing.node) != _rna_identity(node)
        ):
            raise MaterialGraphAnalysisError(
                f"Duplicate reachable node ID '{node_id}' in material "
                f"'{self.material_name}'"
            )
        self.nodes.setdefault(node_id, _ReachableNode(node=node, frame=frame))
        self.channel_nodes.setdefault(channel, set()).add(node_id)
        self.channel_nodes["ALL"].add(node_id)
        return node_id

    def _record_link(self, frame: _GraphFrame, link: Any, channel: str) -> None:
        from_node = getattr(link, "from_node", None)
        to_node = getattr(link, "to_node", None)
        if from_node is None or to_node is None:
            return
        from_id = self._record_node(frame, from_node, channel)
        to_id = self._record_node(frame, to_node, channel)
        snapshot = ShaderLinkSnapshot(
            from_node_id=from_id,
            from_socket=_socket_name(getattr(link, "from_socket", None)),
            to_node_id=to_id,
            to_socket=_socket_name(getattr(link, "to_socket", None)),
        )
        key = (
            snapshot.from_node_id,
            snapshot.from_socket,
            snapshot.to_node_id,
            snapshot.to_socket,
        )
        self.links.setdefault(key, snapshot)

    def _incoming_links(self, socket: Any, frame: _GraphFrame) -> tuple[Any, ...]:
        direct = _iter_collection(getattr(socket, "links", ()), label="socket links")
        if direct:
            return tuple(
                link
                for link in direct
                if not _is_temporary_node(getattr(link, "from_node", None))
                and not _is_temporary_node(getattr(link, "to_node", None))
            )
        node = getattr(socket, "node", None)
        identifier = _socket_identifier(socket)
        name = _socket_name(socket)
        return tuple(
            link
            for link in _iter_links(frame.node_tree)
            if getattr(link, "to_node", None) is node
            and (
                getattr(link, "to_socket", None) is socket
                or (
                    identifier
                    and _socket_identifier(getattr(link, "to_socket", None))
                    == identifier
                )
                or _socket_name(getattr(link, "to_socket", None)) == name
            )
        )

    def walk_input(self, socket: Any | None, frame: _GraphFrame, channel: str) -> None:
        if socket is None:
            return
        node = getattr(socket, "node", None)
        node_name = _node_name(node) if node is not None else "SocketOwner"
        key = (
            _rna_identity(frame.node_tree),
            frame.group_path,
            node_name,
            _socket_identifier(socket) or _socket_name(socket),
            channel,
        )
        if key in self._visited_inputs:
            return
        self._visited_inputs.add(key)
        if node is not None:
            self._record_node(frame, node, channel)
        for link in self._incoming_links(socket, frame):
            self._record_link(frame, link, channel)
            self.walk_output(
                getattr(link, "from_node", None),
                getattr(link, "from_socket", None),
                frame,
                channel,
            )

    def walk_output(
        self,
        node: Any | None,
        output_socket: Any | None,
        frame: _GraphFrame,
        channel: str,
    ) -> None:
        if node is None:
            return
        node_id = self._record_node(frame, node, channel)
        key = (
            _rna_identity(frame.node_tree),
            frame.group_path,
            node_id,
            _socket_identifier(output_socket) or _socket_name(output_socket),
            channel,
        )
        if key in self._visited_outputs:
            return
        self._visited_outputs.add(key)

        if bool(getattr(node, "mute", False)):
            self._walk_muted_output(node, output_socket, frame, channel)
            return

        node_type = _node_type(node)
        if node_type in _GROUP_NODE_TYPES:
            self._walk_group_output(node, output_socket, frame, channel)
            return
        if node_type in _GROUP_INPUT_TYPES:
            self._walk_parent_group_input(output_socket, frame, channel)
            return

        for input_socket in _iter_collection(
            getattr(node, "inputs", ()), label="node inputs"
        ):
            self.walk_input(input_socket, frame, channel)

    def _walk_muted_output(
        self,
        node: Any,
        output_socket: Any | None,
        frame: _GraphFrame,
        channel: str,
    ) -> None:
        """Follow Blender's mute bypass mapping instead of its implementation."""

        internal_links = _iter_collection(
            getattr(node, "internal_links", ()),
            label="muted node internal links",
        )
        mapped_inputs: list[Any] = []
        for link in internal_links:
            from_socket = getattr(link, "from_socket", None)
            to_socket = getattr(link, "to_socket", None)
            if _same_socket(to_socket, output_socket):
                mapped_inputs.append(from_socket)
            elif _same_socket(from_socket, output_socket):
                mapped_inputs.append(to_socket)

        valid_inputs = tuple(
            socket
            for socket in mapped_inputs
            if socket is not None
            and _socket_index(getattr(node, "inputs", None), socket) is not None
        )
        if valid_inputs:
            seen_inputs: set[int] = set()
            for input_socket in valid_inputs:
                identity = _rna_identity(input_socket)
                if identity in seen_inputs:
                    continue
                seen_inputs.add(identity)
                self.walk_input(input_socket, frame, channel)
            return

        self.issues.append(
            f"Muted node '{_node_name(node)}' has no unambiguous internal bypass "
            f"for output '{_socket_name(output_socket)}'; all inputs were analyzed "
            "conservatively"
        )
        for input_socket in _iter_collection(
            getattr(node, "inputs", ()), label="node inputs"
        ):
            self.walk_input(input_socket, frame, channel)

    def _walk_group_output(
        self,
        group_node: Any,
        output_socket: Any | None,
        parent_frame: _GraphFrame,
        channel: str,
    ) -> None:
        group_tree = getattr(group_node, "node_tree", None)
        group_name = _node_name(group_node)
        if group_tree is None:
            self.issues.append(f"Reachable node group '{group_name}' has no node tree")
            return
        if len(parent_frame.group_path) >= _MAX_GROUP_DEPTH:
            self.issues.append(
                f"Node group expansion exceeded {_MAX_GROUP_DEPTH} levels at "
                f"'{group_name}'"
            )
            return
        tree_identity = _rna_identity(group_tree)
        if tree_identity in parent_frame.tree_stack:
            cycle = " -> ".join(
                parent_frame.group_path + (group_name, _tree_name(group_tree))
            )
            self.issues.append(f"Recursive node group cycle detected: {cycle}")
            return

        nodes = _iter_nodes(group_tree)
        group_output = _find_active_node(nodes, "GROUP_OUTPUT")
        if group_output is None:
            self.issues.append(
                f"Reachable node group '{group_name}' has no Group Output node"
            )
            return
        internal_input = _matching_socket(
            getattr(group_output, "inputs", None), output_socket
        )
        if internal_input is None:
            self.issues.append(
                f"Unable to map output '{_socket_name(output_socket)}' of node "
                f"group '{group_name}' to its Group Output interface"
            )
            return

        child_frame = _GraphFrame(
            node_tree=group_tree,
            group_path=parent_frame.group_path + (group_name,),
            parent=parent_frame,
            parent_group_node=group_node,
            tree_stack=parent_frame.tree_stack + (tree_identity,),
        )
        self.node_trees[tree_identity] = group_tree
        self._record_node(child_frame, group_output, channel)
        self.walk_input(internal_input, child_frame, channel)

    def _walk_parent_group_input(
        self,
        internal_output: Any | None,
        child_frame: _GraphFrame,
        channel: str,
    ) -> None:
        parent_frame = child_frame.parent
        group_node = child_frame.parent_group_node
        if parent_frame is None or group_node is None:
            return
        parent_input = _matching_socket(
            getattr(group_node, "inputs", None), internal_output
        )
        if parent_input is None:
            self.issues.append(
                f"Unable to map Group Input output '{_socket_name(internal_output)}' "
                f"for node group '{_node_name(group_node)}'"
            )
            return
        self.walk_input(parent_input, parent_frame, channel)

    def walk_material_output(self, output: Any) -> None:
        frame = self.root_frame()
        self._record_node(frame, output, "ALL")
        for socket_name, channel in (
            ("Surface", "SURFACE"),
            ("Volume", "VOLUME"),
            ("Displacement", "DISPLACEMENT"),
        ):
            self.walk_input(_input_socket(output, socket_name), frame, channel)

    def walk_all_nodes(self) -> None:
        frame = self.root_frame()
        for node in _iter_nodes(self.root_tree):
            self._record_node(frame, node, "ALL")
            if bool(getattr(node, "mute", False)):
                for output_socket in _iter_collection(
                    getattr(node, "outputs", ()), label="node outputs"
                ):
                    self.walk_output(node, output_socket, frame, "ALL")
            elif _node_type(node) == "GROUP":
                for output_socket in _iter_collection(
                    getattr(node, "outputs", ()), label="group outputs"
                ):
                    self.walk_output(node, output_socket, frame, "ALL")
            else:
                for input_socket in _iter_collection(
                    getattr(node, "inputs", ()), label="node inputs"
                ):
                    self.walk_input(input_socket, frame, "ALL")


def _semantic_channels(
    walker: _RecursiveGraphWalker,
) -> tuple[MaterialSemanticChannel, ...]:
    surface_nodes = tuple(
        walker.nodes[node_id].node
        for node_id in walker.channel_nodes.get("SURFACE", ())
        if not bool(getattr(walker.nodes[node_id].node, "mute", False))
    )
    surface_types = {_node_type(node) for node in surface_nodes}
    channels: set[MaterialSemanticChannel] = set()
    if surface_types & _SURFACE_SHADER_TYPES:
        channels.add(MaterialSemanticChannel.SURFACE_COLOR)
    if "EMISSION" in surface_types or any(
        _principled_emission_enabled(node) for node in surface_nodes
    ):
        channels.add(MaterialSemanticChannel.SURFACE_EMISSION)
    if (
        "BSDF_TRANSPARENT" in surface_types
        or "HOLDOUT" in surface_types
        or any(_principled_alpha_enabled(node) for node in surface_nodes)
    ):
        channels.add(MaterialSemanticChannel.ALPHA)

    volume_types = {
        _node_type(walker.nodes[node_id].node)
        for node_id in walker.channel_nodes.get("VOLUME", ())
        if not bool(getattr(walker.nodes[node_id].node, "mute", False))
    }
    if volume_types - {"OUTPUT_MATERIAL", "GROUP_OUTPUT", "GROUP"}:
        channels.add(MaterialSemanticChannel.VOLUME)
    displacement_types = {
        _node_type(walker.nodes[node_id].node)
        for node_id in walker.channel_nodes.get("DISPLACEMENT", ())
        if not bool(getattr(walker.nodes[node_id].node, "mute", False))
    }
    if displacement_types - {"OUTPUT_MATERIAL", "GROUP_OUTPUT", "GROUP"}:
        channels.add(MaterialSemanticChannel.DISPLACEMENT)

    known_non_surface = {
        "EMISSION",
        "OUTPUT_MATERIAL",
        "GROUP",
        "GROUP_INPUT",
        "GROUP_OUTPUT",
        "MIX_SHADER",
        "ADD_SHADER",
        "BSDF_TRANSPARENT",
        "HOLDOUT",
    }
    if surface_nodes and not channels and not surface_types.issubset(known_non_surface):
        channels.add(MaterialSemanticChannel.SURFACE_COLOR)
    return tuple(sorted(channels, key=lambda value: value.value))


def _dependencies(
    material: Any,
    walker: _RecursiveGraphWalker,
) -> tuple[MaterialDependencyKind, ...]:
    result: set[MaterialDependencyKind] = set()
    reachable_nodes = tuple(
        item.node
        for item in walker.nodes.values()
        if not bool(getattr(item.node, "mute", False))
    )
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
    if node_types & (_GROUP_NODE_TYPES | _GROUP_INPUT_TYPES | _GROUP_OUTPUT_TYPES):
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

    if getattr(material, "animation_data", None) is not None:
        result.add(MaterialDependencyKind.TIME)
    for node_tree in walker.node_trees.values():
        if getattr(node_tree, "animation_data", None) is not None:
            result.add(MaterialDependencyKind.TIME)
    return tuple(sorted(result, key=lambda value: value.value))


def analyse_material_graph_detailed(
    material: Any,
    *,
    render_target: str = "ALL",
) -> MaterialGraphAnalysisResult:
    """Analyze a material and retain adapter-private reachable live nodes."""

    if material is None:
        raise MaterialGraphAnalysisError("material cannot be None")
    material_name = _material_name(material)
    node_tree = getattr(material, "node_tree", None)
    if node_tree is None:
        raise MaterialGraphAnalysisError(f"Material '{material_name}' has no node tree")

    target = _normalise_render_target(render_target)
    nodes = _iter_nodes(node_tree)
    output = _find_material_output(node_tree, nodes, target)
    walker = _RecursiveGraphWalker(material_name, node_tree)
    if output is None:
        walker.issues.append(
            f"Material Output for render target '{target}' was not found; "
            "semantic analysis used all nodes"
        )
        walker.walk_all_nodes()
    else:
        walker.walk_material_output(output)

    ordered_nodes = tuple(
        item
        for _, item in sorted(
            walker.nodes.items(), key=lambda pair: pair[0].casefold()
        )
    )
    node_snapshots = tuple(
        ShaderNodeSnapshot(
            node_id=item.node_id,
            node_type=_node_type(item.node),
            node_name=_node_name(item.node),
            group_path=item.frame.group_path,
        )
        for item in ordered_nodes
    )
    link_snapshots = tuple(
        walker.links[key]
        for key in sorted(
            walker.links,
            key=lambda item: tuple(component.casefold() for component in item),
        )
    )
    try:
        snapshot = MaterialGraphSnapshot(
            material_name=material_name,
            active_output_node_id=None if output is None else _node_name(output),
            reachable_nodes=node_snapshots,
            reachable_links=link_snapshots,
            semantic_channels=_semantic_channels(walker),
            dependencies=_dependencies(material, walker),
            issues=tuple(dict.fromkeys(walker.issues)),
        )
    except MaterialGraphAnalysisError:
        raise
    except Exception as exc:
        raise MaterialGraphAnalysisError(
            f"Unable to build semantic graph snapshot for material "
            f"'{material_name}'"
        ) from exc

    logger.debug(
        "Analyzed recursive shader graph '%s' target=%s: nodes=%d "
        "channels=%s dependencies=%s issues=%s",
        material_name,
        target,
        len(snapshot.reachable_nodes),
        tuple(value.value for value in snapshot.semantic_channels),
        tuple(value.value for value in snapshot.dependencies),
        snapshot.issues,
    )
    return MaterialGraphAnalysisResult(
        snapshot=snapshot,
        reachable_nodes=tuple(item.node for item in ordered_nodes),
    )


def analyse_material_graph(
    material: Any,
    *,
    render_target: str = "ALL",
) -> MaterialGraphSnapshot:
    """Analyze reachable nodes, recursively expanding used Shader Node Groups."""

    return analyse_material_graph_detailed(
        material,
        render_target=render_target,
    ).snapshot


__all__ = [
    "MaterialGraphAnalysisError",
    "MaterialGraphAnalysisResult",
    "analyse_material_graph",
    "analyse_material_graph_detailed",
]
