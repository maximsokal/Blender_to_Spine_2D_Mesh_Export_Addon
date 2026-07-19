"""Recursive reachable-node traversal for Blender shader graphs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ..domain.baking.graph import ShaderLinkSnapshot
from .shader_graph_error import MaterialGraphAnalysisError
from .shader_graph_rna import (
    find_active_node,
    input_socket,
    is_temporary_node,
    iter_collection,
    iter_links,
    iter_nodes,
    matching_socket,
    node_name,
    node_type,
    rna_identity,
    same_socket,
    socket_identifier,
    socket_index,
    socket_name,
    tree_name,
)


GROUP_NODE_TYPES = frozenset({"GROUP"})
GROUP_INPUT_TYPES = frozenset({"GROUP_INPUT"})
GROUP_OUTPUT_TYPES = frozenset({"GROUP_OUTPUT"})
MAX_GROUP_DEPTH = 64

LinkKey = tuple[str, str, str, str]


@dataclass(frozen=True, slots=True)
class ShaderGraphFrame:
    node_tree: Any
    group_path: tuple[str, ...]
    parent: "ShaderGraphFrame | None" = None
    parent_group_node: Any | None = None
    tree_stack: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class ReachableShaderNode:
    node: Any
    frame: ShaderGraphFrame

    @property
    def node_id(self) -> str:
        return "::".join(self.frame.group_path + (node_name(self.node),))


@dataclass(frozen=True, slots=True)
class ShaderGraphTraversalResult:
    """Frozen reachable graph state consumed by semantics and snapshot assembly."""

    nodes: Mapping[str, ReachableShaderNode]
    links: Mapping[LinkKey, ShaderLinkSnapshot]
    channel_nodes: Mapping[str, tuple[str, ...]]
    node_trees: tuple[Any, ...]
    issues: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.nodes, Mapping):
            raise TypeError("nodes must be a mapping")
        if not isinstance(self.links, Mapping):
            raise TypeError("links must be a mapping")
        if not isinstance(self.channel_nodes, Mapping):
            raise TypeError("channel_nodes must be a mapping")
        if not isinstance(self.node_trees, tuple):
            raise TypeError("node_trees must be tuple")
        if not isinstance(self.issues, tuple):
            raise TypeError("issues must be tuple")


class RecursiveShaderGraphWalker:
    """Traverse only sockets that contribute to the effective Material Output."""

    def __init__(self, material_name: str, root_tree: Any) -> None:
        self.material_name = material_name
        self.root_tree = root_tree
        self.nodes: dict[str, ReachableShaderNode] = {}
        self.links: dict[LinkKey, ShaderLinkSnapshot] = {}
        self.channel_nodes: dict[str, set[str]] = {
            "SURFACE": set(),
            "VOLUME": set(),
            "DISPLACEMENT": set(),
            "ALL": set(),
        }
        root_identity = rna_identity(root_tree)
        self.node_trees: dict[int, Any] = {root_identity: root_tree}
        self.issues: list[str] = []
        self._visited_outputs: set[
            tuple[int, tuple[str, ...], str, str, str]
        ] = set()
        self._visited_inputs: set[
            tuple[int, tuple[str, ...], str, str, str]
        ] = set()

    def root_frame(self) -> ShaderGraphFrame:
        identity = rna_identity(self.root_tree)
        return ShaderGraphFrame(
            node_tree=self.root_tree,
            group_path=(),
            tree_stack=(identity,),
        )

    def _node_id(self, frame: ShaderGraphFrame, node: Any) -> str:
        return "::".join(frame.group_path + (node_name(node),))

    def _record_node(
        self,
        frame: ShaderGraphFrame,
        node: Any,
        channel: str,
    ) -> str:
        node_id = self._node_id(frame, node)
        existing = self.nodes.get(node_id)
        if (
            existing is not None
            and rna_identity(existing.node) != rna_identity(node)
        ):
            raise MaterialGraphAnalysisError(
                f"Duplicate reachable node ID '{node_id}' in material "
                f"'{self.material_name}'"
            )
        self.nodes.setdefault(
            node_id,
            ReachableShaderNode(node=node, frame=frame),
        )
        self.channel_nodes.setdefault(channel, set()).add(node_id)
        self.channel_nodes["ALL"].add(node_id)
        return node_id

    def _record_link(
        self,
        frame: ShaderGraphFrame,
        link: Any,
        channel: str,
    ) -> None:
        from_node = getattr(link, "from_node", None)
        to_node = getattr(link, "to_node", None)
        if from_node is None or to_node is None:
            return
        from_id = self._record_node(frame, from_node, channel)
        to_id = self._record_node(frame, to_node, channel)
        snapshot = ShaderLinkSnapshot(
            from_node_id=from_id,
            from_socket=socket_name(getattr(link, "from_socket", None)),
            to_node_id=to_id,
            to_socket=socket_name(getattr(link, "to_socket", None)),
        )
        key = (
            snapshot.from_node_id,
            snapshot.from_socket,
            snapshot.to_node_id,
            snapshot.to_socket,
        )
        self.links.setdefault(key, snapshot)

    def _incoming_links(
        self,
        socket: Any,
        frame: ShaderGraphFrame,
    ) -> tuple[Any, ...]:
        direct = iter_collection(
            getattr(socket, "links", ()),
            label="socket links",
        )
        if direct:
            return tuple(
                link
                for link in direct
                if not is_temporary_node(getattr(link, "from_node", None))
                and not is_temporary_node(getattr(link, "to_node", None))
            )
        node = getattr(socket, "node", None)
        identifier = socket_identifier(socket)
        name = socket_name(socket)
        return tuple(
            link
            for link in iter_links(frame.node_tree)
            if getattr(link, "to_node", None) is node
            and (
                getattr(link, "to_socket", None) is socket
                or (
                    identifier
                    and socket_identifier(getattr(link, "to_socket", None))
                    == identifier
                )
                or socket_name(getattr(link, "to_socket", None)) == name
            )
        )

    def walk_input(
        self,
        socket: Any | None,
        frame: ShaderGraphFrame,
        channel: str,
    ) -> None:
        if socket is None:
            return
        node = getattr(socket, "node", None)
        resolved_node_name = node_name(node) if node is not None else "SocketOwner"
        key = (
            rna_identity(frame.node_tree),
            frame.group_path,
            resolved_node_name,
            socket_identifier(socket) or socket_name(socket),
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
        frame: ShaderGraphFrame,
        channel: str,
    ) -> None:
        if node is None:
            return
        node_id = self._record_node(frame, node, channel)
        key = (
            rna_identity(frame.node_tree),
            frame.group_path,
            node_id,
            socket_identifier(output_socket) or socket_name(output_socket),
            channel,
        )
        if key in self._visited_outputs:
            return
        self._visited_outputs.add(key)

        if bool(getattr(node, "mute", False)):
            self._walk_muted_output(node, output_socket, frame, channel)
            return

        resolved_node_type = node_type(node)
        if resolved_node_type in GROUP_NODE_TYPES:
            self._walk_group_output(node, output_socket, frame, channel)
            return
        if resolved_node_type in GROUP_INPUT_TYPES:
            self._walk_parent_group_input(output_socket, frame, channel)
            return

        for input_value in iter_collection(
            getattr(node, "inputs", ()),
            label="node inputs",
        ):
            self.walk_input(input_value, frame, channel)

    def _walk_muted_output(
        self,
        node: Any,
        output_socket: Any | None,
        frame: ShaderGraphFrame,
        channel: str,
    ) -> None:
        """Follow Blender's mute bypass mapping instead of its implementation."""

        internal_links = iter_collection(
            getattr(node, "internal_links", ()),
            label="muted node internal links",
        )
        mapped_inputs: list[Any] = []
        for link in internal_links:
            from_socket = getattr(link, "from_socket", None)
            to_socket = getattr(link, "to_socket", None)
            if same_socket(to_socket, output_socket):
                mapped_inputs.append(from_socket)
            elif same_socket(from_socket, output_socket):
                mapped_inputs.append(to_socket)

        valid_inputs = tuple(
            socket
            for socket in mapped_inputs
            if socket is not None
            and socket_index(getattr(node, "inputs", None), socket) is not None
        )
        if valid_inputs:
            seen_inputs: set[int] = set()
            for input_value in valid_inputs:
                identity = rna_identity(input_value)
                if identity in seen_inputs:
                    continue
                seen_inputs.add(identity)
                self.walk_input(input_value, frame, channel)
            return

        self.issues.append(
            f"Muted node '{node_name(node)}' has no unambiguous internal bypass "
            f"for output '{socket_name(output_socket)}'; all inputs were analyzed "
            "conservatively"
        )
        for input_value in iter_collection(
            getattr(node, "inputs", ()),
            label="node inputs",
        ):
            self.walk_input(input_value, frame, channel)

    def _walk_group_output(
        self,
        group_node: Any,
        output_socket: Any | None,
        parent_frame: ShaderGraphFrame,
        channel: str,
    ) -> None:
        group_tree = getattr(group_node, "node_tree", None)
        group_name = node_name(group_node)
        if group_tree is None:
            self.issues.append(
                f"Reachable node group '{group_name}' has no node tree"
            )
            return
        if len(parent_frame.group_path) >= MAX_GROUP_DEPTH:
            self.issues.append(
                f"Node group expansion exceeded {MAX_GROUP_DEPTH} levels at "
                f"'{group_name}'"
            )
            return
        tree_identity = rna_identity(group_tree)
        if tree_identity in parent_frame.tree_stack:
            cycle = " -> ".join(
                parent_frame.group_path + (group_name, tree_name(group_tree))
            )
            self.issues.append(f"Recursive node group cycle detected: {cycle}")
            return

        nodes = iter_nodes(group_tree)
        group_output = find_active_node(nodes, "GROUP_OUTPUT")
        if group_output is None:
            self.issues.append(
                f"Reachable node group '{group_name}' has no Group Output node"
            )
            return
        internal_input = matching_socket(
            getattr(group_output, "inputs", None),
            output_socket,
        )
        if internal_input is None:
            self.issues.append(
                f"Unable to map output '{socket_name(output_socket)}' of node "
                f"group '{group_name}' to its Group Output interface"
            )
            return

        child_frame = ShaderGraphFrame(
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
        child_frame: ShaderGraphFrame,
        channel: str,
    ) -> None:
        parent_frame = child_frame.parent
        group_node = child_frame.parent_group_node
        if parent_frame is None or group_node is None:
            return
        parent_input = matching_socket(
            getattr(group_node, "inputs", None),
            internal_output,
        )
        if parent_input is None:
            self.issues.append(
                f"Unable to map Group Input output '{socket_name(internal_output)}' "
                f"for node group '{node_name(group_node)}'"
            )
            return
        self.walk_input(parent_input, parent_frame, channel)

    def walk_material_output(self, output: Any) -> None:
        frame = self.root_frame()
        self._record_node(frame, output, "ALL")
        for socket_name_value, channel in (
            ("Surface", "SURFACE"),
            ("Volume", "VOLUME"),
            ("Displacement", "DISPLACEMENT"),
        ):
            self.walk_input(
                input_socket(output, socket_name_value),
                frame,
                channel,
            )

    def walk_all_nodes(self) -> None:
        frame = self.root_frame()
        for node in iter_nodes(self.root_tree):
            self._record_node(frame, node, "ALL")
            if bool(getattr(node, "mute", False)):
                for output_socket in iter_collection(
                    getattr(node, "outputs", ()),
                    label="node outputs",
                ):
                    self.walk_output(node, output_socket, frame, "ALL")
            elif node_type(node) == "GROUP":
                for output_socket in iter_collection(
                    getattr(node, "outputs", ()),
                    label="group outputs",
                ):
                    self.walk_output(node, output_socket, frame, "ALL")
            else:
                for input_value in iter_collection(
                    getattr(node, "inputs", ()),
                    label="node inputs",
                ):
                    self.walk_input(input_value, frame, "ALL")

    def build_result(self) -> ShaderGraphTraversalResult:
        """Freeze the traversal state without reordering snapshot-owned data."""

        channel_nodes = {
            channel: tuple(sorted(node_ids, key=str.casefold))
            for channel, node_ids in self.channel_nodes.items()
        }
        return ShaderGraphTraversalResult(
            nodes=MappingProxyType(dict(self.nodes)),
            links=MappingProxyType(dict(self.links)),
            channel_nodes=MappingProxyType(channel_nodes),
            node_trees=tuple(self.node_trees.values()),
            issues=tuple(self.issues),
        )


__all__ = [
    "GROUP_INPUT_TYPES",
    "GROUP_NODE_TYPES",
    "GROUP_OUTPUT_TYPES",
    "MAX_GROUP_DEPTH",
    "ReachableShaderNode",
    "RecursiveShaderGraphWalker",
    "ShaderGraphFrame",
    "ShaderGraphTraversalResult",
]
