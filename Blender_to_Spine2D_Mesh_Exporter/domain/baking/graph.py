"""Immutable semantic snapshots of reachable Blender shader graphs.

The domain types in this module contain no ``bpy`` references. Blender adapters may
inspect node trees and translate them into these values, while strategy selection and
tests remain executable in ordinary Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class MaterialSemanticChannel(str, Enum):
    """Material outputs that may require independent bake passes."""

    SURFACE_COLOR = "SURFACE_COLOR"
    SURFACE_EMISSION = "SURFACE_EMISSION"
    ALPHA = "ALPHA"
    VOLUME = "VOLUME"
    DISPLACEMENT = "DISPLACEMENT"


class MaterialDependencyKind(str, Enum):
    """External context needed to evaluate a reachable shader graph."""

    IMAGE = "IMAGE"
    TIME = "TIME"
    OBJECT = "OBJECT"
    WORLD = "WORLD"
    CAMERA = "CAMERA"
    VIEW = "VIEW"
    LIGHTING = "LIGHTING"
    OCCLUSION = "OCCLUSION"
    SCENE_OBJECTS = "SCENE_OBJECTS"
    REFLECTION = "REFLECTION"
    TRANSMISSION = "TRANSMISSION"
    GEOMETRY = "GEOMETRY"
    NODE_GROUP = "NODE_GROUP"


@dataclass(frozen=True, slots=True)
class ShaderNodeSnapshot:
    node_id: str
    node_type: str
    node_name: str
    group_path: Tuple[str, ...] = ()
    muted: bool = False
    # Blender exposes Texture Coordinate "From Instancer" as a node property, not
    # as an output socket. Keeping it in the immutable snapshot lets the pure
    # capability policy distinguish ordinary UV/Generated coordinates from
    # instance-context-dependent coordinates without importing bpy into domain code.
    from_instancer: bool = False

    def __post_init__(self) -> None:
        for field_name in ("node_id", "node_type", "node_name"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(self.group_path, tuple) or not all(
            isinstance(value, str) and value.strip() for value in self.group_path
        ):
            raise TypeError("group_path must be a tuple of non-empty strings")
        if not isinstance(self.muted, bool):
            raise TypeError("muted must be bool")
        if not isinstance(self.from_instancer, bool):
            raise TypeError("from_instancer must be bool")


@dataclass(frozen=True, slots=True)
class ShaderLinkSnapshot:
    from_node_id: str
    from_socket: str
    to_node_id: str
    to_socket: str

    def __post_init__(self) -> None:
        for field_name in (
            "from_node_id",
            "from_socket",
            "to_node_id",
            "to_socket",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class MaterialGraphSnapshot:
    material_name: str
    active_output_node_id: str | None
    reachable_nodes: Tuple[ShaderNodeSnapshot, ...]
    reachable_links: Tuple[ShaderLinkSnapshot, ...]
    semantic_channels: Tuple[MaterialSemanticChannel, ...]
    dependencies: Tuple[MaterialDependencyKind, ...]
    issues: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.material_name, str) or not self.material_name.strip():
            raise ValueError("material_name must be a non-empty string")
        if self.active_output_node_id is not None and (
            not isinstance(self.active_output_node_id, str)
            or not self.active_output_node_id.strip()
        ):
            raise ValueError("active_output_node_id must be a non-empty string or None")
        for field_name in (
            "reachable_nodes",
            "reachable_links",
            "semantic_channels",
            "dependencies",
            "issues",
        ):
            if not isinstance(getattr(self, field_name), tuple):
                raise TypeError(f"{field_name} must be tuple")

        node_ids = tuple(node.node_id for node in self.reachable_nodes)
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("reachable_nodes contains duplicate node_id values")
        known_nodes = set(node_ids)
        for link in self.reachable_links:
            if link.from_node_id not in known_nodes or link.to_node_id not in known_nodes:
                raise ValueError("reachable_links references a node outside reachable_nodes")
        if (
            self.active_output_node_id is not None
            and self.active_output_node_id not in known_nodes
        ):
            raise ValueError("active_output_node_id is not present in reachable_nodes")
        if len(self.semantic_channels) != len(set(self.semantic_channels)):
            raise ValueError("semantic_channels cannot contain duplicates")
        if len(self.dependencies) != len(set(self.dependencies)):
            raise ValueError("dependencies cannot contain duplicates")
        if not all(
            isinstance(value, MaterialSemanticChannel)
            for value in self.semantic_channels
        ):
            raise TypeError("semantic_channels must contain MaterialSemanticChannel")
        if not all(
            isinstance(value, MaterialDependencyKind) for value in self.dependencies
        ):
            raise TypeError("dependencies must contain MaterialDependencyKind")
        if not all(isinstance(value, str) and value.strip() for value in self.issues):
            raise TypeError("issues must contain non-empty strings")

    def has_channel(self, channel: MaterialSemanticChannel) -> bool:
        if not isinstance(channel, MaterialSemanticChannel):
            raise TypeError("channel must be MaterialSemanticChannel")
        return channel in self.semantic_channels

    def has_dependency(self, dependency: MaterialDependencyKind) -> bool:
        if not isinstance(dependency, MaterialDependencyKind):
            raise TypeError("dependency must be MaterialDependencyKind")
        return dependency in self.dependencies
