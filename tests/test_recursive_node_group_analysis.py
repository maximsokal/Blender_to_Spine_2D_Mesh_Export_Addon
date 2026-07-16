from __future__ import annotations

from dataclasses import dataclass

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.shader_graph_analyzer import (
    analyse_material_graph,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    MaterialDependencyKind,
    MaterialSemanticChannel,
)


class _SocketCollection(list):
    def get(self, name):
        for socket in self:
            if socket.name == name:
                return socket
        return None


class _Socket:
    def __init__(
        self,
        node,
        name: str,
        *,
        identifier: str | None = None,
        is_output: bool = False,
        default_value=0.0,
    ):
        self.node = node
        self.name = name
        self.identifier = identifier or name
        self.is_output = is_output
        self.default_value = default_value
        self.links = []

    @property
    def is_linked(self):
        return bool(self.links)


class _Node:
    def __init__(self, name: str, node_type: str, *, active=False, node_tree=None):
        self.name = name
        self.type = node_type
        self.is_active_output = active
        self.node_tree = node_tree
        self.inputs = _SocketCollection()
        self.outputs = _SocketCollection()

    def input(self, name: str, *, identifier=None, default_value=0.0):
        socket = _Socket(
            self,
            name,
            identifier=identifier,
            is_output=False,
            default_value=default_value,
        )
        self.inputs.append(socket)
        return socket

    def output(self, name: str, *, identifier=None):
        socket = _Socket(
            self,
            name,
            identifier=identifier,
            is_output=True,
        )
        self.outputs.append(socket)
        return socket


@dataclass
class _Link:
    from_node: _Node
    from_socket: _Socket
    to_node: _Node
    to_socket: _Socket


class _NodeTree:
    def __init__(self, name: str):
        self.name = name
        self.nodes = []
        self.links = []
        self.animation_data = None

    def add(self, node: _Node):
        self.nodes.append(node)
        return node

    def link(self, from_socket: _Socket, to_socket: _Socket):
        link = _Link(from_socket.node, from_socket, to_socket.node, to_socket)
        self.links.append(link)
        from_socket.links.append(link)
        to_socket.links.append(link)
        return link


class _Material:
    def __init__(self, name: str, node_tree: _NodeTree):
        self.name = name
        self.node_tree = node_tree
        self.animation_data = None


def _material_output(tree: _NodeTree):
    output = tree.add(_Node("Material Output", "OUTPUT_MATERIAL", active=True))
    output.input("Surface", identifier="Surface")
    output.input("Volume", identifier="Volume")
    output.input("Displacement", identifier="Displacement")
    return output


def _group_output(tree: _NodeTree, socket_name="Shader", identifier="Shader"):
    output = tree.add(_Node("Group Output", "GROUP_OUTPUT", active=True))
    output.input(socket_name, identifier=identifier)
    return output


def _group_input(tree: _NodeTree, socket_name="Input", identifier="Input"):
    node = tree.add(_Node("Group Input", "GROUP_INPUT"))
    node.output(socket_name, identifier=identifier)
    return node


def _group_node(tree: _NodeTree, name: str, nested_tree: _NodeTree):
    return tree.add(_Node(name, "GROUP", node_tree=nested_tree))


def test_nested_group_propagates_camera_dependencies_and_group_path():
    inner_tree = _NodeTree("InnerTree")
    inner_output = _group_output(inner_tree)
    layer_weight = inner_tree.add(_Node("Layer Weight", "LAYER_WEIGHT"))
    facing = layer_weight.output("Facing")
    emission = inner_tree.add(_Node("Emission", "EMISSION"))
    emission_color = emission.input("Color", default_value=(0.0, 0.0, 0.0, 1.0))
    emission_shader = emission.output("Emission")
    inner_tree.link(facing, emission_color)
    inner_tree.link(emission_shader, inner_output.inputs[0])

    outer_tree = _NodeTree("OuterTree")
    outer_output = _group_output(outer_tree)
    inner_group = _group_node(outer_tree, "InnerGroup", inner_tree)
    inner_group_out = inner_group.output("Shader", identifier="Shader")
    outer_tree.link(inner_group_out, outer_output.inputs[0])

    root = _NodeTree("MaterialTree")
    material_output = _material_output(root)
    outer_group = _group_node(root, "OuterGroup", outer_tree)
    outer_group_out = outer_group.output("Shader", identifier="Shader")
    root.link(outer_group_out, material_output.inputs.get("Surface"))

    snapshot = analyse_material_graph(_Material("NestedCamera", root))

    assert MaterialSemanticChannel.SURFACE_EMISSION in snapshot.semantic_channels
    assert MaterialDependencyKind.NODE_GROUP in snapshot.dependencies
    assert MaterialDependencyKind.CAMERA in snapshot.dependencies
    assert MaterialDependencyKind.VIEW in snapshot.dependencies
    assert MaterialDependencyKind.REFLECTION in snapshot.dependencies
    layer_snapshot = next(
        node for node in snapshot.reachable_nodes if node.node_name == "Layer Weight"
    )
    assert layer_snapshot.group_path == ("OuterGroup", "InnerGroup")
    assert layer_snapshot.node_id == "OuterGroup::InnerGroup::Layer Weight"


def test_unused_group_input_does_not_leak_parent_camera_dependency():
    group_tree = _NodeTree("PreciseGroup")
    group_output = _group_output(group_tree)
    _group_input(group_tree, "UnusedView", "UnusedView")
    diffuse = group_tree.add(_Node("Diffuse", "BSDF_DIFFUSE"))
    diffuse.input("Color", default_value=(0.8, 0.2, 0.1, 1.0))
    diffuse_out = diffuse.output("BSDF")
    group_tree.link(diffuse_out, group_output.inputs[0])

    root = _NodeTree("MaterialTree")
    material_output = _material_output(root)
    group = _group_node(root, "PreciseGroupInstance", group_tree)
    group.input("UnusedView", identifier="UnusedView")
    group_out = group.output("Shader", identifier="Shader")
    fresnel = root.add(_Node("Unused Fresnel", "FRESNEL"))
    fresnel_out = fresnel.output("Fac")
    root.link(fresnel_out, group.inputs[0])
    root.link(group_out, material_output.inputs.get("Surface"))

    snapshot = analyse_material_graph(_Material("Precise", root))

    assert MaterialSemanticChannel.SURFACE_COLOR in snapshot.semantic_channels
    assert MaterialDependencyKind.NODE_GROUP in snapshot.dependencies
    assert MaterialDependencyKind.CAMERA not in snapshot.dependencies
    assert MaterialDependencyKind.VIEW not in snapshot.dependencies
    assert all(node.node_name != "Unused Fresnel" for node in snapshot.reachable_nodes)


def test_group_input_maps_only_the_used_parent_socket():
    group_tree = _NodeTree("ColorGroup")
    group_output = _group_output(group_tree, "Color", "Color")
    group_input = _group_input(group_tree, "Color", "Color")
    group_tree.link(group_input.outputs[0], group_output.inputs[0])

    root = _NodeTree("MaterialTree")
    material_output = _material_output(root)
    group = _group_node(root, "ColorGroupInstance", group_tree)
    group.input("Color", identifier="Color")
    group_out = group.output("Color", identifier="Color")
    emission = root.add(_Node("Emission", "EMISSION"))
    emission.input("Color")
    emission_out = emission.output("Emission")
    image = root.add(_Node("Image", "TEX_IMAGE"))
    image_out = image.output("Color")
    root.link(image_out, group.inputs[0])
    root.link(group_out, emission.inputs[0])
    root.link(emission_out, material_output.inputs.get("Surface"))

    snapshot = analyse_material_graph(_Material("GroupInput", root))

    assert MaterialDependencyKind.IMAGE in snapshot.dependencies
    assert any(node.node_name == "Image" for node in snapshot.reachable_nodes)
    assert any(node.node_name == "Group Input" for node in snapshot.reachable_nodes)


def test_nested_volume_channel_is_discovered():
    volume_tree = _NodeTree("VolumeTree")
    volume_output = _group_output(volume_tree, "Volume", "Volume")
    volume = volume_tree.add(_Node("Principled Volume", "PRINCIPLED_VOLUME"))
    volume.input("Density", default_value=1.0)
    volume_out = volume.output("Volume")
    volume_tree.link(volume_out, volume_output.inputs[0])

    root = _NodeTree("MaterialTree")
    material_output = _material_output(root)
    group = _group_node(root, "VolumeGroup", volume_tree)
    group_out = group.output("Volume", identifier="Volume")
    root.link(group_out, material_output.inputs.get("Volume"))

    snapshot = analyse_material_graph(_Material("NestedVolume", root))

    assert MaterialSemanticChannel.VOLUME in snapshot.semantic_channels
    assert MaterialDependencyKind.NODE_GROUP in snapshot.dependencies


def test_nested_tree_animation_propagates_time_dependency():
    group_tree = _NodeTree("AnimatedGroup")
    group_tree.animation_data = object()
    group_output = _group_output(group_tree)
    emission = group_tree.add(_Node("Emission", "EMISSION"))
    emission.input("Color", default_value=(1.0, 0.0, 0.0, 1.0))
    emission_out = emission.output("Emission")
    group_tree.link(emission_out, group_output.inputs[0])

    root = _NodeTree("MaterialTree")
    output = _material_output(root)
    group = _group_node(root, "AnimatedGroupInstance", group_tree)
    group_out = group.output("Shader", identifier="Shader")
    root.link(group_out, output.inputs.get("Surface"))

    snapshot = analyse_material_graph(_Material("AnimatedNested", root))

    assert MaterialDependencyKind.TIME in snapshot.dependencies


def test_recursive_group_cycle_is_reported_without_infinite_recursion():
    recursive_tree = _NodeTree("RecursiveTree")
    group_output = _group_output(recursive_tree)
    recursive_group = _group_node(recursive_tree, "Self", recursive_tree)
    recursive_out = recursive_group.output("Shader", identifier="Shader")
    recursive_tree.link(recursive_out, group_output.inputs[0])

    root = _NodeTree("MaterialTree")
    output = _material_output(root)
    group = _group_node(root, "Recursive", recursive_tree)
    group_out = group.output("Shader", identifier="Shader")
    root.link(group_out, output.inputs.get("Surface"))

    snapshot = analyse_material_graph(_Material("RecursiveMaterial", root))

    assert MaterialDependencyKind.NODE_GROUP in snapshot.dependencies
    assert any("Recursive node group cycle detected" in issue for issue in snapshot.issues)
    assert len(snapshot.reachable_nodes) < 10
