from __future__ import annotations

from dataclasses import dataclass
import sys

import pytest
from types import SimpleNamespace

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.material_slot_analysis import (
    analyse_material_slot,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.render_engine_contract import (
    render_engine_contract,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.shader_graph_analyzer import (
    analyse_material_graph,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import MaterialKind
from Blender_to_Spine2D_Mesh_Exporter.domain.baking.graph import (
    MaterialDependencyKind,
    MaterialSemanticChannel,
)


class SocketCollection(list):
    def get(self, name):
        for socket in self:
            if socket.name == name:
                return socket
        return None


class Socket:
    def __init__(
        self,
        node,
        name,
        *,
        identifier=None,
        is_output=False,
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


class Node:
    _next_pointer = 1000

    def __init__(
        self,
        name,
        node_type,
        *,
        active=False,
        target="ALL",
        node_tree=None,
        mute=False,
        image=None,
    ):
        self.name = name
        self.type = node_type
        self.is_active_output = active
        self.target = target
        self.node_tree = node_tree
        self.mute = mute
        self.image = image
        self.inputs = SocketCollection()
        self.outputs = SocketCollection()
        self.internal_links = []
        self._pointer = Node._next_pointer
        Node._next_pointer += 1

    def as_pointer(self):
        return self._pointer

    def input(self, name, *, identifier=None, default_value=0.0):
        socket = Socket(
            self,
            name,
            identifier=identifier,
            is_output=False,
            default_value=default_value,
        )
        self.inputs.append(socket)
        return socket

    def output(self, name, *, identifier=None):
        socket = Socket(
            self,
            name,
            identifier=identifier,
            is_output=True,
        )
        self.outputs.append(socket)
        return socket


@dataclass
class Link:
    from_node: Node
    from_socket: Socket
    to_node: Node
    to_socket: Socket


class NodeTree:
    _next_pointer = 5000

    def __init__(self, name):
        self.name = name
        self.nodes = []
        self.links = []
        self.animation_data = None
        self._outputs = {}
        self._pointer = NodeTree._next_pointer
        NodeTree._next_pointer += 1

    def as_pointer(self):
        return self._pointer

    def add(self, node):
        self.nodes.append(node)
        return node

    def link(self, from_socket, to_socket):
        link = Link(from_socket.node, from_socket, to_socket.node, to_socket)
        self.links.append(link)
        from_socket.links.append(link)
        to_socket.links.append(link)
        return link

    def bypass(self, input_socket, output_socket):
        link = Link(input_socket.node, input_socket, output_socket.node, output_socket)
        input_socket.node.internal_links.append(link)
        return link

    def set_output(self, target, node):
        self._outputs[target] = node

    def get_output_node(self, target):
        return self._outputs.get(target) or self._outputs.get("ALL")


class Material:
    def __init__(self, name, tree):
        self.name = name
        self.node_tree = tree
        self.use_nodes = True
        self.animation_data = None


def material_output(tree, name="Material Output", *, target="ALL", active=True):
    node = tree.add(Node(name, "OUTPUT_MATERIAL", active=active, target=target))
    node.input("Surface", identifier="Surface")
    node.input("Volume", identifier="Volume")
    node.input("Displacement", identifier="Displacement")
    return node


def group_output(tree, name="Shader", identifier="Shader"):
    node = tree.add(Node("Group Output", "GROUP_OUTPUT", active=True))
    node.input(name, identifier=identifier)
    return node


def make_diffuse(tree, name="Diffuse"):
    node = tree.add(Node(name, "BSDF_DIFFUSE"))
    node.input("Color", default_value=(0.8, 0.2, 0.1, 1.0))
    node.output("BSDF")
    return node


def make_layer_weight(tree, name="Layer Weight"):
    node = tree.add(Node(name, "LAYER_WEIGHT"))
    node.input("Blend", default_value=0.5)
    node.output("Facing")
    return node


def make_group_tree(name="Reusable"):
    tree = NodeTree(name)
    output = group_output(tree)
    image = tree.add(
        Node(
            "Nested Image",
            "TEX_IMAGE",
            image=SimpleNamespace(
                name="NestedTexture",
                source="FILE",
                filepath="//nested.png",
                filepath_raw="//nested.png",
                frame_duration=1,
            ),
        )
    )
    image.output("Color")
    emission = tree.add(Node("Nested Emission", "EMISSION"))
    emission.input("Color")
    emission.output("Emission")
    tree.link(image.outputs[0], emission.inputs[0])
    tree.link(emission.outputs[0], output.inputs[0])
    return tree


def test_renderer_specific_material_output_is_selected():
    tree = NodeTree("RendererOutputs")
    cycles_output = material_output(tree, "Cycles Output", target="CYCLES")
    eevee_output = material_output(tree, "Eevee Output", target="EEVEE")
    tree.set_output("CYCLES", cycles_output)
    tree.set_output("EEVEE", eevee_output)

    layer = make_layer_weight(tree, "Cycles Layer Weight")
    emission = tree.add(Node("Cycles Emission", "EMISSION"))
    emission.input("Color")
    emission.output("Emission")
    tree.link(layer.outputs[0], emission.inputs[0])
    tree.link(emission.outputs[0], cycles_output.inputs.get("Surface"))

    diffuse = make_diffuse(tree, "Eevee Diffuse")
    tree.link(diffuse.outputs[0], eevee_output.inputs.get("Surface"))

    material = Material("RendererSpecific", tree)
    cycles = analyse_material_graph(material, render_target="CYCLES")
    eevee = analyse_material_graph(material, render_target="BLENDER_EEVEE")

    assert cycles.active_output_node_id == "Cycles Output"
    assert MaterialDependencyKind.CAMERA in cycles.dependencies
    assert any(
        node.node_name == "Cycles Layer Weight" for node in cycles.reachable_nodes
    )
    assert eevee.active_output_node_id == "Eevee Output"
    assert MaterialDependencyKind.CAMERA not in eevee.dependencies
    assert all(
        node.node_name != "Cycles Layer Weight" for node in eevee.reachable_nodes
    )


def test_muted_node_uses_only_internal_bypass_input():
    tree = NodeTree("MutedMix")
    output = material_output(tree)
    tree.set_output("ALL", output)
    diffuse = make_diffuse(tree)
    layer = make_layer_weight(tree)
    camera_emission = tree.add(Node("Camera Emission", "EMISSION"))
    camera_emission.input("Color")
    camera_emission.output("Emission")
    tree.link(layer.outputs[0], camera_emission.inputs[0])

    muted = tree.add(Node("Muted Mix", "MIX_SHADER", mute=True))
    first = muted.input("Shader")
    second = muted.input("Shader_001")
    shader_out = muted.output("Shader")
    tree.link(diffuse.outputs[0], first)
    tree.link(camera_emission.outputs[0], second)
    tree.bypass(first, shader_out)
    tree.link(shader_out, output.inputs.get("Surface"))

    graph = analyse_material_graph(Material("Muted", tree))

    assert MaterialDependencyKind.CAMERA not in graph.dependencies
    assert any(node.node_name == "Diffuse" for node in graph.reachable_nodes)
    assert all(node.node_name != "Layer Weight" for node in graph.reachable_nodes)
    assert not graph.issues


def test_muted_group_bypasses_nested_camera_graph():
    nested = NodeTree("NestedCamera")
    nested_output = group_output(nested)
    layer = make_layer_weight(nested, "Nested Layer")
    emission = nested.add(Node("Nested Emission", "EMISSION"))
    emission.input("Color")
    emission.output("Emission")
    nested.link(layer.outputs[0], emission.inputs[0])
    nested.link(emission.outputs[0], nested_output.inputs[0])

    root = NodeTree("Root")
    output = material_output(root)
    root.set_output("ALL", output)
    diffuse = make_diffuse(root)
    group = root.add(Node("Muted Group", "GROUP", node_tree=nested, mute=True))
    group_in = group.input("Shader", identifier="Shader")
    group_out = group.output("Shader", identifier="Shader")
    root.link(diffuse.outputs[0], group_in)
    root.bypass(group_in, group_out)
    root.link(group_out, output.inputs.get("Surface"))

    graph = analyse_material_graph(Material("MutedGroupMaterial", root))

    assert MaterialDependencyKind.CAMERA not in graph.dependencies
    assert MaterialDependencyKind.NODE_GROUP not in graph.dependencies
    assert all(node.node_name != "Nested Layer" for node in graph.reachable_nodes)


def test_muted_camera_node_does_not_contribute_its_own_dependency():
    root = NodeTree("MutedCameraNode")
    output = material_output(root)
    root.set_output("ALL", output)
    layer = make_layer_weight(root, "Muted Layer Weight")
    layer.mute = True
    root.link(layer.outputs[0], output.inputs.get("Surface"))
    root.bypass(layer.inputs[0], layer.outputs[0])

    graph = analyse_material_graph(Material("MutedCameraNodeMaterial", root))

    assert MaterialDependencyKind.CAMERA not in graph.dependencies
    assert MaterialDependencyKind.VIEW not in graph.dependencies
    assert any(
        node.node_name == "Muted Layer Weight" for node in graph.reachable_nodes
    )


def test_reused_group_tree_has_instance_qualified_nodes():
    nested = make_group_tree()
    root = NodeTree("Root")
    output = material_output(root)
    root.set_output("ALL", output)
    first = root.add(Node("First Instance", "GROUP", node_tree=nested))
    second = root.add(Node("Second Instance", "GROUP", node_tree=nested))
    first.output("Shader", identifier="Shader")
    second.output("Shader", identifier="Shader")
    add = root.add(Node("Add", "ADD_SHADER"))
    add.input("Shader")
    add.input("Shader_001")
    add.output("Shader")
    root.link(first.outputs[0], add.inputs[0])
    root.link(second.outputs[0], add.inputs[1])
    root.link(add.outputs[0], output.inputs.get("Surface"))

    graph = analyse_material_graph(Material("Instances", root))
    nested_images = [
        node for node in graph.reachable_nodes if node.node_name == "Nested Image"
    ]

    assert len(nested_images) == 2
    assert {node.group_path for node in nested_images} == {
        ("First Instance",),
        ("Second Instance",),
    }
    assert len({node.node_id for node in nested_images}) == 2


def test_material_kind_uses_only_reachable_recursive_nodes():
    nested = make_group_tree()
    root = NodeTree("Root")
    output = material_output(root)
    root.set_output("ALL", output)
    group = root.add(Node("Image Group", "GROUP", node_tree=nested))
    group.output("Shader", identifier="Shader")
    root.link(group.outputs[0], output.inputs.get("Surface"))

    unused = root.add(Node("Unused Missing Image", "TEX_IMAGE", image=None))
    unused.output("Color")

    analysis = analyse_material_slot(0, Material("NestedImageMaterial", root), render_target="CYCLES")

    assert analysis.kind is MaterialKind.IMAGE
    assert tuple(item.image_name for item in analysis.image_dependencies) == (
        "NestedTexture",
    )
    assert all("Unused Missing Image" not in issue for issue in analysis.issues)
    assert "GROUP" in analysis.node_types
    assert "TEX_IMAGE" in analysis.node_types


def test_duplicate_reachable_image_dependency_is_not_unsupported():
    image_data = SimpleNamespace(
        name="Shared",
        source="FILE",
        filepath="//shared.png",
        filepath_raw="//shared.png",
        frame_duration=1,
    )
    root = NodeTree("Root")
    output = material_output(root)
    root.set_output("ALL", output)
    first = root.add(Node("Image A", "TEX_IMAGE", image=image_data))
    second = root.add(Node("Image B", "TEX_IMAGE", image=image_data))
    first.output("Color")
    second.output("Color")
    mix = root.add(Node("Mix Color", "MIX"))
    mix.input("A")
    mix.input("B")
    mix.output("Result")
    root.link(first.outputs[0], mix.inputs[0])
    root.link(second.outputs[0], mix.inputs[1])
    emission = root.add(Node("Emission", "EMISSION"))
    emission.input("Color")
    emission.output("Emission")
    root.link(mix.outputs[0], emission.inputs[0])
    root.link(emission.outputs[0], output.inputs.get("Surface"))

    analysis = analyse_material_slot(0, Material("SharedImage", root), render_target="CYCLES")

    assert analysis.kind is MaterialKind.IMAGE
    assert len(analysis.image_dependencies) == 1


def test_no_output_preserves_legacy_root_node_fallback():
    root = NodeTree("Damaged")
    image = root.add(
        Node(
            "Orphan Image",
            "TEX_IMAGE",
            image=SimpleNamespace(
                name="Orphan",
                source="FILE",
                filepath="//orphan.png",
                filepath_raw="//orphan.png",
                frame_duration=1,
            ),
        )
    )
    image.output("Color")

    with pytest.raises(Exception, match="active output|Material Output"):
        analyse_material_slot(0, Material("DamagedMaterial", root), render_target="CYCLES")


def test_holdout_is_alpha_semantics():
    root = NodeTree("Holdout")
    output = material_output(root)
    root.set_output("ALL", output)
    holdout = root.add(Node("Holdout", "HOLDOUT"))
    holdout.output("Holdout")
    root.link(holdout.outputs[0], output.inputs.get("Surface"))

    graph = analyse_material_graph(Material("HoldoutMaterial", root))

    assert MaterialSemanticChannel.ALPHA in graph.semantic_channels


def test_material_analysis_uses_active_blender_render_engine(monkeypatch):
    tree = NodeTree("RendererOutputs")
    cycles_output = material_output(tree, "Cycles Output", target="CYCLES")
    eevee_output = material_output(tree, "Eevee Output", target="EEVEE")
    tree.set_output("CYCLES", cycles_output)
    tree.set_output("EEVEE", eevee_output)
    layer = make_layer_weight(tree)
    emission = tree.add(Node("Emission", "EMISSION"))
    emission.input("Color")
    emission.output("Emission")
    tree.link(layer.outputs[0], emission.inputs[0])
    tree.link(emission.outputs[0], cycles_output.inputs.get("Surface"))
    diffuse = make_diffuse(tree)
    tree.link(diffuse.outputs[0], eevee_output.inputs.get("Surface"))

    fake_bpy = SimpleNamespace(
        context=SimpleNamespace(
            scene=SimpleNamespace(
                render=SimpleNamespace(engine="BLENDER_EEVEE_NEXT")
            )
        )
    )
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    analysis = analyse_material_slot(0, Material("AutoTarget", tree), render_target="BLENDER_EEVEE")

    assert analysis.graph.active_output_node_id == "Eevee Output"
    assert MaterialDependencyKind.CAMERA not in analysis.graph.dependencies


def test_render_engine_target_normalization():
    assert render_engine_contract("CYCLES").shader_target == "CYCLES"
    assert render_engine_contract("BLENDER_EEVEE").shader_target == "EEVEE"
