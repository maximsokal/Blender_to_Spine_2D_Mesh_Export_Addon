from types import SimpleNamespace

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import analyse_material_graph
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    MaterialDependencyKind,
    MaterialSemanticChannel,
)


class FakeSocket:
    def __init__(self, name, default_value=0.0):
        self.name = name
        self.default_value = default_value
        self.is_linked = False
        self.links = []


class FakeSockets(list):
    def get(self, name):
        for socket in self:
            if socket.name == name:
                return socket
        return None


class FakeNode:
    def __init__(
        self,
        node_type,
        name,
        *,
        inputs=(),
        outputs=(),
        active=False,
        image=None,
    ):
        self.type = node_type
        self.name = name
        self.inputs = FakeSockets(inputs)
        self.outputs = FakeSockets(outputs)
        self.is_active_output = active
        self.image = image


class FakeLink:
    def __init__(self, from_node, from_socket, to_node, to_socket):
        self.from_node = from_node
        self.from_socket = from_socket
        self.to_node = to_node
        self.to_socket = to_socket
        from_socket.is_linked = True
        to_socket.is_linked = True
        from_socket.links.append(self)
        to_socket.links.append(self)


class FakeMaterial:
    def __init__(self, nodes, links=(), *, animation_data=None):
        self.name = "GraphMaterial"
        self.use_nodes = True
        self.animation_data = animation_data
        self.node_tree = SimpleNamespace(
            nodes=tuple(nodes),
            links=tuple(links),
            animation_data=None,
        )


def output_node():
    return FakeNode(
        "OUTPUT_MATERIAL",
        "Material Output",
        inputs=(
            FakeSocket("Surface"),
            FakeSocket("Volume"),
            FakeSocket("Displacement"),
        ),
        active=True,
    )


def test_unreachable_emission_node_does_not_change_surface_semantics():
    output = output_node()
    principled = FakeNode(
        "BSDF_PRINCIPLED",
        "Principled",
        inputs=(
            FakeSocket("Emission Color", (0.0, 0.0, 0.0, 1.0)),
            FakeSocket("Emission Strength", 1.0),
            FakeSocket("Alpha", 1.0),
        ),
        outputs=(FakeSocket("BSDF"),),
    )
    unused_emission = FakeNode(
        "EMISSION",
        "Unused Glow",
        inputs=(FakeSocket("Color", (1.0, 0.0, 0.0, 1.0)),),
        outputs=(FakeSocket("Emission"),),
    )
    link = FakeLink(
        principled,
        principled.outputs[0],
        output,
        output.inputs.get("Surface"),
    )

    graph = analyse_material_graph(FakeMaterial((output, principled, unused_emission), (link,)))

    assert graph.semantic_channels == (MaterialSemanticChannel.SURFACE_COLOR,)
    assert tuple(node.node_name for node in graph.reachable_nodes) == (
        "Material Output",
        "Principled",
    )


def test_add_shader_surface_and_emission_produces_two_semantic_channels():
    output = output_node()
    principled = FakeNode(
        "BSDF_PRINCIPLED",
        "Body",
        inputs=(FakeSocket("Alpha", 1.0),),
        outputs=(FakeSocket("BSDF"),),
    )
    emission = FakeNode(
        "EMISSION",
        "Glow",
        inputs=(FakeSocket("Color", (0.0, 0.0, 1.0, 1.0)),),
        outputs=(FakeSocket("Emission"),),
    )
    add = FakeNode(
        "ADD_SHADER",
        "Add Shader",
        inputs=(FakeSocket("Shader"), FakeSocket("Shader_001")),
        outputs=(FakeSocket("Shader"),),
    )
    links = (
        FakeLink(principled, principled.outputs[0], add, add.inputs[0]),
        FakeLink(emission, emission.outputs[0], add, add.inputs[1]),
        FakeLink(add, add.outputs[0], output, output.inputs.get("Surface")),
    )

    graph = analyse_material_graph(FakeMaterial((output, principled, emission, add), links))

    assert graph.semantic_channels == (
        MaterialSemanticChannel.SURFACE_COLOR,
        MaterialSemanticChannel.SURFACE_EMISSION,
    )


def test_reachable_sequence_image_records_image_and_time_dependencies():
    output = output_node()
    image = SimpleNamespace(source="SEQUENCE", frame_duration=8)
    texture = FakeNode(
        "TEX_IMAGE",
        "Walk Sequence",
        outputs=(FakeSocket("Color"),),
        image=image,
    )
    principled = FakeNode(
        "BSDF_PRINCIPLED",
        "Body",
        inputs=(FakeSocket("Base Color"), FakeSocket("Alpha", 1.0)),
        outputs=(FakeSocket("BSDF"),),
    )
    links = (
        FakeLink(texture, texture.outputs[0], principled, principled.inputs[0]),
        FakeLink(principled, principled.outputs[0], output, output.inputs.get("Surface")),
    )

    graph = analyse_material_graph(FakeMaterial((output, texture, principled), links))

    assert MaterialDependencyKind.IMAGE in graph.dependencies
    assert MaterialDependencyKind.TIME in graph.dependencies


def test_volume_output_is_reported_independently_from_surface():
    output = output_node()
    volume = FakeNode(
        "PRINCIPLED_VOLUME",
        "Volume",
        outputs=(FakeSocket("Volume"),),
    )
    link = FakeLink(volume, volume.outputs[0], output, output.inputs.get("Volume"))

    graph = analyse_material_graph(FakeMaterial((output, volume), (link,)))

    assert graph.semantic_channels == (MaterialSemanticChannel.VOLUME,)
