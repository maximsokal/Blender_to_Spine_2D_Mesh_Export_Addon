"""Real Blender regressions for explicit source-UV binding on temporary materials."""

from __future__ import annotations

import warnings

import bpy

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.material_uv_binding import (
    bind_material_implicit_uv_sampling,
)


def _enable_nodes(material) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*Material.use_nodes.*",
            category=DeprecationWarning,
        )
        material.use_nodes = True


def _graph_signature(material) -> tuple[object, ...]:
    nodes = tuple(
        sorted(
            (
                node.name,
                node.bl_idname,
                str(getattr(node, "uv_map", "") or ""),
            )
            for node in material.node_tree.nodes
        )
    )
    links = tuple(
        sorted(
            (
                link.from_node.name,
                link.from_socket.name,
                link.to_node.name,
                link.to_socket.name,
            )
            for link in material.node_tree.links
        )
    )
    return nodes, links


def _incoming_source_node(socket):
    links = tuple(socket.links)
    assert len(links) == 1
    return links[0].from_node


def test_temporary_material_binds_only_implicit_uv_consumers(clean_blender_data):
    source = bpy.data.materials.new(name="SourceUvBindingMaterial")
    _enable_nodes(source)
    source.node_tree.nodes.clear()

    texture_coordinate = source.node_tree.nodes.new(type="ShaderNodeTexCoord")
    texture_coordinate.name = "Implicit Texture Coordinate"
    mapping = source.node_tree.nodes.new(type="ShaderNodeMapping")
    mapping.name = "Mapping"
    linked_image = source.node_tree.nodes.new(type="ShaderNodeTexImage")
    linked_image.name = "Linked Source Image"
    unlinked_image = source.node_tree.nodes.new(type="ShaderNodeTexImage")
    unlinked_image.name = "Unlinked Source Image"

    source.node_tree.links.new(
        texture_coordinate.outputs["UV"],
        mapping.inputs["Vector"],
    )
    source.node_tree.links.new(
        mapping.outputs["Vector"],
        linked_image.inputs["Vector"],
    )
    source_before = _graph_signature(source)

    temporary = source.copy()
    temporary.name = "__Spine2D_UvBindingCopy"
    bake_target = temporary.node_tree.nodes.new(type="ShaderNodeTexImage")
    bake_target.name = "__Spine2D_BakeTarget"

    report = bind_material_implicit_uv_sampling(
        temporary,
        "SourceUV",
        excluded_nodes=(bake_target,),
    )

    assert report.texture_coordinate_link_count == 1
    assert report.unlinked_image_texture_count == 1
    explicit_nodes = tuple(
        node
        for node in temporary.node_tree.nodes
        if node.name.startswith("__Spine2D_SourceUV_")
    )
    assert len(explicit_nodes) == 1
    explicit_uv = explicit_nodes[0]
    assert explicit_uv.bl_idname == "ShaderNodeUVMap"
    assert explicit_uv.uv_map == "SourceUV"

    temporary_mapping = temporary.node_tree.nodes["Mapping"]
    temporary_unlinked = temporary.node_tree.nodes["Unlinked Source Image"]
    assert _incoming_source_node(temporary_mapping.inputs["Vector"]) == explicit_uv
    assert _incoming_source_node(temporary_unlinked.inputs["Vector"]) == explicit_uv
    assert len(tuple(bake_target.inputs["Vector"].links)) == 0
    assert _graph_signature(source) == source_before
