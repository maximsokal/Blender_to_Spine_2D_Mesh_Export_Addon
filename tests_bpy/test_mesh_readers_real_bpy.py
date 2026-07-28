"""Integration tests for Blender 5.2 mesh, UV, attribute, and depsgraph adapters."""

from __future__ import annotations

import bpy
import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.evaluated_mesh_reader import (
    read_evaluated_mesh_snapshot,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.mesh_edge_attributes import (
    SHARP_EDGE_ATTRIBUTE,
    UV_SEAM_ATTRIBUTE,
    read_boolean_edge_attribute,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.mesh_reader import (
    MeshReadError,
    read_source_mesh_snapshot,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.mesh_uv_attributes import (
    read_uv_coordinates,
    write_uv_coordinate,
)


def _source_signature(obj) -> tuple[object, ...]:
    mesh = obj.data
    layer = mesh.uv_layers.active
    return (
        tuple(
            tuple(float(component) for component in vertex.co)
            for vertex in mesh.vertices
        ),
        tuple(tuple(int(value) for value in edge.vertices) for edge in mesh.edges),
        tuple(
            tuple(int(value) for value in polygon.vertices)
            for polygon in mesh.polygons
        ),
        read_uv_coordinates(layer, expected_length=len(mesh.loops)),
        read_boolean_edge_attribute(mesh, UV_SEAM_ATTRIBUTE),
        read_boolean_edge_attribute(mesh, SHARP_EDGE_ATTRIBUTE),
        tuple(float(value) for row in obj.matrix_world for value in row),
    )


def _uv_role_signature(obj) -> tuple[object, ...]:
    mesh = obj.data
    active = mesh.uv_layers.active
    return (
        tuple(layer.name for layer in mesh.uv_layers),
        None if active is None else active.name,
        tuple(
            layer.name
            for layer in mesh.uv_layers
            if bool(getattr(layer, "active_render", False))
        ),
        tuple(
            (
                layer.name,
                read_uv_coordinates(layer, expected_length=len(mesh.loops)),
            )
            for layer in mesh.uv_layers
        ),
    )


def _set_unique_render_uv(mesh, layer_name: str) -> None:
    if mesh is None:
        raise TypeError("mesh cannot be None")
    if not isinstance(layer_name, str) or not layer_name.strip():
        raise ValueError("layer_name must be a non-empty string")
    resolved_name = layer_name.strip()
    if mesh.uv_layers.get(resolved_name) is None:
        raise AssertionError(f"UV layer does not exist: {resolved_name}")

    for layer in mesh.uv_layers:
        layer.active_render = str(layer.name) == resolved_name

    active_render_names = tuple(
        str(layer.name)
        for layer in mesh.uv_layers
        if bool(getattr(layer, "active_render", False))
    )
    assert active_render_names == (resolved_name,)


def _datablock_signature() -> tuple[frozenset[str], ...]:
    return (
        frozenset(item.name_full for item in bpy.data.objects),
        frozenset(item.name_full for item in bpy.data.meshes),
        frozenset(item.name_full for item in bpy.data.collections),
    )


def test_uv_attribute_helpers_roundtrip_real_blender_collection(quad_object):
    mesh = quad_object.data
    layer = mesh.uv_layers.active
    assert layer is not None

    write_uv_coordinate(
        layer,
        2,
        (0.75, 0.25),
        expected_length=len(mesh.loops),
    )

    coordinates = read_uv_coordinates(layer, expected_length=len(mesh.loops))
    assert coordinates[2] == pytest.approx((0.75, 0.25))
    assert len(coordinates) == len(mesh.loops) == 4


def test_source_mesh_snapshot_reads_uv_seam_sharp_without_mutation(quad_object):
    before = _source_signature(quad_object)

    snapshot = read_source_mesh_snapshot(quad_object)

    after = _source_signature(quad_object)
    assert after == before
    assert snapshot.source_object_id == quad_object.name_full
    assert snapshot.active_uv_layer == "UVMap"
    assert snapshot.render_uv_layer == "UVMap"
    assert snapshot.uv_layer_names == ("UVMap",)
    assert len(snapshot.vertices) == 4
    assert len(snapshot.edges) == 4
    assert len(snapshot.loops) == 4
    assert len(snapshot.faces) == 1
    assert sum(int(edge.seam) for edge in snapshot.edges) == 1
    assert sum(int(edge.sharp) for edge in snapshot.edges) == 1

    expected_uvs = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    actual_uvs = tuple(loop.uvs[0].coordinate for loop in snapshot.loops)
    assert len(actual_uvs) == len(expected_uvs)
    for actual, expected in zip(actual_uvs, expected_uvs, strict=True):
        assert actual == pytest.approx(expected)


def test_source_mesh_snapshot_rejects_wrong_edge_attribute_type(quad_object):
    mesh = quad_object.data
    existing = mesh.attributes.get(UV_SEAM_ATTRIBUTE)
    assert existing is not None
    mesh.attributes.remove(existing)
    mesh.attributes.new(
        name=UV_SEAM_ATTRIBUTE,
        type="FLOAT",
        domain="EDGE",
    )

    with pytest.raises(MeshReadError, match="BOOLEAN"):
        read_source_mesh_snapshot(quad_object)


def test_evaluated_mesh_reader_restores_source_and_removes_temporary_data(quad_object):
    source_before = _source_signature(quad_object)
    datablocks_before = _datablock_signature()

    result = read_evaluated_mesh_snapshot(
        quad_object,
        depsgraph=bpy.context.evaluated_depsgraph_get(),
        scene=bpy.context.scene,
    )

    source_after = _source_signature(quad_object)
    datablocks_after = _datablock_signature()
    assert source_after == source_before
    assert datablocks_after == datablocks_before
    assert result.modifier_stack == ()
    assert len(result.snapshot.vertices) == 4
    assert len(result.snapshot.edges) == 4
    assert len(result.snapshot.loops) == 4
    assert len(result.snapshot.faces) == 1
    assert result.snapshot.active_uv_layer == "UVMap"
    assert result.snapshot.render_uv_layer == "UVMap"


def test_evaluated_mesh_reader_preserves_independent_source_uv_roles(quad_object):
    mesh = quad_object.data
    original = mesh.uv_layers.get("UVMap")
    assert original is not None
    source_render = mesh.uv_layers.new(name="SourceUV")
    for index, coordinate in enumerate(
        ((0.25, 0.5), (0.25, 0.5), (0.25, 0.5), (0.25, 0.5))
    ):
        write_uv_coordinate(
            source_render,
            index,
            coordinate,
            expected_length=len(mesh.loops),
        )

    mesh.uv_layers.active = source_render
    _set_unique_render_uv(mesh, "SourceUV")

    source_before = _uv_role_signature(quad_object)
    assert source_before[1] == "SourceUV"
    assert source_before[2] == ("SourceUV",)
    datablocks_before = _datablock_signature()

    result = read_evaluated_mesh_snapshot(
        quad_object,
        depsgraph=bpy.context.evaluated_depsgraph_get(),
        scene=bpy.context.scene,
        uv_layer_names=("UVMap", "SourceUV"),
    )

    assert _uv_role_signature(quad_object) == source_before
    assert _datablock_signature() == datablocks_before
    assert result.snapshot.uv_layer_names == ("UVMap", "SourceUV")
    assert result.snapshot.active_uv_layer == "SourceUV"
    assert result.snapshot.render_uv_layer == "SourceUV"
    assert all(
        loop.uv("SourceUV") == pytest.approx((0.25, 0.5))
        for loop in result.snapshot.loops
    )
