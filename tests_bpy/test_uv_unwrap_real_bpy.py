"""Exercise the add-on's context-sensitive UV operator pipeline in real Blender 5.2."""

from __future__ import annotations

import bpy

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.mesh_reader import (
    read_source_mesh_snapshot,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.mesh_uv_attributes import (
    read_uv_coordinates,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.uv_unwrap import (
    unwrap_snapshot_uv,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.uv import UvUnwrapSettings


def _source_signature(obj) -> tuple[object, ...]:
    mesh = obj.data
    active = mesh.uv_layers.active
    return (
        obj.mode,
        tuple(tuple(float(component) for component in vertex.co) for vertex in mesh.vertices),
        tuple(tuple(int(value) for value in edge.vertices) for edge in mesh.edges),
        tuple(tuple(int(value) for value in polygon.vertices) for polygon in mesh.polygons),
        None
        if active is None
        else read_uv_coordinates(active, expected_length=len(mesh.loops)),
    )


def _datablock_signature() -> tuple[frozenset[str], ...]:
    return (
        frozenset(item.name_full for item in bpy.data.objects),
        frozenset(item.name_full for item in bpy.data.meshes),
        frozenset(item.name_full for item in bpy.data.collections),
    )


def test_unwrap_snapshot_uv_runs_real_operators_and_restores_source(quad_object):
    snapshot = read_source_mesh_snapshot(quad_object)
    source_before = _source_signature(quad_object)
    datablocks_before = _datablock_signature()

    result = unwrap_snapshot_uv(
        snapshot,
        UvUnwrapSettings(),
        context=bpy.context,
        scene=bpy.context.scene,
    )

    assert _source_signature(quad_object) == source_before
    assert _datablock_signature() == datablocks_before
    assert quad_object.mode == "OBJECT"
    assert result.snapshot.active_uv_layer == "SpineBakeUV"
    # Export layout becomes active while source material sampling keeps the
    # original renderer UV role. These two roles are intentionally distinct.
    assert result.snapshot.render_uv_layer == "UVMap"
    assert result.statistics.loop_count == len(snapshot.loops) == 4
    assert result.statistics.outside_unit_square_count == 0
    assert 0.0 <= result.statistics.minimum_u <= result.statistics.maximum_u <= 1.0
    assert 0.0 <= result.statistics.minimum_v <= result.statistics.maximum_v <= 1.0
