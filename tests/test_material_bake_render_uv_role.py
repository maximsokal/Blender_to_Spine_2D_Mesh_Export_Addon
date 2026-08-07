"""Regression coverage for generated material-bake UV role ownership."""

from __future__ import annotations

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_uv_preparation import (
    transfer_normal_uv_to_material_bake_snapshot,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
    EdgeId,
    FaceId,
    LoopId,
    LoopUV,
    MeshEdge,
    MeshFace,
    MeshLoop,
    MeshSnapshot,
    MeshVertex,
    SourceEdgeId,
    SourceFaceId,
    SourceLoopId,
    SourceVertexId,
    VertexId,
)


_OBJECT_ID = "UvRoleFixture"
_DESTINATION_UV = "SpineBakeUV"
_SOURCE_UV = "SourceUV"


def _snapshot(
    *,
    snapshot_id: str,
    destination_uv: bool,
    source_uv: bool,
    active_uv_layer: str | None,
    render_uv_layer: str | None,
) -> MeshSnapshot:
    vertices = tuple(
        MeshVertex(
            id=VertexId(index),
            source_id=SourceVertexId(_OBJECT_ID, index),
            position=position,
            normal=(0.0, 0.0, 1.0),
        )
        for index, position in enumerate(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            )
        )
    )
    edges = (
        MeshEdge(
            id=EdgeId(0),
            source_id=SourceEdgeId(_OBJECT_ID, 0),
            vertex_ids=(VertexId(0), VertexId(1)),
        ),
        MeshEdge(
            id=EdgeId(1),
            source_id=SourceEdgeId(_OBJECT_ID, 1),
            vertex_ids=(VertexId(1), VertexId(2)),
        ),
        MeshEdge(
            id=EdgeId(2),
            source_id=SourceEdgeId(_OBJECT_ID, 2),
            vertex_ids=(VertexId(2), VertexId(0)),
        ),
    )
    destination_coordinates = (
        (0.1, 0.2),
        (0.8, 0.2),
        (0.1, 0.9),
    )
    source_coordinates = (
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
    )
    loops = []
    for index in range(3):
        uvs = []
        if source_uv:
            uvs.append(
                LoopUV(
                    layer_name=_SOURCE_UV,
                    coordinate=source_coordinates[index],
                )
            )
        if destination_uv:
            uvs.append(
                LoopUV(
                    layer_name=_DESTINATION_UV,
                    coordinate=destination_coordinates[index],
                )
            )
        loops.append(
            MeshLoop(
                id=LoopId(index),
                source_id=SourceLoopId(_OBJECT_ID, 0, index),
                vertex_id=VertexId(index),
                edge_id=EdgeId(index),
                uvs=tuple(uvs),
            )
        )

    uv_layer_names = tuple(
        name
        for name, enabled in (
            (_SOURCE_UV, source_uv),
            (_DESTINATION_UV, destination_uv),
        )
        if enabled
    )
    return MeshSnapshot(
        snapshot_id=snapshot_id,
        source_object_id=_OBJECT_ID,
        object_name=_OBJECT_ID,
        vertices=vertices,
        edges=edges,
        loops=tuple(loops),
        faces=(
            MeshFace(
                id=FaceId(0),
                source_id=SourceFaceId(_OBJECT_ID, 0),
                loop_ids=(LoopId(0), LoopId(1), LoopId(2)),
                material_index=0,
                normal=(0.0, 0.0, 1.0),
            ),
        ),
        uv_layer_names=uv_layer_names,
        active_uv_layer=active_uv_layer,
        render_uv_layer=render_uv_layer,
    )


def test_uv_less_source_promotes_generated_destination_to_first_render_role() -> None:
    projected = _snapshot(
        snapshot_id="projected",
        destination_uv=True,
        source_uv=False,
        active_uv_layer=_DESTINATION_UV,
        render_uv_layer=_DESTINATION_UV,
    )
    material = _snapshot(
        snapshot_id="material-no-uv",
        destination_uv=False,
        source_uv=False,
        active_uv_layer=None,
        render_uv_layer=None,
    )

    updated, report = transfer_normal_uv_to_material_bake_snapshot(
        projected,
        material,
        layer_name=_DESTINATION_UV,
    )

    assert report.updated_loop_count == 3
    assert updated.active_uv_layer == _DESTINATION_UV
    assert updated.render_uv_layer == _DESTINATION_UV
    assert updated.vertices == material.vertices
    assert updated.edges == material.edges
    assert updated.faces == material.faces
    assert updated.world_matrix == material.world_matrix


def test_existing_source_render_role_is_preserved_while_destination_becomes_active() -> None:
    projected = _snapshot(
        snapshot_id="projected",
        destination_uv=True,
        source_uv=False,
        active_uv_layer=_DESTINATION_UV,
        render_uv_layer=_DESTINATION_UV,
    )
    material = _snapshot(
        snapshot_id="material-source-uv",
        destination_uv=False,
        source_uv=True,
        active_uv_layer=_SOURCE_UV,
        render_uv_layer=_SOURCE_UV,
    )

    updated, report = transfer_normal_uv_to_material_bake_snapshot(
        projected,
        material,
        layer_name=_DESTINATION_UV,
    )

    assert report.updated_loop_count == 3
    assert updated.active_uv_layer == _DESTINATION_UV
    assert updated.render_uv_layer == _SOURCE_UV
    assert set(updated.uv_layer_names) == {_SOURCE_UV, _DESTINATION_UV}
    for original_loop, updated_loop in zip(
        material.loops,
        updated.loops,
        strict=True,
    ):
        assert updated_loop.uv(_SOURCE_UV) == original_loop.uv(_SOURCE_UV)
