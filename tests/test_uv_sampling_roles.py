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
from Blender_to_Spine2D_Mesh_Exporter.domain.uv import (
    UvLayout,
    UvLoopCoordinate,
    apply_uv_layout,
)


def _snapshot(*, render_uv_layer="SourceUV"):
    vertices = tuple(
        MeshVertex(
            id=VertexId(index),
            source_id=SourceVertexId("Object", index),
            position=position,
            normal=(0.0, 0.0, 1.0),
        )
        for index, position in enumerate(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            )
        )
    )
    edges = tuple(
        MeshEdge(
            id=EdgeId(index),
            source_id=SourceEdgeId("Object", index),
            vertex_ids=(VertexId(first), VertexId(second)),
        )
        for index, (first, second) in enumerate(((0, 1), (1, 2), (2, 3), (3, 0)))
    )
    loops = tuple(
        MeshLoop(
            id=LoopId(index),
            source_id=SourceLoopId("Object", 0, index),
            vertex_id=VertexId(index),
            edge_id=EdgeId(index),
            uvs=(LoopUV("SourceUV", (0.25, 0.5)),),
        )
        for index in range(4)
    )
    return MeshSnapshot(
        snapshot_id="Object:snapshot",
        source_object_id="Object",
        object_name="Object",
        vertices=vertices,
        edges=edges,
        loops=loops,
        faces=(
            MeshFace(
                id=FaceId(0),
                source_id=SourceFaceId("Object", 0),
                loop_ids=tuple(LoopId(index) for index in range(4)),
                material_index=0,
                normal=(0.0, 0.0, 1.0),
            ),
        ),
        uv_layer_names=("SourceUV",),
        active_uv_layer="SourceUV",
        render_uv_layer=render_uv_layer,
    )


def _layout(snapshot):
    return UvLayout(
        snapshot_id=snapshot.snapshot_id,
        layer_name="SpineBakeUV",
        coordinates=tuple(
            UvLoopCoordinate(
                loop_id=loop.id,
                source_loop_id=loop.source_id,
                coordinate=coordinate,
            )
            for loop, coordinate in zip(
                snapshot.loops,
                ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            )
        ),
    )


def test_legacy_snapshot_defaults_render_uv_to_active_uv():
    snapshot = _snapshot(render_uv_layer=None)

    assert snapshot.active_uv_layer == "SourceUV"
    assert snapshot.render_uv_layer == "SourceUV"


def test_applying_export_layout_preserves_source_sampling_uv():
    source = _snapshot()
    updated = apply_uv_layout(source, _layout(source))

    assert updated.active_uv_layer == "SpineBakeUV"
    assert updated.render_uv_layer == "SourceUV"
    assert set(updated.uv_layer_names) == {"SourceUV", "SpineBakeUV"}


def test_mesh_writer_has_separate_active_and_render_uv_roles():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "Blender_to_Spine2D_Mesh_Exporter"
        / "blender_adapter"
        / "mesh_writer.py"
    ).read_text(encoding="utf-8")

    assert "layers.active = active" in source
    assert "layer.active_render" in source
    assert "snapshot.render_uv_layer" in source
