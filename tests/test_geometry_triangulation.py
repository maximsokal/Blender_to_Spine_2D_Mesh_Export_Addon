from dataclasses import replace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1AttachmentProjectionSettings,
    A1VertexZBinding,
    project_triangulated_disk_attachment,
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
    MeshSnapshotValidator,
    MeshVertex,
    SourceEdgeId,
    SourceFaceId,
    SourceLoopId,
    SourceVertexId,
    TriangulationError,
    TriangulationSettings,
    VertexId,
    triangulate_snapshot,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    LegacyRigBuildRequest,
    LegacyZGroup,
    SpineValidator,
    build_legacy_mesh_attachment,
    build_legacy_rig,
)


def build_polygon_snapshot(points, *, name="Polygon", normal=(0.0, 0.0, 1.0)):
    source = name
    vertices = tuple(
        MeshVertex(
            id=VertexId(index),
            source_id=SourceVertexId(source, index),
            position=tuple(float(value) for value in point),
            normal=normal,
        )
        for index, point in enumerate(points)
    )
    edges = tuple(
        MeshEdge(
            id=EdgeId(index),
            source_id=SourceEdgeId(source, index),
            vertex_ids=(VertexId(index), VertexId((index + 1) % len(points))),
        )
        for index in range(len(points))
    )
    loops = tuple(
        MeshLoop(
            id=LoopId(index),
            source_id=SourceLoopId(source, 0, index),
            vertex_id=VertexId(index),
            edge_id=EdgeId(index),
            uvs=(LoopUV("UVMap", (float(point[0]) / 2.0, float(point[1]) / 2.0)),),
        )
        for index, point in enumerate(points)
    )
    face = MeshFace(
        id=FaceId(0),
        source_id=SourceFaceId(source, 0),
        loop_ids=tuple(loop.id for loop in loops),
        material_index=0,
        normal=normal,
    )
    snapshot = MeshSnapshot(
        snapshot_id=name,
        source_object_id=source,
        object_name=name,
        vertices=vertices,
        edges=edges,
        loops=loops,
        faces=(face,),
        uv_layer_names=("UVMap",),
        active_uv_layer="UVMap",
    )
    MeshSnapshotValidator().validate_or_raise(snapshot)
    return snapshot


def triangle_vertex_indices(snapshot):
    loop_map = snapshot.loop_by_id()
    return tuple(
        tuple(loop_map[loop_id].vertex_id.index for loop_id in face.loop_ids)
        for face in snapshot.faces
    )


def test_convex_quad_triangulates_deterministically_with_one_generated_edge():
    source = build_polygon_snapshot(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0)),
        name="Quad",
    )

    first = triangulate_snapshot(source)
    second = triangulate_snapshot(source)

    assert first == second
    assert triangle_vertex_indices(first.snapshot) == ((3, 0, 1), (1, 2, 3))
    assert len(first.snapshot.faces) == 2
    assert len(first.snapshot.edges) == 5
    assert len(first.generated_edge_ids) == 1
    generated = first.snapshot.edge_by_id()[first.generated_edge_ids[0]]
    assert generated.source_id is None
    assert generated.vertex_ids == (VertexId(1), VertexId(3))
    assert all(face.source_id == SourceFaceId("Quad", 0) for face in first.snapshot.faces)
    MeshSnapshotValidator().validate_or_raise(first.snapshot)


def test_source_loop_lineage_is_reused_for_generated_triangles():
    source = build_polygon_snapshot(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0)),
        name="Quad",
    )
    result = triangulate_snapshot(source)
    source_ids = tuple(loop.source_id for loop in result.snapshot.loops)

    assert source_ids == (
        SourceLoopId("Quad", 0, 3),
        SourceLoopId("Quad", 0, 0),
        SourceLoopId("Quad", 0, 1),
        SourceLoopId("Quad", 0, 1),
        SourceLoopId("Quad", 0, 2),
        SourceLoopId("Quad", 0, 3),
    )
    assert tuple(loop.uv("UVMap") for loop in result.snapshot.loops) == (
        (0.0, 1.0),
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    )


def test_concave_polygon_produces_n_minus_two_non_degenerate_triangles():
    source = build_polygon_snapshot(
        (
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 2.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 2.0, 0.0),
        ),
        name="Concave",
    )
    result = triangulate_snapshot(source)

    assert len(result.snapshot.faces) == 3
    assert all(len(face.loop_ids) == 3 for face in result.snapshot.faces)
    assert len(result.faces[0].output_face_ids) == 3
    assert result.faces[0].original_corner_count == 5
    MeshSnapshotValidator().validate_or_raise(result.snapshot)


def test_self_intersecting_polygon_is_rejected():
    source = build_polygon_snapshot(
        (
            (0.0, 0.0, 0.0),
            (2.0, 2.0, 0.0),
            (0.0, 2.0, 0.0),
            (2.0, 0.0, 0.0),
        ),
        name="BowTie",
    )
    with pytest.raises(TriangulationError):
        triangulate_snapshot(source)


def test_non_planar_ngon_is_rejected_by_explicit_tolerance():
    source = build_polygon_snapshot(
        (
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 2.0, 0.1),
            (0.0, 2.0, 0.0),
        ),
        name="NonPlanar",
    )
    with pytest.raises(TriangulationError, match="not planar"):
        triangulate_snapshot(
            source,
            TriangulationSettings(planarity_tolerance=1e-5),
        )


def test_triangulated_quad_projects_and_builds_spine_attachment():
    source = build_polygon_snapshot(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0)),
        name="Quad",
    )
    triangulated = triangulate_snapshot(source).snapshot
    rig = build_legacy_rig(
        LegacyRigBuildRequest(
            prefix="Quad",
            texture_width=100,
            texture_height=100,
            z_groups=(LegacyZGroup(0.0, height_real_pixels=0.0),),
        )
    )
    projection = project_triangulated_disk_attachment(
        triangulated,
        rig,
        A1AttachmentProjectionSettings(
            slot_name="Quad_Segment_0",
            attachment_name="Quad_Segment_0",
            vertex_prefix="Quad_Segment_0",
            image_path="images/Quad_Baked",
            uv_layer_name="UVMap",
            attachment_width=200.0,
            attachment_height=200.0,
            center_x=1.0,
            center_y=1.0,
            z_bindings=tuple(
                A1VertexZBinding(VertexId(index), 1) for index in range(4)
            ),
        ),
    )
    result = build_legacy_mesh_attachment(rig, projection.request)

    assert projection.request.hull == 4
    assert len(projection.request.triangles) == 6
    assert len(result.vertex_bones) == 4
    assert SpineValidator().validate(result.document) == ()
