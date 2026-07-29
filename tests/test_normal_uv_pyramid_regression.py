from math import sqrt

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1DocumentAssemblySettings,
    assemble_a1_document,
    build_a1_z_group_assignment,
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
    SegmentationSettings,
    SourceEdgeId,
    SourceFaceId,
    SourceLoopId,
    SourceVertexId,
    VertexId,
    decompose_complex_segments,
    materialize_decomposed_snapshots,
    segment_mesh_a1,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    LegacyRigBuildRequest,
    SpineValidator,
    build_legacy_rig,
    decode_weighted_vertices,
)


UV_LAYER = "SpineBakeUV"


def _normal(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
    third: tuple[float, float, float],
) -> tuple[float, float, float]:
    first_edge = tuple(second[index] - first[index] for index in range(3))
    second_edge = tuple(third[index] - first[index] for index in range(3))
    cross = (
        first_edge[1] * second_edge[2] - first_edge[2] * second_edge[1],
        first_edge[2] * second_edge[0] - first_edge[0] * second_edge[2],
        first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0],
    )
    length = sqrt(sum(component * component for component in cross))
    return tuple(component / length for component in cross)


def _build_four_face_pyramid() -> MeshSnapshot:
    source_id = "Pyramid"
    positions = (
        (0.0, 0.0, 1.0),
        (-1.0, -1.0, 0.0),
        (1.0, -1.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    # Three side faces plus the triangular base. The winding is consistently
    # outward, but segmentation only depends on the resulting geometric normals.
    face_vertices = (
        (1, 2, 0),
        (2, 3, 0),
        (3, 1, 0),
        (1, 3, 2),
    )
    triangle_uv = ((0.0, 0.0), (1.0, 0.0), (0.5, 1.0))

    vertices = tuple(
        MeshVertex(
            id=VertexId(index),
            source_id=SourceVertexId(source_id, index),
            position=position,
            normal=(0.0, 0.0, 1.0),
        )
        for index, position in enumerate(positions)
    )

    edge_index_by_pair: dict[tuple[int, int], int] = {}
    edge_pairs: list[tuple[int, int]] = []
    loops: list[MeshLoop] = []
    faces: list[MeshFace] = []

    for face_index, polygon in enumerate(face_vertices):
        face_loop_ids: list[LoopId] = []
        for corner_index, vertex_index in enumerate(polygon):
            next_vertex_index = polygon[(corner_index + 1) % len(polygon)]
            edge_pair = tuple(sorted((vertex_index, next_vertex_index)))
            edge_index = edge_index_by_pair.get(edge_pair)
            if edge_index is None:
                edge_index = len(edge_pairs)
                edge_index_by_pair[edge_pair] = edge_index
                edge_pairs.append(edge_pair)

            loop_id = LoopId(len(loops))
            face_loop_ids.append(loop_id)
            loops.append(
                MeshLoop(
                    id=loop_id,
                    source_id=SourceLoopId(
                        source_id,
                        face_index,
                        corner_index,
                    ),
                    vertex_id=VertexId(vertex_index),
                    edge_id=EdgeId(edge_index),
                    uvs=(
                        LoopUV(
                            layer_name=UV_LAYER,
                            coordinate=triangle_uv[corner_index],
                        ),
                    ),
                )
            )

        first, second, third = (
            positions[vertex_index] for vertex_index in polygon
        )
        faces.append(
            MeshFace(
                id=FaceId(face_index),
                source_id=SourceFaceId(source_id, face_index),
                loop_ids=tuple(face_loop_ids),
                material_index=0,
                normal=_normal(first, second, third),
            )
        )

    edges = tuple(
        MeshEdge(
            id=EdgeId(index),
            source_id=SourceEdgeId(source_id, index),
            vertex_ids=(VertexId(first), VertexId(second)),
        )
        for index, (first, second) in enumerate(edge_pairs)
    )
    snapshot = MeshSnapshot(
        snapshot_id="Pyramid:source",
        source_object_id=source_id,
        object_name=source_id,
        vertices=vertices,
        edges=edges,
        loops=tuple(loops),
        faces=tuple(faces),
        uv_layer_names=(UV_LAYER,),
        active_uv_layer=UV_LAYER,
    )
    MeshSnapshotValidator().validate_or_raise(snapshot)
    return snapshot


def test_normal_four_face_pyramid_keeps_four_spine_attachments():
    source = _build_four_face_pyramid()
    segmentation = segment_mesh_a1(
        source,
        SegmentationSettings(
            angle_limit_degrees=30.0,
            split_uv_boundaries=False,
            respect_seams=True,
        ),
    )

    assert tuple(len(segment.face_ids) for segment in segmentation.segments) == (
        1,
        1,
        1,
        1,
    )

    decomposition = decompose_complex_segments(source, segmentation)
    regions = materialize_decomposed_snapshots(source, decomposition)
    assert len(regions) == 4
    assert all(len(region.faces) == 1 for region in regions)

    z_plan = build_a1_z_group_assignment(source)
    rig = build_legacy_rig(
        LegacyRigBuildRequest(
            prefix="Pyramid",
            texture_width=128,
            texture_height=128,
            z_groups=z_plan.groups,
        )
    )
    result = assemble_a1_document(
        rig,
        z_plan,
        regions,
        A1DocumentAssemblySettings(
            prefix="Pyramid",
            uv_layer_name=UV_LAYER,
            image_path="images/Pyramid_Baked",
            attachment_width=128.0,
            attachment_height=128.0,
            center_x=0.0,
            center_y=0.0,
        ),
    )

    assert len(result.projections) == 4
    assert tuple(slot.name for slot in result.document.slots) == (
        "Pyramid_Segment_0",
        "Pyramid_Segment_1",
        "Pyramid_Segment_2",
        "Pyramid_Segment_3",
    )
    attachments = result.document.skins[0].attachments
    assert set(attachments) == {
        "Pyramid_Segment_0",
        "Pyramid_Segment_1",
        "Pyramid_Segment_2",
        "Pyramid_Segment_3",
    }

    vertex_bone_indices = tuple(
        index
        for index, bone in enumerate(result.document.bones)
        if "_Segment_" in bone.name and "_vertex_" in bone.name
    )
    assert len(vertex_bone_indices) == 4

    weighted_indices: list[int] = []
    weighted_vertex_count = 0
    for component in result.document_build.components:
        decoded = decode_weighted_vertices(
            component.attachment.vertices,
            expected_vertex_count=len(component.request.vertices),
        )
        assert len(decoded) == 3
        weighted_vertex_count += len(decoded)
        weighted_indices.extend(
            vertex.influences[0].bone_index for vertex in decoded
        )
        assert all(
            len(vertex.influences) == 1
            and (
                vertex.influences[0].x,
                vertex.influences[0].y,
                vertex.influences[0].weight,
            )
            == (0.0, 0.0, 1.0)
            for vertex in decoded
        )

    assert weighted_vertex_count == 12
    assert set(weighted_indices) == set(vertex_bone_indices)
    assert SpineValidator().validate(result.document) == ()
