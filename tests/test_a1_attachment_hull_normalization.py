from dataclasses import replace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application.a1_attachment_projection import (
    A1AttachmentProjectionError,
    A1AttachmentProjectionResult,
    A1AttachmentVertexKey,
)
from Blender_to_Spine2D_Mesh_Exporter.application.a1_attachment_projection_service import (
    normalize_a1_attachment_projection_hull,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import LoopId, VertexId
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    LegacyAttachmentVertex,
    LegacyMeshAttachmentRequest,
)


def _concave_projection():
    positions = (
        (0.0, 0.0),
        (2.0, 0.0),
        (2.0, 2.0),
        (1.0, 1.0),
        (0.0, 2.0),
    )
    keys = tuple(
        A1AttachmentVertexKey(
            VertexId(index),
            (position[0] / 2.0, position[1] / 2.0),
        )
        for index, position in enumerate(positions)
    )
    triangles = (0, 1, 3, 1, 2, 3, 0, 3, 4)
    request = LegacyMeshAttachmentRequest(
        slot_name="Concave_Segment_0",
        attachment_name="Concave_Segment_0",
        vertex_prefix="Concave_Segment_0",
        image_path="images/Concave_Baked",
        width=100.0,
        height=100.0,
        vertices=tuple(
            LegacyAttachmentVertex(
                index=index,
                uv=keys[index].uv,
                bone_position_pixels=position,
                z_group_index=0,
            )
            for index, position in enumerate(positions)
        ),
        triangles=triangles,
        hull=5,
        edges=(0, 1, 1, 3, 3, 0, 1, 2, 2, 3, 3, 4, 4, 0),
    )
    return A1AttachmentProjectionResult(
        request=request,
        hull_vertex_keys=keys,
        ordered_vertex_keys=keys,
        loop_to_attachment_index=tuple(
            (LoopId(index), attachment_index)
            for index, attachment_index in enumerate(triangles)
        ),
    )


def test_concave_boundary_vertex_moves_after_physical_convex_hull_prefix():
    normalized = normalize_a1_attachment_projection_hull(_concave_projection())

    assert normalized.request.hull == 4
    assert normalized.ordered_vertex_ids == (
        VertexId(0),
        VertexId(1),
        VertexId(2),
        VertexId(4),
        VertexId(3),
    )
    assert normalized.hull_vertex_ids == (
        VertexId(0),
        VertexId(1),
        VertexId(2),
        VertexId(4),
    )
    assert normalized.request.triangles == (0, 1, 4, 1, 2, 4, 0, 4, 3)
    assert normalized.request.edges == (
        0,
        1,
        1,
        4,
        4,
        0,
        1,
        2,
        2,
        4,
        4,
        3,
        3,
        0,
    )
    hull_positions = tuple(
        vertex.bone_position_pixels
        for vertex in normalized.request.vertices[: normalized.request.hull]
    )
    assert hull_positions == (
        (0.0, 0.0),
        (2.0, 0.0),
        (2.0, 2.0),
        (0.0, 2.0),
    )


def test_normalization_is_idempotent():
    first = normalize_a1_attachment_projection_hull(_concave_projection())
    second = normalize_a1_attachment_projection_hull(first)

    assert second == first


def test_zero_area_triangle_in_spine_pixel_space_is_rejected():
    projection = _concave_projection()
    collapsed_vertices = tuple(
        replace(vertex, bone_position_pixels=(1.0, 0.0))
        if vertex.index == 3
        else vertex
        for vertex in projection.request.vertices
    )
    collapsed = replace(
        projection,
        request=replace(projection.request, vertices=collapsed_vertices),
    )

    with pytest.raises(A1AttachmentProjectionError, match="zero area"):
        normalize_a1_attachment_projection_hull(collapsed)
