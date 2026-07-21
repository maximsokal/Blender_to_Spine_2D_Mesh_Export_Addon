import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1AttachmentProjectionResult,
    A1AttachmentVertexKey,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import LoopId, VertexId
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    LegacyAttachmentVertex,
    LegacyMeshAttachmentRequest,
)


def build_keys():
    return (
        A1AttachmentVertexKey(VertexId(0), (0.0, 0.0)),
        A1AttachmentVertexKey(VertexId(1), (1.0, 0.0)),
        A1AttachmentVertexKey(VertexId(2), (0.0, 1.0)),
    )


def build_request(*, hull=3, vertex_uvs=None):
    resolved_uvs = vertex_uvs or ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    vertices = tuple(
        LegacyAttachmentVertex(
            index=index,
            uv=resolved_uvs[index],
            bone_position_pixels=(float(index), float(index)),
            z_group_index=0,
        )
        for index in range(3)
    )
    return LegacyMeshAttachmentRequest(
        slot_name="Segment",
        attachment_name="Segment",
        vertex_prefix="Segment",
        image_path="images/Segment.png",
        width=64.0,
        height=64.0,
        vertices=vertices,
        triangles=(0, 1, 2),
        hull=hull,
        edges=(0, 1, 1, 2, 2, 0),
    )


def build_result(**changes):
    keys = build_keys()
    values = {
        "request": build_request(),
        "hull_vertex_keys": keys,
        "ordered_vertex_keys": keys,
        "loop_to_attachment_index": (
            (LoopId(9), 0),
            (LoopId(2), 1),
            (LoopId(7), 2),
        ),
    }
    values.update(changes)
    return A1AttachmentProjectionResult(**values)


def test_valid_result_preserves_face_corner_mapping_order_without_sorting_loop_ids():
    result = build_result()

    assert result.request.triangles == (0, 1, 2)
    assert tuple(index for _, index in result.loop_to_attachment_index) == (0, 1, 2)
    assert tuple(loop_id.index for loop_id, _ in result.loop_to_attachment_index) == (
        9,
        2,
        7,
    )
    assert result.attachment_index_for_loop(LoopId(2)) == 1


def test_result_requires_one_loop_mapping_per_triangle_corner():
    with pytest.raises(ValueError, match="one entry for every triangle corner"):
        build_result(
            loop_to_attachment_index=((LoopId(9), 0), (LoopId(2), 1))
        )


def test_result_requires_mapping_indices_to_match_triangle_corner_order():
    with pytest.raises(ValueError, match="exactly match request.triangles"):
        build_result(
            loop_to_attachment_index=(
                (LoopId(9), 0),
                (LoopId(2), 2),
                (LoopId(7), 1),
            )
        )


def test_result_rejects_duplicate_loop_identity():
    with pytest.raises(ValueError, match="duplicate LoopId"):
        build_result(
            loop_to_attachment_index=(
                (LoopId(9), 0),
                (LoopId(9), 1),
                (LoopId(7), 2),
            )
        )


def test_result_requires_request_vertex_uvs_to_match_ordered_keys():
    request = build_request(
        vertex_uvs=((0.0, 0.0), (0.75, 0.0), (0.0, 1.0))
    )

    with pytest.raises(ValueError, match="does not match ordered key UV"):
        build_result(request=request)


def test_result_requires_a_physical_hull_with_at_least_three_keys():
    keys = build_keys()

    with pytest.raises(ValueError, match="at least three"):
        build_result(
            request=build_request(hull=2),
            hull_vertex_keys=keys[:2],
            ordered_vertex_keys=keys,
        )
