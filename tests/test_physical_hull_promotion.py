"""Regression for physical hull points outside the raw topological boundary."""

from Blender_to_Spine2D_Mesh_Exporter.application import (
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
    LegacyRigBuildRequest,
    LegacyZGroup,
    SpineValidator,
    build_legacy_mesh_attachment,
    build_legacy_rig,
)


def _build_raw_projection_with_physical_tail_hull_point():
    """Build a disk fan whose topological center is physically extreme in XY."""

    positions = (
        (-1.0, -1.0),
        (1.0, -1.0),
        (1.0, 1.0),
        (-1.0, 1.0),
        (2.0, 0.0),
    )
    uvs = (
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (0.5, 0.5),
    )
    vertices = tuple(
        LegacyAttachmentVertex(
            index=index,
            uv=uvs[index],
            bone_position_pixels=positions[index],
            z_group_index=1,
        )
        for index in range(5)
    )
    keys = tuple(
        A1AttachmentVertexKey(VertexId(index), uvs[index])
        for index in range(5)
    )
    triangles = (
        0,
        1,
        4,
        1,
        2,
        4,
        2,
        3,
        4,
        3,
        0,
        4,
    )
    request = LegacyMeshAttachmentRequest(
        slot_name="Fan_Segment_0",
        attachment_name="Fan_Segment_0",
        vertex_prefix="Fan_Segment_0",
        image_path="images/Fan_Baked",
        width=100.0,
        height=100.0,
        vertices=vertices,
        triangles=triangles,
        hull=4,
        edges=(
            0,
            1,
            1,
            2,
            2,
            3,
            3,
            0,
            0,
            4,
            1,
            4,
            2,
            4,
            3,
            4,
        ),
    )
    return A1AttachmentProjectionResult(
        request=request,
        hull_vertex_keys=keys[:4],
        ordered_vertex_keys=keys,
        loop_to_attachment_index=tuple(
            (LoopId(loop_index), attachment_index)
            for loop_index, attachment_index in enumerate(triangles)
        ),
    )


def _make_rig():
    return build_legacy_rig(
        LegacyRigBuildRequest(
            prefix="Fan",
            texture_width=100,
            texture_height=100,
            z_groups=(LegacyZGroup(0.0, height_real_pixels=0.0),),
        )
    )


def test_topological_interior_vertex_is_promoted_into_physical_hull():
    raw = _build_raw_projection_with_physical_tail_hull_point()

    result = normalize_a1_attachment_projection_hull(raw)

    assert result.request.hull == 5
    assert result.ordered_vertex_ids == (
        VertexId(0),
        VertexId(1),
        VertexId(4),
        VertexId(2),
        VertexId(3),
    )
    assert tuple(
        vertex.bone_position_pixels
        for vertex in result.request.vertices[: result.request.hull]
    ) == (
        (-1.0, -1.0),
        (1.0, -1.0),
        (2.0, 0.0),
        (1.0, 1.0),
        (-1.0, 1.0),
    )
    assert result.request.triangles == (
        0,
        1,
        2,
        1,
        3,
        2,
        3,
        4,
        2,
        4,
        0,
        2,
    )
    assert result.request.edges == (
        0,
        1,
        1,
        3,
        3,
        4,
        4,
        0,
        0,
        2,
        1,
        2,
        3,
        2,
        4,
        2,
    )
    assert tuple(
        attachment_index
        for _loop_id, attachment_index in result.loop_to_attachment_index
    ) == result.request.triangles

    attachment = build_legacy_mesh_attachment(_make_rig(), result.request)
    assert attachment.attachment.hull == 5
    assert SpineValidator().validate(attachment.document) == ()


def test_physical_hull_promotion_is_repeatable_and_idempotent():
    raw = _build_raw_projection_with_physical_tail_hull_point()

    first = normalize_a1_attachment_projection_hull(raw)
    second = normalize_a1_attachment_projection_hull(raw)

    assert first == second
    assert normalize_a1_attachment_projection_hull(first) == first
