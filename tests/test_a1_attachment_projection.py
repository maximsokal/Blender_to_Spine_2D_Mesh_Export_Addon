from dataclasses import replace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1AttachmentProjectionError,
    A1AttachmentProjectionSettings,
    A1AttachmentVertexKey,
    A1VertexZBinding,
    project_triangulated_disk_attachment,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
    LoopId,
    LoopUV,
    VertexId,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    LegacyRigBuildRequest,
    LegacyZGroup,
    SpineValidator,
    build_legacy_mesh_attachment,
    build_legacy_rig,
)

from test_geometry_domain import build_square_snapshot


def make_rig():
    return build_legacy_rig(
        LegacyRigBuildRequest(
            prefix="Cube",
            texture_width=100,
            texture_height=100,
            z_groups=(LegacyZGroup(0.0, height_real_pixels=0.0),),
        )
    )


def make_settings():
    return A1AttachmentProjectionSettings(
        slot_name="Cube_Segment_0",
        attachment_name="Cube_Segment_0",
        vertex_prefix="Cube_Segment_0",
        image_path="images/Cube_Baked",
        uv_layer_name="UVMap",
        attachment_width=100.0,
        attachment_height=100.0,
        center_x=0.5,
        center_y=0.5,
        z_bindings=tuple(
            A1VertexZBinding(VertexId(index), 1) for index in range(4)
        ),
    )


def build_boundary_uv_split_snapshot():
    snapshot = build_square_snapshot()
    changed_loop = replace(
        snapshot.loops[3],
        uvs=(LoopUV("UVMap", (0.25, 0.25)),),
    )
    return replace(
        snapshot,
        loops=snapshot.loops[:3] + (changed_loop,) + snapshot.loops[4:],
    )


def test_square_projection_places_deterministic_hull_first():
    result = project_triangulated_disk_attachment(
        build_square_snapshot(),
        make_rig(),
        make_settings(),
    )

    assert result.hull_vertex_ids == (
        VertexId(0),
        VertexId(1),
        VertexId(2),
        VertexId(3),
    )
    assert result.ordered_vertex_ids == result.hull_vertex_ids
    assert result.old_to_attachment_index == (
        (VertexId(0), 0),
        (VertexId(1), 1),
        (VertexId(2), 2),
        (VertexId(3), 3),
    )
    assert result.request.hull == 4
    assert result.request.triangles == (0, 1, 2, 0, 2, 3)
    assert result.request.edges == (0, 1, 1, 2, 2, 0, 2, 3, 3, 0)
    assert result.loop_to_attachment_index == (
        (LoopId(0), 0),
        (LoopId(1), 1),
        (LoopId(2), 2),
        (LoopId(3), 0),
        (LoopId(4), 2),
        (LoopId(5), 3),
    )


def test_projection_converts_xy_to_legacy_pixel_bone_positions():
    result = project_triangulated_disk_attachment(
        build_square_snapshot(),
        make_rig(),
        make_settings(),
    )

    assert tuple(vertex.bone_position_pixels for vertex in result.request.vertices) == (
        (-50.0, 50.0),
        (50.0, 50.0),
        (50.0, -50.0),
        (-50.0, -50.0),
    )
    assert tuple(vertex.uv for vertex in result.request.vertices) == (
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    )
    assert all(vertex.z_group_index == 1 for vertex in result.request.vertices)


def test_projected_request_builds_a_fully_valid_spine_document():
    rig = make_rig()
    projection = project_triangulated_disk_attachment(
        build_square_snapshot(),
        rig,
        make_settings(),
    )
    attachment = build_legacy_mesh_attachment(rig, projection.request)

    assert attachment.attachment.hull == 4
    assert len(attachment.vertex_bones) == 4
    assert SpineValidator().validate(attachment.document) == ()


def test_uv_split_vertex_is_duplicated_without_merging_texture_coordinates():
    result = project_triangulated_disk_attachment(
        build_boundary_uv_split_snapshot(),
        make_rig(),
        make_settings(),
    )

    assert result.hull_vertex_ids == (
        VertexId(0),
        VertexId(0),
        VertexId(1),
        VertexId(2),
        VertexId(3),
    )
    assert result.ordered_vertex_ids == result.hull_vertex_ids
    assert result.hull_vertex_keys[:2] == (
        A1AttachmentVertexKey(VertexId(0), (0.25, 0.25)),
        A1AttachmentVertexKey(VertexId(0), (0.0, 0.0)),
    )
    assert tuple(vertex.uv for vertex in result.request.vertices) == (
        (0.25, 0.25),
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    )
    assert result.request.hull == 5
    assert result.request.triangles == (1, 2, 3, 0, 3, 4)
    assert result.request.edges == (
        1,
        2,
        2,
        3,
        3,
        1,
        0,
        3,
        3,
        4,
        4,
        0,
    )
    assert result.loop_to_attachment_index == (
        (LoopId(0), 1),
        (LoopId(1), 2),
        (LoopId(2), 3),
        (LoopId(3), 0),
        (LoopId(4), 3),
        (LoopId(5), 4),
    )
    assert result.attachment_indices_for(VertexId(0)) == (0, 1)
    assert result.attachment_index_for_loop(LoopId(0)) == 1
    assert result.attachment_index_for_loop(LoopId(3)) == 0
    assert result.attachment_index_for(
        VertexId(0),
        uv=(0.25, 0.25),
    ) == 0
    assert result.attachment_index_for(VertexId(0), uv=(0.0, 0.0)) == 1
    with pytest.raises(A1AttachmentProjectionError, match="provide uv"):
        result.attachment_index_for(VertexId(0))
    with pytest.raises(A1AttachmentProjectionError, match="one-to-many"):
        _ = result.old_to_attachment_index


def test_uv_split_projection_builds_valid_document_and_duplicate_vertex_bones():
    rig = make_rig()
    projection = project_triangulated_disk_attachment(
        build_boundary_uv_split_snapshot(),
        rig,
        make_settings(),
    )
    attachment = build_legacy_mesh_attachment(rig, projection.request)

    assert len(attachment.vertex_bones) == 5
    assert attachment.vertex_bones[0].x == attachment.vertex_bones[1].x
    assert attachment.vertex_bones[0].y == attachment.vertex_bones[1].y
    assert attachment.attachment.hull == 5
    assert SpineValidator().validate(attachment.document) == ()


def test_z_bindings_must_cover_vertices_exactly():
    settings = replace(
        make_settings(),
        z_bindings=(
            A1VertexZBinding(VertexId(0), 1),
            A1VertexZBinding(VertexId(1), 1),
            A1VertexZBinding(VertexId(2), 1),
        ),
    )
    with pytest.raises(A1AttachmentProjectionError, match="cover snapshot vertices"):
        project_triangulated_disk_attachment(
            build_square_snapshot(),
            make_rig(),
            settings,
        )


def test_unknown_rig_z_group_is_rejected_before_attachment_build():
    settings = replace(
        make_settings(),
        z_bindings=tuple(
            A1VertexZBinding(VertexId(index), 99) for index in range(4)
        ),
    )
    with pytest.raises(A1AttachmentProjectionError, match="unknown rig groups"):
        project_triangulated_disk_attachment(
            build_square_snapshot(),
            make_rig(),
            settings,
        )


def test_projection_is_repeatable_with_and_without_uv_splits():
    rig = make_rig()
    settings = make_settings()
    for snapshot in (build_square_snapshot(), build_boundary_uv_split_snapshot()):
        assert project_triangulated_disk_attachment(
            snapshot,
            rig,
            settings,
        ) == project_triangulated_disk_attachment(snapshot, rig, settings)
