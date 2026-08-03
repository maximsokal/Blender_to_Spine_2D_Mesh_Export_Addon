"""Regression tests for camera-zero depth and one compensated depth attachment."""

from __future__ import annotations

from dataclasses import replace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1AttachmentProjectionSettings,
    build_a1_z_group_assignment,
)
from Blender_to_Spine2D_Mesh_Exporter.application.a1_depth_attachment_projection import (
    project_depth_camera_attachment,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.camera_projection import (
    A1CameraProjectionFrame,
    A1CameraProjectionKind,
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
    SourceFaceId,
    SourceLoopId,
    SourceVertexId,
    VertexId,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry.depth_camera_distance import (
    convert_depth_result_to_camera_distance,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry.depth_camera_projection import (
    DepthCameraProjectionResult,
    DepthProjectionBaseMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.projection import A1ProjectedPoint
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    LegacyRigBuildRequest,
    LegacyZGroupOriginMode,
    SpineValidator,
    build_legacy_mesh_attachment,
    build_legacy_rig,
)


_OBJECT_ID = "DepthRigSpace"
_IDENTITY = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)


def _snapshot(
    depths: tuple[float, float, float, float, float, float] = (
        2.0,
        2.0,
        4.0,
        6.0,
        6.0,
        8.0,
    ),
) -> MeshSnapshot:
    positions = (
        (-1.0, -1.0, depths[0]),
        (0.0, -1.0, depths[1]),
        (-1.0, 0.0, depths[2]),
        (1.0, 0.0, depths[3]),
        (2.0, 0.0, depths[4]),
        (2.0, 1.0, depths[5]),
    )
    vertices = tuple(
        MeshVertex(
            id=VertexId(index),
            source_id=SourceVertexId(_OBJECT_ID, index),
            position=position,
            normal=(0.0, 0.0, 1.0),
        )
        for index, position in enumerate(positions)
    )
    face_vertices = ((0, 1, 2), (3, 4, 5))
    edge_pairs = ((0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5))
    edge_by_pair = {
        tuple(sorted(pair)): EdgeId(index)
        for index, pair in enumerate(edge_pairs)
    }
    edges = tuple(
        MeshEdge(
            id=EdgeId(index),
            source_id=None,
            vertex_ids=(VertexId(first), VertexId(second)),
        )
        for index, (first, second) in enumerate(edge_pairs)
    )

    loops: list[MeshLoop] = []
    faces: list[MeshFace] = []
    for face_index, face in enumerate(face_vertices):
        loop_ids: list[LoopId] = []
        for corner_index, vertex_index in enumerate(face):
            following = face[(corner_index + 1) % 3]
            position = positions[vertex_index]
            loop_id = LoopId(len(loops))
            loops.append(
                MeshLoop(
                    id=loop_id,
                    source_id=SourceLoopId(
                        _OBJECT_ID,
                        face_index,
                        corner_index,
                    ),
                    vertex_id=VertexId(vertex_index),
                    edge_id=edge_by_pair[
                        tuple(sorted((vertex_index, following)))
                    ],
                    uvs=(
                        LoopUV(
                            layer_name="SpineBakeUV",
                            coordinate=(
                                (position[0] + 1.0) / 3.0,
                                (position[1] + 1.0) / 2.0,
                            ),
                        ),
                    ),
                )
            )
            loop_ids.append(loop_id)
        faces.append(
            MeshFace(
                id=FaceId(face_index),
                source_id=SourceFaceId(_OBJECT_ID, face_index),
                loop_ids=tuple(loop_ids),
                material_index=0,
                normal=(0.0, 0.0, 1.0),
                smooth=True,
            )
        )

    snapshot = MeshSnapshot(
        snapshot_id="DepthRigSpaceSnapshot",
        source_object_id=_OBJECT_ID,
        object_name="Depth Rig Space",
        vertices=vertices,
        edges=edges,
        loops=tuple(loops),
        faces=tuple(faces),
        uv_layer_names=("SpineBakeUV",),
        active_uv_layer="SpineBakeUV",
        render_uv_layer="SpineBakeUV",
        world_matrix=_IDENTITY,
    )
    MeshSnapshotValidator().validate_or_raise(snapshot)
    return snapshot


def _frame() -> A1CameraProjectionFrame:
    return A1CameraProjectionFrame(
        camera_id="Camera",
        kind=A1CameraProjectionKind.ORTHOGRAPHIC,
        texture_width=100,
        texture_height=100,
        clip_start=0.1,
        clip_end=100.0,
        view_matrix=_IDENTITY,
        projection_matrix=_IDENTITY,
    )


def _projection_result(snapshot: MeshSnapshot) -> DepthCameraProjectionResult:
    camera_z = tuple(float(vertex.position[2]) for vertex in snapshot.vertices)
    return DepthCameraProjectionResult(
        snapshot=snapshot,
        frame=_frame(),
        projected_origin=A1ProjectedPoint(0.0, 0.0, -5.0),
        base_mode=DepthProjectionBaseMode.FARTHEST_VISIBLE,
        base_depth=min(camera_z),
        farthest_visible_depth=min(camera_z),
        nearest_visible_depth=max(camera_z),
        maximum_relief=max(camera_z) - min(camera_z),
        requested_spacing_pixels=4.0,
        resolved_spacing_x_pixels=4.0,
        resolved_spacing_y_pixels=4.0,
        source_triangle_count=2,
        sampled_point_count=len(snapshot.vertices),
    )


def test_camera_local_negative_z_becomes_positive_distance_without_topology_change():
    camera_snapshot = _snapshot((-2.0, -2.0, -4.0, -6.0, -6.0, -8.0))
    result = _projection_result(camera_snapshot)

    converted = convert_depth_result_to_camera_distance(result)

    assert tuple(vertex.position[2] for vertex in converted.snapshot.vertices) == (
        2.0,
        2.0,
        4.0,
        6.0,
        6.0,
        8.0,
    )
    assert converted.farthest_visible_depth == result.farthest_visible_depth
    assert converted.nearest_visible_depth == result.nearest_visible_depth
    assert converted.base_depth == result.base_depth
    assert converted.snapshot.faces == result.snapshot.faces
    assert converted.snapshot.loops == result.snapshot.loops
    assert converted.snapshot.edges == result.snapshot.edges


def test_two_objects_keep_distinct_ranges_from_one_camera_zero():
    near = convert_depth_result_to_camera_distance(
        _projection_result(_snapshot((-2.0, -2.0, -3.0, -3.5, -3.5, -4.0)))
    )
    far = convert_depth_result_to_camera_distance(
        _projection_result(_snapshot((-8.0, -8.0, -9.0, -9.5, -9.5, -10.0)))
    )

    near_range = tuple(vertex.position[2] for vertex in near.snapshot.vertices)
    far_range = tuple(vertex.position[2] for vertex in far.snapshot.vertices)
    assert max(near_range) < min(far_range)
    assert min(near_range) > 0.0
    assert min(far_range) > 0.0


def test_disconnected_depth_surface_builds_one_compensated_attachment():
    snapshot = _snapshot()
    z_groups = build_a1_z_group_assignment(snapshot)
    rig = build_legacy_rig(
        LegacyRigBuildRequest(
            prefix="Depth",
            texture_width=100,
            texture_height=100,
            z_groups=z_groups.groups,
            main_position_pixels=(15.0, -10.0),
            z_group_origin_mode=LegacyZGroupOriginMode.OBJECT_ORIGIN,
        )
    )
    projection = project_depth_camera_attachment(
        snapshot,
        rig,
        A1AttachmentProjectionSettings(
            slot_name="Depth_Segment_0",
            attachment_name="Depth_Segment_0",
            vertex_prefix="Depth_Segment_0",
            image_path="images/Depth_Baked",
            uv_layer_name="SpineBakeUV",
            attachment_width=100.0,
            attachment_height=100.0,
            center_x=0.0,
            center_y=0.0,
            z_bindings=z_groups.projection_bindings(snapshot),
        ),
    )

    assert projection.request.slot_name == "Depth_Segment_0"
    assert projection.request.attachment_name == "Depth_Segment_0"
    assert len(projection.request.vertices) == 6
    assert len(projection.request.triangles) == 6
    assert set(projection.request.triangles) == set(range(6))

    vertex_map = snapshot.vertex_by_id()
    offset_by_group = {
        group.index: group.y_offset_pixels for group in rig.info.z_groups
    }
    for attachment_vertex, key in zip(
        projection.request.vertices,
        projection.ordered_vertex_keys,
        strict=True,
    ):
        source = vertex_map[key.vertex_id]
        expected_x = float(source.position[0]) * rig.info.uniform_scale
        expected_y = -float(source.position[1]) * rig.info.uniform_scale
        reconstructed_y = (
            attachment_vertex.bone_position_pixels[1]
            + offset_by_group[attachment_vertex.z_group_index]
        )
        assert attachment_vertex.bone_position_pixels[0] == pytest.approx(expected_x)
        assert reconstructed_y == pytest.approx(expected_y)

    built = build_legacy_mesh_attachment(rig, projection.request)
    assert len(built.document.slots) == 1
    assert len(built.vertex_bones) == 6
    assert SpineValidator().validate(built.document) == ()


def test_camera_distance_conversion_rejects_points_on_or_behind_camera_plane():
    invalid = _snapshot((0.0, -2.0, -4.0, -6.0, -6.0, -8.0))
    with pytest.raises(ValueError, match="on or behind the camera plane"):
        convert_depth_result_to_camera_distance(_projection_result(invalid))
