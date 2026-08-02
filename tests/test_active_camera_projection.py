"""Pure regressions for Normal / UV Segments active-camera projection."""

from __future__ import annotations

from math import isclose, sqrt

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.camera_projection import (
    A1CameraProjectionFrame,
    A1CameraProjectionKind,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
    EdgeId,
    FaceId,
    LoopId,
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
    calculate_a1_projected_snapshot_depth_range,
    project_a1_mesh_snapshot_camera,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.projection import A1ProjectionError


_OBJECT_ID = "CameraFixture"
_IDENTITY = (
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
)


def _translation_matrix(x: float, y: float, z: float) -> tuple[float, ...]:
    return (
        1.0,
        0.0,
        0.0,
        x,
        0.0,
        1.0,
        0.0,
        y,
        0.0,
        0.0,
        1.0,
        z,
        0.0,
        0.0,
        0.0,
        1.0,
    )


def _perspective_matrix(
    *,
    aspect_ratio: float,
    near: float = 0.1,
    far: float = 100.0,
) -> tuple[float, ...]:
    focal = 1.0
    return (
        focal / aspect_ratio,
        0.0,
        0.0,
        0.0,
        0.0,
        focal,
        0.0,
        0.0,
        0.0,
        0.0,
        (far + near) / (near - far),
        (2.0 * far * near) / (near - far),
        0.0,
        0.0,
        -1.0,
        0.0,
    )


def _orthographic_matrix(
    *,
    left: float = -2.0,
    right: float = 2.0,
    bottom: float = -1.0,
    top: float = 1.0,
    near: float = 0.1,
    far: float = 100.0,
) -> tuple[float, ...]:
    return (
        2.0 / (right - left),
        0.0,
        0.0,
        -(right + left) / (right - left),
        0.0,
        2.0 / (top - bottom),
        0.0,
        -(top + bottom) / (top - bottom),
        0.0,
        0.0,
        -2.0 / (far - near),
        -(far + near) / (far - near),
        0.0,
        0.0,
        0.0,
        1.0,
    )


def _perspective_frame(width: int = 200, height: int = 100) -> A1CameraProjectionFrame:
    return A1CameraProjectionFrame(
        camera_id="PerspectiveCamera",
        kind=A1CameraProjectionKind.PERSPECTIVE,
        texture_width=width,
        texture_height=height,
        clip_start=0.1,
        clip_end=100.0,
        view_matrix=_IDENTITY,
        projection_matrix=_perspective_matrix(
            aspect_ratio=float(width) / float(height)
        ),
    )


def _orthographic_frame() -> A1CameraProjectionFrame:
    return A1CameraProjectionFrame(
        camera_id="OrthographicCamera",
        kind=A1CameraProjectionKind.ORTHOGRAPHIC,
        texture_width=200,
        texture_height=100,
        clip_start=0.1,
        clip_end=100.0,
        view_matrix=_IDENTITY,
        projection_matrix=_orthographic_matrix(),
    )


def _snapshot(
    *,
    translation: tuple[float, float, float] = (0.0, 0.0, -5.0),
) -> MeshSnapshot:
    positions = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.5),
        (0.0, 1.0, -1.0),
    )
    normal = (0.0, 0.0, 1.0)
    vertices = tuple(
        MeshVertex(
            id=VertexId(index),
            source_id=SourceVertexId(_OBJECT_ID, index),
            position=position,
            normal=normal,
        )
        for index, position in enumerate(positions)
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
    loops = tuple(
        MeshLoop(
            id=LoopId(index),
            source_id=SourceLoopId(_OBJECT_ID, 0, index),
            vertex_id=VertexId(index),
            edge_id=EdgeId(index),
        )
        for index in range(3)
    )
    return MeshSnapshot(
        snapshot_id="camera-fixture",
        source_object_id=_OBJECT_ID,
        object_name=_OBJECT_ID,
        vertices=vertices,
        edges=edges,
        loops=loops,
        faces=(
            MeshFace(
                id=FaceId(0),
                source_id=SourceFaceId(_OBJECT_ID, 0),
                loop_ids=(LoopId(0), LoopId(1), LoopId(2)),
                material_index=0,
                normal=normal,
            ),
        ),
        world_matrix=_translation_matrix(*translation),
    )


def test_perspective_frame_projects_to_export_texture_pixels() -> None:
    frame = _perspective_frame(width=200, height=100)

    center = frame.project_world_point((0.0, 0.0, -2.0))
    right = frame.project_world_point((1.0, 0.0, -2.0))
    up = frame.project_world_point((0.0, 1.0, -2.0))

    assert center.canonical_position == (0.0, 0.0, -2.0)
    assert isclose(right.u, 25.0, abs_tol=1.0e-12)
    assert isclose(right.v, 0.0, abs_tol=1.0e-12)
    assert isclose(up.u, 0.0, abs_tol=1.0e-12)
    assert isclose(up.v, 25.0, abs_tol=1.0e-12)


def test_texture_aspect_ratio_changes_horizontal_projection() -> None:
    wide = _perspective_frame(width=200, height=100)
    square = _perspective_frame(width=100, height=100)

    wide_point = wide.project_world_point((1.0, 0.0, -2.0))
    square_point = square.project_world_point((1.0, 0.0, -2.0))

    assert isclose(wide_point.u, 25.0, abs_tol=1.0e-12)
    assert isclose(square_point.u, 25.0, abs_tol=1.0e-12)
    assert wide.texture_width != square.texture_width


def test_orthographic_frame_preserves_depth_independent_screen_scale() -> None:
    frame = _orthographic_frame()

    near = frame.project_world_point((1.0, 0.5, -2.0))
    far = frame.project_world_point((1.0, 0.5, -20.0))

    assert near.u == far.u == 50.0
    assert near.v == far.v == 25.0
    assert near.depth == -2.0
    assert far.depth == -20.0


def test_camera_projection_keeps_origin_and_attachment_pixels_consistent() -> None:
    frame = _perspective_frame()
    source = _snapshot(translation=(0.5, -0.25, -5.0))
    source_positions = tuple(vertex.position for vertex in source.vertices)
    scale = 100.0

    result = project_a1_mesh_snapshot_camera(
        source,
        frame,
        uniform_scale=scale,
    )

    projected = result.snapshot
    assert tuple(vertex.position for vertex in source.vertices) == source_positions
    assert result.projected_origin.u == projected.world_matrix[3] * scale
    assert result.projected_origin.v == projected.world_matrix[7] * scale
    assert projected.world_matrix[11] == 0.0

    for source_vertex, projected_vertex in zip(
        source.vertices,
        projected.vertices,
        strict=True,
    ):
        world_point = tuple(
            source.world_matrix[index] + source_vertex.position[axis]
            for axis, index in enumerate((3, 7, 11))
        )
        expected = frame.project_world_point(world_point)
        final_spine_x = result.projected_origin.u + projected_vertex.position[0] * scale
        final_spine_y = result.projected_origin.v - projected_vertex.position[1] * scale

        assert isclose(final_spine_x, expected.u, abs_tol=1.0e-10)
        assert isclose(final_spine_y, expected.v, abs_tol=1.0e-10)
        assert projected_vertex.position[2] == result.projected_origin.depth

    assert {vertex.position[2] for vertex in projected.vertices} == {
        result.projected_origin.depth
    }
    normal_length = sqrt(sum(value * value for value in projected.vertices[0].normal))
    assert isclose(normal_length, 1.0, abs_tol=1.0e-12)


def test_camera_depth_range_represents_one_rigid_object_layer() -> None:
    result = project_a1_mesh_snapshot_camera(
        _snapshot(translation=(0.0, 0.0, -5.0)),
        _perspective_frame(),
        uniform_scale=100.0,
    )

    depth_range = calculate_a1_projected_snapshot_depth_range(result.snapshot)

    assert depth_range.origin_depth == 0.0
    assert depth_range.nearest_vertex_id == VertexId(0)
    assert depth_range.nearest_vertex_depth == -5.0
    assert depth_range.farthest_vertex_id == VertexId(0)
    assert depth_range.farthest_vertex_depth == -5.0
    assert depth_range.depth_span == 0.0


def test_geometry_outside_camera_frame_is_allowed() -> None:
    point = _perspective_frame().project_world_point((100.0, 0.0, -5.0))

    assert point.u > 100.0


def test_origin_and_vertices_on_or_behind_near_plane_fail_closed() -> None:
    frame = _perspective_frame()

    with pytest.raises(A1ProjectionError, match="object_origin.*near plane"):
        project_a1_mesh_snapshot_camera(
            _snapshot(translation=(0.0, 0.0, -0.1)),
            frame,
            uniform_scale=100.0,
        )

    with pytest.raises(A1ProjectionError, match=r"vertex\[1\].*near plane"):
        project_a1_mesh_snapshot_camera(
            _snapshot(translation=(0.0, 0.0, -0.5)),
            frame,
            uniform_scale=100.0,
        )


def test_camera_frame_rejects_non_orthonormal_view_matrix() -> None:
    scaled_view = list(_IDENTITY)
    scaled_view[0] = 2.0

    with pytest.raises(ValueError, match="unit length"):
        A1CameraProjectionFrame(
            camera_id="Broken",
            kind=A1CameraProjectionKind.PERSPECTIVE,
            texture_width=100,
            texture_height=100,
            clip_start=0.1,
            clip_end=100.0,
            view_matrix=tuple(scaled_view),
            projection_matrix=_perspective_matrix(aspect_ratio=1.0),
        )
