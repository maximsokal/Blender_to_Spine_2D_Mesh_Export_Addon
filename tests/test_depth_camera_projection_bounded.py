"""Regression contracts for source-bounded Depth Camera Projection geometry."""

from __future__ import annotations

from dataclasses import replace

from Blender_to_Spine2D_Mesh_Exporter.domain.camera_projection import (
    A1CameraProjectionFrame,
    A1CameraProjectionKind,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
    DepthCameraProjectionSettings,
    EdgeId,
    FaceId,
    LoopId,
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
    build_depth_camera_projection_surface,
)


_OBJECT_ID = "BoundedDepthObject"
_UV_LAYER = "SpineBakeUV"
_IDENTITY = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)
_TRANSLATED_IN_FRONT = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, -5.0,
    0.0, 0.0, 0.0, 1.0,
)


def _frame() -> A1CameraProjectionFrame:
    return A1CameraProjectionFrame(
        camera_id="Camera",
        kind=A1CameraProjectionKind.ORTHOGRAPHIC,
        texture_width=128,
        texture_height=128,
        clip_start=0.1,
        clip_end=100.0,
        view_matrix=_IDENTITY,
        projection_matrix=_IDENTITY,
    )


def _snapshot(
    positions: tuple[tuple[float, float, float], ...],
    face_vertices: tuple[tuple[int, int, int], ...],
) -> MeshSnapshot:
    vertices = tuple(
        MeshVertex(
            id=VertexId(index),
            source_id=SourceVertexId(_OBJECT_ID, index),
            position=position,
            normal=(0.0, 0.0, 1.0),
        )
        for index, position in enumerate(positions)
    )
    edge_pairs = tuple(
        sorted(
            {
                tuple(sorted((face[index], face[(index + 1) % 3])))
                for face in face_vertices
                for index in range(3)
            }
        )
    )
    edge_id_by_pair = {
        pair: EdgeId(index) for index, pair in enumerate(edge_pairs)
    }
    edges = tuple(
        MeshEdge(
            id=edge_id_by_pair[pair],
            source_id=None,
            vertex_ids=(VertexId(pair[0]), VertexId(pair[1])),
        )
        for pair in edge_pairs
    )

    loops: list[MeshLoop] = []
    faces: list[MeshFace] = []
    for face_index, face in enumerate(face_vertices):
        loop_ids: list[LoopId] = []
        for corner_index, vertex_index in enumerate(face):
            following = face[(corner_index + 1) % 3]
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
                    edge_id=edge_id_by_pair[
                        tuple(sorted((vertex_index, following)))
                    ],
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
                smooth=False,
            )
        )

    snapshot = MeshSnapshot(
        snapshot_id="BoundedDepthSource",
        source_object_id=_OBJECT_ID,
        object_name="Bounded Depth Object",
        vertices=vertices,
        edges=edges,
        loops=tuple(loops),
        faces=tuple(faces),
        world_matrix=_TRANSLATED_IN_FRONT,
    )
    MeshSnapshotValidator().validate_or_raise(snapshot)
    return snapshot


def _settings(**changes: object) -> DepthCameraProjectionSettings:
    return replace(
        DepthCameraProjectionSettings(
            smoothing=0.0,
            edge_threshold_fraction=1.0,
            mesh_error_pixels=4.0,
            max_points=128,
        ),
        **changes,
    )


def test_low_poly_quad_keeps_source_count_and_exact_projected_silhouette() -> None:
    source = _snapshot(
        (
            (-0.5, -0.5, 0.0),
            (0.5, -0.5, 0.0),
            (0.5, 0.5, 2.0),
            (-0.5, 0.5, 2.0),
        ),
        ((0, 1, 2), (0, 2, 3)),
    )

    result = build_depth_camera_projection_surface(
        source,
        _frame(),
        uniform_scale=128.0,
        uv_layer_name=_UV_LAYER,
        settings=_settings(),
    )

    assert result.sampled_point_count == 4
    assert len(result.snapshot.vertices) == len(source.vertices)
    assert len(result.snapshot.faces) == 2
    assert result.sampled_point_count <= 128

    unique_uvs = {
        tuple(round(component, 6) for component in loop.uv(_UV_LAYER) or ())
        for loop in result.snapshot.loops
    }
    assert unique_uvs == {
        (0.25, 0.25),
        (0.25, 0.75),
        (0.75, 0.25),
        (0.75, 0.75),
    }


def test_single_triangle_uses_three_vertices_instead_of_forcing_a_grid() -> None:
    source = _snapshot(
        (
            (-0.5, -0.5, 0.0),
            (0.5, -0.5, 0.5),
            (0.0, 0.5, 1.0),
        ),
        ((0, 1, 2),),
    )

    result = build_depth_camera_projection_surface(
        source,
        _frame(),
        uniform_scale=128.0,
        uv_layer_name=_UV_LAYER,
        settings=_settings(),
    )

    assert result.sampled_point_count == 3
    assert len(result.snapshot.vertices) == 3
    assert len(result.snapshot.faces) == 1


def test_source_bounded_projection_is_deterministic() -> None:
    source = _snapshot(
        (
            (-0.5, -0.5, 0.0),
            (0.5, -0.5, 0.0),
            (0.5, 0.5, 2.0),
            (-0.5, 0.5, 2.0),
        ),
        ((0, 1, 2), (0, 2, 3)),
    )

    first = build_depth_camera_projection_surface(
        source,
        _frame(),
        uniform_scale=128.0,
        uv_layer_name=_UV_LAYER,
        settings=_settings(),
    )
    second = build_depth_camera_projection_surface(
        source,
        _frame(),
        uniform_scale=128.0,
        uv_layer_name=_UV_LAYER,
        settings=_settings(),
    )

    assert first == second
    assert first.snapshot == second.snapshot


def test_user_budget_still_limits_dense_source_below_original_count() -> None:
    positions = tuple(
        (x / 4.0 - 0.5, y / 4.0 - 0.5, float(y) * 0.1)
        for y in range(5)
        for x in range(5)
    )
    faces: list[tuple[int, int, int]] = []
    for y in range(4):
        for x in range(4):
            lower_left = y * 5 + x
            lower_right = lower_left + 1
            upper_left = lower_left + 5
            upper_right = upper_left + 1
            faces.append((lower_left, lower_right, upper_right))
            faces.append((lower_left, upper_right, upper_left))
    source = _snapshot(positions, tuple(faces))

    result = build_depth_camera_projection_surface(
        source,
        _frame(),
        uniform_scale=128.0,
        uv_layer_name=_UV_LAYER,
        settings=_settings(max_points=9, mesh_error_pixels=0.25),
    )

    assert 3 <= result.sampled_point_count <= 9
    assert result.sampled_point_count < len(source.vertices)
