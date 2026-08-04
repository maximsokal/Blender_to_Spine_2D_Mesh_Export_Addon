"""Regressions for local repair of camera-clipped depth polygons."""

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


_OBJECT_ID = "ClippedDepthObject"
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
        snapshot_id="ClippedDepthSource",
        source_object_id=_OBJECT_ID,
        object_name="Clipped Depth Object",
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


def _all_uvs(result: object) -> tuple[tuple[float, float], ...]:
    snapshot = result.snapshot
    return tuple(
        loop.uv(_UV_LAYER)
        for loop in snapshot.loops
        if loop.uv(_UV_LAYER) is not None
    )


def test_partial_triangle_is_repaired_locally_with_source_vertex_budget() -> None:
    source = _snapshot(
        (
            (-1.40, -0.50, 0.0),
            (0.50, -0.50, 0.2),
            (0.50, 0.50, 0.8),
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
    assert len(result.snapshot.vertices) == len(source.vertices)
    assert len(result.snapshot.faces) == 1
    uvs = _all_uvs(result)
    assert uvs
    assert all(0.0 <= value <= 1.0 for uv in uvs for value in uv)
    assert any(abs(uv[0]) <= 1.0e-9 for uv in uvs)


def test_only_frame_intersected_fan_faces_are_locally_retriangulated() -> None:
    source = _snapshot(
        (
            (-1.40, -0.60, 0.0),
            (0.40, -0.60, 0.2),
            (0.40, 0.60, 0.6),
            (-1.40, 0.60, 0.3),
            (0.00, 0.00, 1.0),
        ),
        (
            (0, 1, 4),
            (1, 2, 4),
            (2, 3, 4),
            (3, 0, 4),
        ),
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
    assert 3 <= first.sampled_point_count <= len(source.vertices)
    assert first.snapshot.faces
    MeshSnapshotValidator().validate_or_raise(first.snapshot)

    uvs = _all_uvs(first)
    assert all(0.0 <= value <= 1.0 for uv in uvs for value in uv)
    assert any(abs(uv[0]) <= 1.0e-9 for uv in uvs)
    assert any(abs(uv[0] - 0.70) <= 1.0e-9 for uv in uvs)
    assert any(abs(uv[0] - 0.50) <= 1.0e-9 for uv in uvs)
