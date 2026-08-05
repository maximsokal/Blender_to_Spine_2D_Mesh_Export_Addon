"""Regress budgeted Depth envelopes for many tiny disconnected source islands."""

from __future__ import annotations

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
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry.depth_camera_projection import (
    DepthCameraProjectionError,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry.depth_camera_projection_component_envelope import (
    is_sparse_lattice_failure,
)


_OBJECT_ID = "SparseDisconnectedDepth"
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


def _disconnected_quad_snapshot(
    *,
    component_count: int,
) -> MeshSnapshot:
    if isinstance(component_count, bool) or not isinstance(component_count, int):
        raise TypeError("component_count must be int")
    if component_count < 1:
        raise ValueError("component_count must be positive")

    columns = 13
    rows = (component_count + columns - 1) // columns
    half_size = 7.5e-4
    vertices: list[MeshVertex] = []
    edge_specs: list[tuple[int, int, int]] = []
    loop_specs: list[tuple[int, int, int, int]] = []
    face_loop_indices: list[tuple[int, int, int]] = []

    for component_index in range(component_count):
        column = component_index % columns
        row = component_index // columns
        center_x = -0.78 + 1.56 * float(column) / float(columns - 1)
        center_y = (
            0.0
            if rows == 1
            else -0.52 + 1.04 * float(row) / float(rows - 1)
        )
        depth = float(component_index % 7) * 2.0e-4
        base_vertex = len(vertices)
        positions = (
            (center_x - half_size, center_y - half_size, depth),
            (center_x + half_size, center_y - half_size, depth),
            (center_x + half_size, center_y + half_size, depth + 1.0e-4),
            (center_x - half_size, center_y + half_size, depth + 1.0e-4),
        )
        for position in positions:
            vertex_index = len(vertices)
            vertices.append(
                MeshVertex(
                    id=VertexId(vertex_index),
                    source_id=SourceVertexId(_OBJECT_ID, vertex_index),
                    position=position,
                    normal=(0.0, 0.0, 1.0),
                )
            )

        component_faces = (
            (base_vertex + 0, base_vertex + 1, base_vertex + 2),
            (base_vertex + 0, base_vertex + 2, base_vertex + 3),
        )
        for face_vertices in component_faces:
            face_index = len(face_loop_indices)
            loop_indices: list[int] = []
            for corner_index, first in enumerate(face_vertices):
                second = face_vertices[(corner_index + 1) % 3]
                edge_specs.append((min(first, second), max(first, second), face_index))
                loop_specs.append((face_index, corner_index, first, len(edge_specs) - 1))
                loop_indices.append(len(loop_specs) - 1)
            face_loop_indices.append(tuple(loop_indices))

    unique_pairs = tuple(
        sorted({(first, second) for first, second, _face in edge_specs})
    )
    edge_id_by_pair = {
        pair: EdgeId(index) for index, pair in enumerate(unique_pairs)
    }
    edges = tuple(
        MeshEdge(
            id=edge_id_by_pair[pair],
            source_id=None,
            vertex_ids=(VertexId(pair[0]), VertexId(pair[1])),
        )
        for pair in unique_pairs
    )

    loops = tuple(
        MeshLoop(
            id=LoopId(loop_index),
            source_id=SourceLoopId(_OBJECT_ID, face_index, corner_index),
            vertex_id=VertexId(vertex_index),
            edge_id=edge_id_by_pair[
                (
                    min(vertex_index, face_loop_vertex),
                    max(vertex_index, face_loop_vertex),
                )
            ],
        )
        for loop_index, (face_index, corner_index, vertex_index, _edge_spec_index) in enumerate(
            loop_specs
        )
        for face_vertices in (
            tuple(
                loop_specs[index][2]
                for index in face_loop_indices[face_index]
            ),
        )
        for face_loop_vertex in (
            face_vertices[(corner_index + 1) % 3],
        )
    )
    faces = tuple(
        MeshFace(
            id=FaceId(face_index),
            source_id=SourceFaceId(_OBJECT_ID, face_index),
            loop_ids=tuple(LoopId(index) for index in loop_indices),
            material_index=0,
            normal=(0.0, 0.0, 1.0),
            smooth=False,
        )
        for face_index, loop_indices in enumerate(face_loop_indices)
    )
    snapshot = MeshSnapshot(
        snapshot_id="SparseDisconnectedDepthSource",
        source_object_id=_OBJECT_ID,
        object_name="Sparse Disconnected Depth",
        vertices=tuple(vertices),
        edges=edges,
        loops=loops,
        faces=faces,
        world_matrix=_TRANSLATED_IN_FRONT,
    )
    MeshSnapshotValidator().validate_or_raise(snapshot)
    return snapshot


def _settings(*, max_points: int = 128) -> DepthCameraProjectionSettings:
    return DepthCameraProjectionSettings(
        smoothing=0.0,
        edge_threshold_fraction=1.0,
        mesh_error_pixels=4.0,
        max_points=max_points,
    )


def test_public_projection_repairs_many_tiny_disconnected_islands_within_budget() -> None:
    source = _disconnected_quad_snapshot(component_count=130)
    source_before = source

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

    assert source == source_before
    assert len(source.vertices) == 520
    assert len(source.faces) == 260
    assert first == second
    assert first.snapshot == second.snapshot
    assert 4 <= first.sampled_point_count <= 128
    assert first.sampled_point_count % 4 == 0
    assert len(first.snapshot.faces) == first.sampled_point_count // 2
    assert all(len(face.loop_ids) == 3 for face in first.snapshot.faces)
    assert first.source_triangle_count == 260
    assert first.requested_spacing_pixels == 4.0
    assert first.sampled_point_count == len(first.snapshot.vertices)
    assert first.snapshot.active_uv_layer == _UV_LAYER
    assert first.snapshot.render_uv_layer == _UV_LAYER

    uvs = tuple(
        loop.uv(_UV_LAYER)
        for loop in first.snapshot.loops
    )
    assert all(uv is not None for uv in uvs)
    assert all(
        0.0 <= component <= 1.0
        for uv in uvs
        if uv is not None
        for component in uv
    )


def test_component_envelope_scales_down_to_one_four_point_cluster() -> None:
    source = _disconnected_quad_snapshot(component_count=130)

    result = build_depth_camera_projection_surface(
        source,
        _frame(),
        uniform_scale=128.0,
        uv_layer_name=_UV_LAYER,
        settings=_settings(max_points=4),
    )

    assert result.sampled_point_count == 4
    assert len(result.snapshot.vertices) == 4
    assert len(result.snapshot.faces) == 2
    assert result.sampled_point_count <= len(source.vertices)


def test_only_sparse_lattice_errors_are_recoverable() -> None:
    assert is_sparse_lattice_failure(
        DepthCameraProjectionError(
            "depth lattice did not intersect at least three visible points; "
            "reduce Depth Mesh Error or increase Max Depth Points"
        )
    )
    assert is_sparse_lattice_failure(
        DepthCameraProjectionError(
            "Depth Edge Threshold disconnected every sampled triangle and source "
            "topology could not prove local continuity"
        )
    )
    assert not is_sparse_lattice_failure(
        DepthCameraProjectionError(
            "all source triangles collapse in active-camera screen space"
        )
    )
    assert not is_sparse_lattice_failure(ValueError("unrelated"))
