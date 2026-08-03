"""Pure contracts for the optimized visible Depth Camera Projection surface."""

from __future__ import annotations

from dataclasses import replace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.camera_projection import (
    A1CameraProjectionFrame,
    A1CameraProjectionKind,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
    DepthCameraProjectionError,
    DepthCameraProjectionSettings,
    DepthProjectionBaseMode,
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


_OBJECT_ID = "DepthObject"
_UV_LAYER = "SpineBakeUV"
_IDENTITY = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)


def _translation(z: float) -> tuple[float, ...]:
    return (
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, z,
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
    *,
    origin_z: float = -5.0,
    local_depths: tuple[float, float, float, float] = (0.0, 0.0, 2.0, 2.0),
) -> MeshSnapshot:
    positions = (
        (-0.5, -0.5, local_depths[0]),
        (0.5, -0.5, local_depths[1]),
        (0.5, 0.5, local_depths[2]),
        (-0.5, 0.5, local_depths[3]),
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
    edge_pairs = ((0, 1), (1, 2), (2, 3), (0, 3), (0, 2))
    edges = tuple(
        MeshEdge(
            id=EdgeId(index),
            source_id=None,
            vertex_ids=(VertexId(first), VertexId(second)),
        )
        for index, (first, second) in enumerate(edge_pairs)
    )
    edge_by_pair = {
        tuple(sorted(pair)): EdgeId(index)
        for index, pair in enumerate(edge_pairs)
    }
    face_vertices = ((0, 1, 2), (0, 2, 3))
    loops: list[MeshLoop] = []
    faces: list[MeshFace] = []
    for face_index, face in enumerate(face_vertices):
        loop_ids = []
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
                    edge_id=edge_by_pair[tuple(sorted((vertex_index, following)))],
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
        snapshot_id="DepthSource",
        source_object_id=_OBJECT_ID,
        object_name="Depth Object",
        vertices=vertices,
        edges=edges,
        loops=tuple(loops),
        faces=tuple(faces),
        world_matrix=_translation(origin_z),
    )
    MeshSnapshotValidator().validate_or_raise(snapshot)
    return snapshot


def _settings(**changes: object) -> DepthCameraProjectionSettings:
    return replace(
        DepthCameraProjectionSettings(
            smoothing=0.0,
            edge_threshold_fraction=1.0,
            mesh_error_pixels=16.0,
            max_points=25,
        ),
        **changes,
    )


def test_farthest_visible_base_builds_camera_facing_relief_only() -> None:
    result = build_depth_camera_projection_surface(
        _snapshot(),
        _frame(),
        uniform_scale=128.0,
        uv_layer_name=_UV_LAYER,
        settings=_settings(),
    )

    assert result.base_mode is DepthProjectionBaseMode.FARTHEST_VISIBLE
    assert result.base_depth == pytest.approx(result.farthest_visible_depth)
    assert result.farthest_visible_depth == pytest.approx(-5.0)
    assert result.nearest_visible_depth == pytest.approx(-3.0)
    assert result.maximum_relief == pytest.approx(2.0)
    assert 3 <= result.sampled_point_count <= 25
    assert result.sampled_point_count == len(result.snapshot.vertices)

    depths = tuple(vertex.position[2] for vertex in result.snapshot.vertices)
    assert min(depths) == pytest.approx(result.base_depth)
    assert max(depths) == pytest.approx(result.nearest_visible_depth)
    assert all(result.base_depth <= depth < 0.0 for depth in depths)
    assert all(depth - result.base_depth >= 0.0 for depth in depths)


def test_depth_surface_has_direct_full_frame_uv_and_valid_topology() -> None:
    result = build_depth_camera_projection_surface(
        _snapshot(),
        _frame(),
        uniform_scale=128.0,
        uv_layer_name=_UV_LAYER,
        settings=_settings(),
    )
    surface = result.snapshot

    MeshSnapshotValidator().validate_or_raise(surface)
    assert surface.active_uv_layer == _UV_LAYER
    assert surface.render_uv_layer == _UV_LAYER
    assert surface.uv_layer_names == (_UV_LAYER,)
    assert all(len(face.loop_ids) == 3 for face in surface.faces)
    assert all(
        0.0 <= component <= 1.0
        for loop in surface.loops
        for component in loop.uv(_UV_LAYER) or ()
    )
    used_vertices = {loop.vertex_id for loop in surface.loops}
    assert used_vertices == {vertex.id for vertex in surface.vertices}


def test_point_budget_is_hard_and_deterministic() -> None:
    settings = _settings(mesh_error_pixels=0.25, max_points=12)
    first = build_depth_camera_projection_surface(
        _snapshot(),
        _frame(),
        uniform_scale=128.0,
        uv_layer_name=_UV_LAYER,
        settings=settings,
    )
    second = build_depth_camera_projection_surface(
        _snapshot(),
        _frame(),
        uniform_scale=128.0,
        uv_layer_name=_UV_LAYER,
        settings=settings,
    )

    assert first.sampled_point_count <= 12
    assert first == second
    assert first.snapshot == second.snapshot


def test_edge_threshold_can_fail_closed_when_every_cell_crosses_depth_jump() -> None:
    source = _snapshot(local_depths=(0.0, 0.0, 4.0, 4.0))
    with pytest.raises(
        DepthCameraProjectionError,
        match="Depth Edge Threshold disconnected every sampled triangle",
    ):
        build_depth_camera_projection_surface(
            source,
            _frame(),
            uniform_scale=128.0,
            uv_layer_name=_UV_LAYER,
            settings=_settings(
                edge_threshold_fraction=0.0,
                mesh_error_pixels=64.0,
                max_points=4,
            ),
        )


def test_object_origin_base_accepts_origin_behind_visible_surface() -> None:
    result = build_depth_camera_projection_surface(
        _snapshot(origin_z=-5.0),
        _frame(),
        uniform_scale=128.0,
        uv_layer_name=_UV_LAYER,
        settings=_settings(base_mode=DepthProjectionBaseMode.OBJECT_ORIGIN),
    )

    assert result.base_mode is DepthProjectionBaseMode.OBJECT_ORIGIN
    assert result.base_depth == pytest.approx(-5.0)
    assert result.maximum_relief == pytest.approx(2.0)


def test_object_origin_base_rejects_visible_points_behind_origin() -> None:
    with pytest.raises(
        DepthCameraProjectionError,
        match="OBJECT_ORIGIN depth base lies in front",
    ):
        build_depth_camera_projection_surface(
            _snapshot(
                origin_z=-4.0,
                local_depths=(-1.0, -1.0, 1.0, 1.0),
            ),
            _frame(),
            uniform_scale=128.0,
            uv_layer_name=_UV_LAYER,
            settings=_settings(base_mode=DepthProjectionBaseMode.OBJECT_ORIGIN),
        )


@pytest.mark.parametrize(
    ("changes", "error"),
    (
        ({"smoothing": -0.1}, "smoothing must be in"),
        ({"edge_threshold_fraction": 1.1}, "edge_threshold_fraction must be in"),
        ({"mesh_error_pixels": 0.0}, "mesh_error_pixels must be positive"),
        ({"max_points": 3}, "max_points must be at least 4"),
        ({"max_points": 4097}, "max_points cannot exceed 4096"),
    ),
)
def test_depth_settings_reject_invalid_values(
    changes: dict[str, object],
    error: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        replace(DepthCameraProjectionSettings(), **changes)
