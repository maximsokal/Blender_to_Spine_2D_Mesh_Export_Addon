"""Regress dense parallax budgeting without Blender API dependencies."""

from __future__ import annotations

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
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry.depth_camera_projection import (
    _ProjectedTriangle,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry.depth_camera_projection_visible_topology import (
    _ClipPoint,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry.depth_parallax import (
    DepthParallaxCameraView,
    DepthParallaxReserveSurface,
    DepthParallaxViewId,
    _FaceGeometry,
    _front_records,
    _snapshot_from_records,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry.depth_parallax_budgeted import (
    _build_screen_grid,
    _merge_view_assignments,
    _proxy_records_for_view,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry.depth_parallax_identity import (
    _evaluated_render_face_indices,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.projection import A1ProjectedPoint


_OBJECT_ID = "BudgetedParallax"
_UV_LAYER = "SpineBakeUV"
_IDENTITY = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
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


def _front_snapshot(*, snapshot_id: str = "BudgetedFront") -> MeshSnapshot:
    positions = (
        (-0.5, -0.5, -5.0),
        (0.5, -0.5, -5.0),
        (0.5, 0.5, -4.8),
        (-0.5, 0.5, -4.8),
    )
    faces_vertices = ((0, 1, 2), (0, 2, 3))
    edge_pairs = tuple(
        sorted(
            {
                tuple(sorted((face[index], face[(index + 1) % 3])))
                for face in faces_vertices
                for index in range(3)
            }
        )
    )
    edge_id_by_pair = {
        pair: EdgeId(index) for index, pair in enumerate(edge_pairs)
    }
    vertices = tuple(
        MeshVertex(
            id=VertexId(index),
            source_id=SourceVertexId(_OBJECT_ID, index),
            position=position,
            normal=(0.0, 0.0, 1.0),
        )
        for index, position in enumerate(positions)
    )
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
    for face_index, face_vertices in enumerate(faces_vertices):
        loop_ids: list[LoopId] = []
        for corner_index, vertex_index in enumerate(face_vertices):
            following = face_vertices[(corner_index + 1) % 3]
            loop_id = LoopId(len(loops))
            position = positions[vertex_index]
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
                    uvs=(
                        LoopUV(
                            layer_name=_UV_LAYER,
                            coordinate=(
                                position[0] + 0.5,
                                position[1] + 0.5,
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
        snapshot_id=snapshot_id,
        source_object_id=_OBJECT_ID,
        object_name="Budgeted Parallax",
        vertices=vertices,
        edges=edges,
        loops=tuple(loops),
        faces=tuple(faces),
        uv_layer_names=(_UV_LAYER,),
        active_uv_layer=_UV_LAYER,
        render_uv_layer=_UV_LAYER,
        world_matrix=_IDENTITY,
    )
    MeshSnapshotValidator().validate_or_raise(snapshot)
    return snapshot


def _view(
    view_id: DepthParallaxViewId = DepthParallaxViewId.RIGHT,
) -> DepthParallaxCameraView:
    return DepthParallaxCameraView(
        view_id=view_id,
        yaw_radians=0.2,
        pitch_radians=0.0,
        frame=_frame(),
        camera_world_matrix=_IDENTITY,
    )


def _face_geometry() -> tuple[_FaceGeometry, ...]:
    source_ids = tuple(SourceVertexId(_OBJECT_ID, index) for index in range(4, 8))
    return (
        _FaceGeometry(
            face_index=10,
            source_face_index=30,
            source_vertex_ids=(source_ids[0], source_ids[1], source_ids[2]),
            world_points=(
                (-0.45, -0.45, -5.0),
                (0.45, -0.45, -5.0),
                (0.45, 0.45, -4.7),
            ),
            normal_world=(0.0, -0.316227766, 0.948683298),
            centroid_world=(0.15, -0.15, -4.9),
        ),
        _FaceGeometry(
            face_index=11,
            source_face_index=31,
            source_vertex_ids=(source_ids[0], source_ids[2], source_ids[3]),
            world_points=(
                (-0.45, -0.45, -5.0),
                (0.45, 0.45, -4.7),
                (-0.45, 0.45, -4.7),
            ),
            normal_world=(0.0, -0.316227766, 0.948683298),
            centroid_world=(-0.15, 0.15, -4.8),
        ),
    )


def test_proxy_records_add_isolated_vertices_within_reserved_budget() -> None:
    front = _front_snapshot()
    records = _proxy_records_for_view(
        _face_geometry(),
        _view(),
        _frame(),
        A1ProjectedPoint(u=0.0, v=0.0, depth=-5.0),
        128.0,
        point_count=4,
        generated_source_vertex_base=4,
        source_object_id=_OBJECT_ID,
    )

    assert len(records) == 2
    proxy_source_ids = {
        source_id
        for record in records
        for source_id in record.source_vertex_ids
    }
    assert proxy_source_ids == {
        SourceVertexId(_OBJECT_ID, 4),
        SourceVertexId(_OBJECT_ID, 5),
        SourceVertexId(_OBJECT_ID, 6),
        SourceVertexId(_OBJECT_ID, 7),
    }
    assert proxy_source_ids.isdisjoint(
        {vertex.source_id for vertex in front.vertices}
    )

    union = _snapshot_from_records(
        front,
        tuple(_front_records(front, _UV_LAYER)) + records,
        uv_layer_name=_UV_LAYER,
        snapshot_suffix="parallax-budget-proxy",
        preserve_source_vertex_ids=False,
    )
    assert len(union.vertices) == len(front.vertices) + 4
    assert len(union.vertices) == 8
    assert {face.material_index for face in union.faces} == {0, 1}

    loops = union.loop_by_id()
    front_indices = {
        loops[loop_id].vertex_id.index
        for face in union.faces
        if face.material_index == 0
        for loop_id in face.loop_ids
    }
    reserve_indices = {
        loops[loop_id].vertex_id.index
        for face in union.faces
        if face.material_index == 1
        for loop_id in face.loop_ids
    }
    assert front_indices
    assert reserve_indices
    assert front_indices.isdisjoint(reserve_indices)


def test_three_point_proxy_uses_one_triangle() -> None:
    records = _proxy_records_for_view(
        _face_geometry(),
        _view(),
        _frame(),
        A1ProjectedPoint(u=0.0, v=0.0, depth=-5.0),
        128.0,
        point_count=3,
        generated_source_vertex_base=100,
        source_object_id=_OBJECT_ID,
    )

    assert len(records) == 1
    assert len(set(records[0].source_vertex_ids)) == 3
    assert len(set(records[0].positions)) == 3


def test_low_budget_view_assignments_merge_to_nearest_retained_direction() -> None:
    assigned = {
        DepthParallaxViewId.RIGHT: (1, 2, 3, 4),
        DepthParallaxViewId.UP: (5,),
        DepthParallaxViewId.LEFT: (6, 7, 8),
        DepthParallaxViewId.DOWN: (9,),
    }

    merged = _merge_view_assignments(assigned, maximum_view_count=2)

    assert tuple(merged) == (
        DepthParallaxViewId.RIGHT,
        DepthParallaxViewId.LEFT,
    )
    assert set(merged[DepthParallaxViewId.RIGHT]) == {1, 2, 3, 4, 5, 9}
    assert set(merged[DepthParallaxViewId.LEFT]) == {6, 7, 8}
    assert set().union(*(set(values) for values in merged.values())) == set(range(1, 10))


def test_budget_proxy_identity_uses_explicit_complete_render_ownership() -> None:
    proxy = _front_snapshot(
        snapshot_id="BudgetedFront:parallax-budget-proxy:parallax-right"
    )
    surface = DepthParallaxReserveSurface(
        view=_view(),
        snapshot=proxy,
        source_face_indices=(3, 7, 11, 19),
        maximum_accumulated_angle_radians=0.4,
    )

    assert _evaluated_render_face_indices(surface) == (3, 7, 11, 19)


def test_screen_grid_returns_local_candidates_for_dense_projected_faces() -> None:
    triangles: list[_ProjectedTriangle] = []
    polygons: dict[int, tuple[_ClipPoint, ...]] = {}
    face_index = 0
    for row in range(10):
        for column in range(10):
            minimum_x = -60.0 + column * 12.0
            minimum_y = -60.0 + row * 12.0
            triangle = _ProjectedTriangle(
                face_index=face_index,
                points=(
                    (minimum_x, minimum_y, -5.0),
                    (minimum_x + 8.0, minimum_y, -5.0),
                    (minimum_x, minimum_y + 8.0, -5.0),
                ),
            )
            triangles.append(triangle)
            polygons[face_index] = (
                _ClipPoint(minimum_x, minimum_y, -5.0, None),
                _ClipPoint(minimum_x + 8.0, minimum_y, -5.0, None),
                _ClipPoint(minimum_x, minimum_y + 8.0, -5.0, None),
            )
            face_index += 1

    grid = _build_screen_grid(tuple(triangles), polygons, _frame())
    candidates = grid.candidates(-56.0, -56.0)

    assert candidates
    assert len(candidates) < len(triangles)
    assert 0 in {triangle.face_index for triangle in candidates}
