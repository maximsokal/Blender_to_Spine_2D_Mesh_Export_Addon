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


def _view() -> DepthParallaxCameraView:
    return DepthParallaxCameraView(
        view_id=DepthParallaxViewId.RIGHT,
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


def test_proxy_records_reuse_front_vertices_and_do_not_increase_union_budget() -> None:
    front = _front_snapshot()
    records = _proxy_records_for_view(
        front,
        _face_geometry(),
        _view(),
        _frame(),
        A1ProjectedPoint(u=0.0, v=0.0, depth=-5.0),
        128.0,
    )

    front_positions = {vertex.position for vertex in front.vertices}
    front_source_ids = {vertex.source_id for vertex in front.vertices}
    assert records
    assert all(
        position in front_positions
        for record in records
        for position in record.positions
    )
    assert all(
        source_id in front_source_ids
        for record in records
        for source_id in record.source_vertex_ids
    )

    union = _snapshot_from_records(
        front,
        tuple(_front_records(front, _UV_LAYER)) + records,
        uv_layer_name=_UV_LAYER,
        snapshot_suffix="parallax-budget-proxy",
        preserve_source_vertex_ids=False,
    )
    assert len(union.vertices) == len(front.vertices)
    assert len(union.vertices) == 4
    assert {face.material_index for face in union.faces} == {0, 1}


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
