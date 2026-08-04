"""Generated parallax union identity regressions for real-scene face-index collisions."""

from __future__ import annotations

from Blender_to_Spine2D_Mesh_Exporter.domain.camera_projection import (
    A1CameraProjectionFrame,
    A1CameraProjectionKind,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
    EdgeId,
    FaceId,
    IDENTITY_MATRIX_4X4,
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
    DepthCameraProjectionResult,
    DepthProjectionBaseMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry.depth_parallax import (
    DepthParallaxCameraView,
    DepthParallaxGeometryPackage,
    DepthParallaxReserveSurface,
    DepthParallaxViewId,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry.depth_parallax_identity import (
    canonicalize_depth_parallax_package_identity,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.projection import A1ProjectedPoint


_OBJECT_ID = "FlowerShopBancoCollision"
_UV_LAYER = "SpineBakeUV"


def _frame() -> A1CameraProjectionFrame:
    return A1CameraProjectionFrame(
        camera_id="IdentityCamera",
        kind=A1CameraProjectionKind.ORTHOGRAPHIC,
        texture_width=64,
        texture_height=64,
        clip_start=0.1,
        clip_end=100.0,
        view_matrix=IDENTITY_MATRIX_4X4,
        projection_matrix=IDENTITY_MATRIX_4X4,
    )


def _triangle_snapshot(
    *,
    snapshot_id: str,
    material_index: int,
    vertex_offset: int,
) -> MeshSnapshot:
    positions = (
        (-1.0 + vertex_offset * 0.1, -1.0, -5.0),
        (0.0 + vertex_offset * 0.1, -1.0, -5.0),
        (-0.5 + vertex_offset * 0.1, 0.0, -5.0),
    )
    vertices = tuple(
        MeshVertex(
            id=VertexId(index),
            source_id=SourceVertexId(_OBJECT_ID, vertex_offset + index),
            position=position,
            normal=(0.0, 0.0, 1.0),
        )
        for index, position in enumerate(positions)
    )
    edges = (
        MeshEdge(
            id=EdgeId(0),
            source_id=None,
            vertex_ids=(VertexId(0), VertexId(1)),
        ),
        MeshEdge(
            id=EdgeId(1),
            source_id=None,
            vertex_ids=(VertexId(1), VertexId(2)),
        ),
        MeshEdge(
            id=EdgeId(2),
            source_id=None,
            vertex_ids=(VertexId(0), VertexId(2)),
        ),
    )
    loops = (
        MeshLoop(
            id=LoopId(0),
            source_id=SourceLoopId(_OBJECT_ID, 0, 0),
            vertex_id=VertexId(0),
            edge_id=EdgeId(0),
            uvs=(LoopUV(_UV_LAYER, (0.0, 0.0)),),
        ),
        MeshLoop(
            id=LoopId(1),
            source_id=SourceLoopId(_OBJECT_ID, 0, 1),
            vertex_id=VertexId(1),
            edge_id=EdgeId(1),
            uvs=(LoopUV(_UV_LAYER, (1.0, 0.0)),),
        ),
        MeshLoop(
            id=LoopId(2),
            source_id=SourceLoopId(_OBJECT_ID, 0, 2),
            vertex_id=VertexId(2),
            edge_id=EdgeId(2),
            uvs=(LoopUV(_UV_LAYER, (0.5, 1.0)),),
        ),
    )
    face = MeshFace(
        id=FaceId(0),
        source_id=SourceFaceId(_OBJECT_ID, 0),
        loop_ids=(LoopId(0), LoopId(1), LoopId(2)),
        material_index=material_index,
        normal=(0.0, 0.0, 1.0),
        smooth=True,
    )
    snapshot = MeshSnapshot(
        snapshot_id=snapshot_id,
        source_object_id=_OBJECT_ID,
        object_name=snapshot_id,
        vertices=vertices,
        edges=edges,
        loops=loops,
        faces=(face,),
        uv_layer_names=(_UV_LAYER,),
        active_uv_layer=_UV_LAYER,
        render_uv_layer=_UV_LAYER,
    )
    MeshSnapshotValidator().validate_or_raise(snapshot)
    return snapshot


def _collision_union(front: MeshSnapshot, reserve: MeshSnapshot) -> MeshSnapshot:
    vertices = tuple(
        MeshVertex(
            id=VertexId(index),
            source_id=vertex.source_id,
            position=vertex.position,
            normal=vertex.normal,
        )
        for index, vertex in enumerate((*front.vertices, *reserve.vertices))
    )
    edges = (
        MeshEdge(EdgeId(0), None, (VertexId(0), VertexId(1))),
        MeshEdge(EdgeId(1), None, (VertexId(1), VertexId(2))),
        MeshEdge(EdgeId(2), None, (VertexId(0), VertexId(2))),
        MeshEdge(EdgeId(3), None, (VertexId(3), VertexId(4))),
        MeshEdge(EdgeId(4), None, (VertexId(4), VertexId(5))),
        MeshEdge(EdgeId(5), None, (VertexId(3), VertexId(5))),
    )
    loops = (
        MeshLoop(
            LoopId(0),
            SourceLoopId(_OBJECT_ID, 0, 0),
            VertexId(0),
            EdgeId(0),
            front.loops[0].uvs,
        ),
        MeshLoop(
            LoopId(1),
            SourceLoopId(_OBJECT_ID, 0, 1),
            VertexId(1),
            EdgeId(1),
            front.loops[1].uvs,
        ),
        MeshLoop(
            LoopId(2),
            SourceLoopId(_OBJECT_ID, 0, 2),
            VertexId(2),
            EdgeId(2),
            front.loops[2].uvs,
        ),
        MeshLoop(
            LoopId(3),
            SourceLoopId(_OBJECT_ID, 0, 0),
            VertexId(3),
            EdgeId(3),
            reserve.loops[0].uvs,
        ),
        MeshLoop(
            LoopId(4),
            SourceLoopId(_OBJECT_ID, 0, 1),
            VertexId(4),
            EdgeId(4),
            reserve.loops[1].uvs,
        ),
        MeshLoop(
            LoopId(5),
            SourceLoopId(_OBJECT_ID, 0, 2),
            VertexId(5),
            EdgeId(5),
            reserve.loops[2].uvs,
        ),
    )
    faces = (
        MeshFace(
            FaceId(0),
            SourceFaceId(_OBJECT_ID, 0),
            (LoopId(0), LoopId(1), LoopId(2)),
            0,
            (0.0, 0.0, 1.0),
            True,
        ),
        MeshFace(
            FaceId(1),
            SourceFaceId(_OBJECT_ID, 0),
            (LoopId(3), LoopId(4), LoopId(5)),
            1,
            (0.0, 0.0, 1.0),
            True,
        ),
    )
    union = MeshSnapshot(
        snapshot_id="parallax-collision-union",
        source_object_id=_OBJECT_ID,
        object_name="Parallax Collision Union",
        vertices=vertices,
        edges=edges,
        loops=loops,
        faces=faces,
        uv_layer_names=(_UV_LAYER,),
        active_uv_layer=_UV_LAYER,
        render_uv_layer=_UV_LAYER,
    )
    MeshSnapshotValidator().validate_or_raise(union)
    return union


def _front_result(front: MeshSnapshot) -> DepthCameraProjectionResult:
    return DepthCameraProjectionResult(
        snapshot=front,
        frame=_frame(),
        projected_origin=A1ProjectedPoint(0.0, 0.0, -5.0),
        base_mode=DepthProjectionBaseMode.FARTHEST_VISIBLE,
        base_depth=-5.0,
        farthest_visible_depth=-5.0,
        nearest_visible_depth=-5.0,
        maximum_relief=0.0,
        requested_spacing_pixels=4.0,
        resolved_spacing_x_pixels=4.0,
        resolved_spacing_y_pixels=4.0,
        source_triangle_count=1,
        sampled_point_count=3,
    )


def _package() -> DepthParallaxGeometryPackage:
    frame = _frame()
    front = _triangle_snapshot(
        snapshot_id="front-collision",
        material_index=0,
        vertex_offset=0,
    )
    reserve = _triangle_snapshot(
        snapshot_id="reserve-collision",
        material_index=0,
        vertex_offset=3,
    )
    union = _collision_union(front, reserve)
    view = DepthParallaxCameraView(
        view_id=DepthParallaxViewId.RIGHT,
        yaw_radians=0.1,
        pitch_radians=0.0,
        frame=frame,
        camera_world_matrix=IDENTITY_MATRIX_4X4,
    )
    surface = DepthParallaxReserveSurface(
        view=view,
        snapshot=reserve,
        source_face_indices=(0,),
        maximum_accumulated_angle_radians=0.1,
    )
    return DepthParallaxGeometryPackage(
        front_result=_front_result(front),
        union_snapshot=union,
        front_snapshot=front,
        reserve_surfaces=(surface,),
        horizon_angle_radians=0.2,
        front_face_indices=(1,),
        reserve_face_indices=(0,),
    )


def _front_only_package() -> DepthParallaxGeometryPackage:
    front = _triangle_snapshot(
        snapshot_id="front-only-stable-identity",
        material_index=0,
        vertex_offset=11,
    )
    return DepthParallaxGeometryPackage(
        front_result=_front_result(front),
        union_snapshot=front,
        front_snapshot=front,
        reserve_surfaces=(),
        horizon_angle_radians=0.0,
        front_face_indices=(0,),
        reserve_face_indices=(),
    )


def test_front_only_package_preserves_exact_local_identity() -> None:
    package = _front_only_package()
    before_vertex_ids = tuple(vertex.id for vertex in package.union_snapshot.vertices)
    before_source_ids = tuple(
        vertex.source_id for vertex in package.union_snapshot.vertices
    )

    resolved = canonicalize_depth_parallax_package_identity(
        package,
        uv_layer_name=_UV_LAYER,
    )

    assert resolved is package
    assert resolved.union_snapshot is package.union_snapshot
    assert resolved.front_snapshot is package.front_snapshot
    assert resolved.front_result.snapshot is package.front_result.snapshot
    assert tuple(vertex.id for vertex in resolved.union_snapshot.vertices) == before_vertex_ids
    assert (
        tuple(vertex.source_id for vertex in resolved.union_snapshot.vertices)
        == before_source_ids
    )


def test_generated_front_and_reserve_face_index_collision_is_canonicalized() -> None:
    package = _package()
    assert len({face.source_id for face in package.union_snapshot.faces}) == 1
    assert len({loop.source_id for loop in package.union_snapshot.loops}) == 3

    resolved = canonicalize_depth_parallax_package_identity(
        package,
        uv_layer_name=_UV_LAYER,
    )

    assert tuple(
        face.source_id.face_index for face in resolved.union_snapshot.faces
    ) == (0, 1)
    assert len({loop.source_id for loop in resolved.union_snapshot.loops}) == 6
    assert resolved.front_result.snapshot == resolved.front_snapshot
    assert len(resolved.front_snapshot.faces) == 1
    assert len(resolved.reserve_surfaces) == 1
    assert len(resolved.reserve_surfaces[0].snapshot.faces) == 1
    assert resolved.reserve_surfaces[0].source_face_indices == (0,)

    union_vertex_ids = {vertex.source_id for vertex in resolved.union_snapshot.vertices}
    for subset in (
        resolved.front_snapshot,
        resolved.reserve_surfaces[0].snapshot,
    ):
        assert all(vertex.source_id in union_vertex_ids for vertex in subset.vertices)
