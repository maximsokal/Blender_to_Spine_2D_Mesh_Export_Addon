"""Blender-independent contracts for the 0.90.0 angular parallax reserve."""

from __future__ import annotations

from dataclasses import replace
from math import radians

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.camera_projection import (
    A1CameraProjectionFrame,
    A1CameraProjectionKind,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
    DepthCameraProjectionError,
    DepthCameraProjectionSettings,
    DepthParallaxCameraView,
    DepthParallaxViewId,
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
    build_depth_parallax_geometry_package,
)


_OBJECT_ID = "ParallaxDepthObject"
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


def _frame(camera_id: str = "Camera") -> A1CameraProjectionFrame:
    return A1CameraProjectionFrame(
        camera_id=camera_id,
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
        snapshot_id="ParallaxDepthSource",
        source_object_id=_OBJECT_ID,
        object_name="Parallax Depth Object",
        vertices=vertices,
        edges=edges,
        loops=tuple(loops),
        faces=tuple(faces),
        world_matrix=_TRANSLATED_IN_FRONT,
    )
    MeshSnapshotValidator().validate_or_raise(snapshot)
    return snapshot


def _low_density_bent_flap() -> MeshSnapshot:
    return _snapshot(
        (
            (-0.5, -0.5, 0.0),
            (0.5, -0.5, 0.0),
            (0.5, 0.5, 0.0),
            (-0.5, 0.5, 0.0),
            (0.2, -0.5, -0.3),
            (0.2, 0.5, -0.3),
        ),
        (
            (0, 1, 2),
            (0, 2, 3),
            (1, 4, 5),
            (1, 5, 2),
        ),
    )


def _high_density_bent_flap() -> MeshSnapshot:
    return _snapshot(
        (
            (-0.5, -0.5, 0.0),
            (0.5, -0.5, 0.0),
            (0.5, 0.5, 0.0),
            (-0.5, 0.5, 0.0),
            (0.35, -0.5, -0.15),
            (0.35, 0.5, -0.15),
            (0.2, -0.5, -0.3),
            (0.2, 0.5, -0.3),
        ),
        (
            (0, 1, 2),
            (0, 2, 3),
            (1, 4, 5),
            (1, 5, 2),
            (4, 6, 7),
            (4, 7, 5),
        ),
    )


def _front_result(source: MeshSnapshot):
    return build_depth_camera_projection_surface(
        source,
        _frame(),
        uniform_scale=128.0,
        uv_layer_name=_UV_LAYER,
        settings=DepthCameraProjectionSettings(
            smoothing=0.0,
            edge_threshold_fraction=1.0,
            mesh_error_pixels=4.0,
            max_points=128,
        ),
    )


def _reserve_views() -> tuple[DepthParallaxCameraView, ...]:
    values = []
    for view_id in DepthParallaxViewId:
        values.append(
            DepthParallaxCameraView(
                view_id=view_id,
                yaw_radians=0.0,
                pitch_radians=0.0,
                frame=_frame(f"Camera:{view_id.value}"),
                camera_world_matrix=_IDENTITY,
                lens_scale=1.0,
            )
        )
    return tuple(values)


def _package(
    source: MeshSnapshot,
    angle_degrees: float,
    *,
    max_points: int = 128,
):
    return build_depth_parallax_geometry_package(
        source,
        _front_result(source),
        _reserve_views() if angle_degrees > 0.0 else (),
        uniform_scale=128.0,
        uv_layer_name=_UV_LAYER,
        horizon_angle_radians=radians(angle_degrees),
        max_points=max_points,
    )


def _all_uvs(snapshot: MeshSnapshot) -> tuple[tuple[float, float], ...]:
    return tuple(
        uv
        for loop in snapshot.loops
        for uv in (loop.uv(_UV_LAYER),)
        if uv is not None
    )


def test_zero_angle_preserves_front_only_attachment_contract() -> None:
    source = _low_density_bent_flap()
    front = _front_result(source)
    package = _package(source, 0.0)

    assert package.reserve_surfaces == ()
    assert package.reserve_face_indices == ()
    assert package.attachment_count == 1
    assert package.front_snapshot == package.union_snapshot
    assert len(package.union_snapshot.vertices) == len(front.snapshot.vertices)
    assert len(package.union_snapshot.faces) == len(front.snapshot.faces)
    assert {face.material_index for face in package.union_snapshot.faces} == {0}


def test_angle_below_hinge_does_not_retain_hidden_flap() -> None:
    package = _package(_low_density_bent_flap(), 40.0)

    assert package.reserve_face_indices == ()
    assert package.reserve_surfaces == ()
    assert package.attachment_count == 1


def test_angle_above_hinge_adds_hidden_flap_and_texture_attachment() -> None:
    source = _low_density_bent_flap()
    package = _package(source, 50.0)

    assert package.front_face_indices == (0, 1)
    assert package.reserve_face_indices == (2, 3)
    assert package.reserve_enabled
    assert package.attachment_count == 2
    assert len(package.reserve_surfaces) == 1
    assert len(package.union_snapshot.vertices) == len(source.vertices)
    assert len(package.union_snapshot.faces) == len(source.faces)
    assert len(package.union_snapshot.vertices) <= 128
    assert package.reserve_surfaces[0].source_face_indices == (2, 3)
    assert package.reserve_surfaces[0].view.view_id in set(DepthParallaxViewId)

    for snapshot in (
        package.union_snapshot,
        package.front_snapshot,
        package.reserve_surfaces[0].snapshot,
    ):
        MeshSnapshotValidator().validate_or_raise(snapshot)
        uvs = _all_uvs(snapshot)
        assert uvs
        assert all(0.0 <= value <= 1.0 for uv in uvs for value in uv)


def test_same_surface_angle_is_density_independent() -> None:
    low = _package(_low_density_bent_flap(), 50.0)
    high = _package(_high_density_bent_flap(), 50.0)

    assert low.reserve_face_indices == (2, 3)
    assert high.reserve_face_indices == (2, 3, 4, 5)
    assert len(low.reserve_surfaces) == len(high.reserve_surfaces) == 1
    low_angle = low.reserve_surfaces[0].maximum_accumulated_angle_radians
    high_angle = high.reserve_surfaces[0].maximum_accumulated_angle_radians
    assert high_angle == pytest.approx(low_angle, abs=1.0e-9)
    assert low_angle == pytest.approx(radians(45.0), abs=1.0e-9)


def test_combined_front_and_reserve_points_obey_max_depth_points() -> None:
    source = _low_density_bent_flap()

    with pytest.raises(
        DepthCameraProjectionError,
        match="Positive Parallax Horizon has no reserve point budget after FRONT",
    ) as error:
        _package(source, 50.0, max_points=5)

    message = str(error.value)
    assert "front_points=4" in message
    assert "max_points=5" in message


def test_parallax_geometry_package_is_deterministic() -> None:
    source = _high_density_bent_flap()

    first = _package(source, 50.0)
    second = _package(source, 50.0)

    assert first == second
    assert first.union_snapshot == second.union_snapshot
    assert first.reserve_surfaces == second.reserve_surfaces


def test_positive_angle_requires_complete_virtual_view_set() -> None:
    source = _low_density_bent_flap()
    views = _reserve_views()[:-1]

    with pytest.raises(ValueError, match="requires all eight reserve views"):
        build_depth_parallax_geometry_package(
            source,
            _front_result(source),
            views,
            uniform_scale=128.0,
            uv_layer_name=_UV_LAYER,
            horizon_angle_radians=radians(50.0),
            max_points=128,
        )


def test_settings_remain_immutable_across_front_and_reserve_builds() -> None:
    settings = DepthCameraProjectionSettings(
        smoothing=0.0,
        edge_threshold_fraction=1.0,
        mesh_error_pixels=4.0,
        max_points=128,
    )
    source = _low_density_bent_flap()
    front = build_depth_camera_projection_surface(
        source,
        _frame(),
        uniform_scale=128.0,
        uv_layer_name=_UV_LAYER,
        settings=settings,
    )
    original = replace(settings)

    build_depth_parallax_geometry_package(
        source,
        front,
        _reserve_views(),
        uniform_scale=128.0,
        uv_layer_name=_UV_LAYER,
        horizon_angle_radians=radians(50.0),
        max_points=settings.max_points,
    )

    assert settings == original
