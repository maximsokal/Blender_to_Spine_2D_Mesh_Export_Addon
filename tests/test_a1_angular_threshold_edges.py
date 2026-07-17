from dataclasses import replace
from math import cos, radians, sin

from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
    A1AngularMode,
    SegmentationSettings,
    segment_mesh_a1,
)

from test_a1_segmentation_decomposition import build_three_quad_strip


def _normal_at_degrees(angle):
    value = radians(angle)
    return (-sin(value), 0.0, cos(value))


def _with_face_angles(snapshot, angles):
    return replace(
        snapshot,
        faces=tuple(
            replace(face, normal=_normal_at_degrees(angle))
            for face, angle in zip(snapshot.faces, angles)
        ),
    )


def _partition(plan):
    return tuple(
        tuple(face_id.index for face_id in segment.face_ids)
        for segment in plan.segments
    )


def test_zero_local_limit_rejects_even_coplanar_transitions():
    snapshot = _with_face_angles(build_three_quad_strip(), (0.0, 0.0, 0.0))

    plan = segment_mesh_a1(
        snapshot,
        SegmentationSettings(
            angle_limit_degrees=30.0,
            split_uv_boundaries=False,
        ),
        angular_mode=A1AngularMode.SEED_CONE_AND_LOCAL_DIHEDRAL,
        local_angle_limit_degrees=0.0,
    )

    assert _partition(plan) == ((0,), (1,), (2,))


def test_180_local_limit_remains_strict_for_opposite_normals():
    snapshot = _with_face_angles(build_three_quad_strip(), (0.0, 90.0, -90.0))
    settings = SegmentationSettings(
        angle_limit_degrees=180.0,
        split_uv_boundaries=False,
    )

    legacy = segment_mesh_a1(snapshot, settings)
    hybrid = segment_mesh_a1(
        snapshot,
        settings,
        angular_mode=A1AngularMode.SEED_CONE_AND_LOCAL_DIHEDRAL,
        local_angle_limit_degrees=180.0,
    )

    assert _partition(legacy) == ((0, 1, 2),)
    assert _partition(hybrid) == ((0, 1), (2,))


def test_missing_hybrid_local_limit_reuses_global_seed_limit():
    snapshot = _with_face_angles(build_three_quad_strip(), (0.0, 25.0, -25.0))

    implicit_local = segment_mesh_a1(
        snapshot,
        SegmentationSettings(
            angle_limit_degrees=30.0,
            split_uv_boundaries=False,
        ),
        angular_mode=A1AngularMode.SEED_CONE_AND_LOCAL_DIHEDRAL,
    )

    assert _partition(implicit_local) == ((0, 1), (2,))
