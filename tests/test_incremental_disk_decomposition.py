from dataclasses import replace
from math import radians, sin, cos

import pytest

import Blender_to_Spine2D_Mesh_Exporter.domain.geometry.decomposition as decomposition_module
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
    A1AngularMode,
    DiskRegionState,
    DiskTopologyIndex,
    FaceId,
    RegionTopologyError,
    SegmentationSettings,
    analyse_face_region,
    build_edge_to_faces,
    build_face_adjacency,
    decompose_complex_segments,
    is_simple_disk,
    segment_mesh_a1,
)

from test_a1_segmentation_decomposition import (
    build_quad_ring,
    build_three_quad_strip,
)


def _normal_at_degrees(angle: float):
    value = radians(angle)
    return (-sin(value), 0.0, cos(value))


def _replace_face_normals(snapshot, angles):
    assert len(snapshot.faces) == len(angles)
    return replace(
        snapshot,
        faces=tuple(
            replace(face, normal=_normal_at_degrees(angle))
            for face, angle in zip(snapshot.faces, angles)
        ),
    )


def _segment_face_indices(plan):
    return tuple(
        tuple(face_id.index for face_id in segment.face_ids)
        for segment in plan.segments
    )


def _region_face_indices(plan):
    return tuple(
        tuple(face_id.index for face_id in region.face_ids)
        for region in plan.regions
    )


def test_default_angular_mode_is_exact_explicit_legacy_mode():
    snapshot = build_three_quad_strip()
    settings = SegmentationSettings(
        angle_limit_degrees=30.0,
        split_uv_boundaries=False,
    )

    implicit = segment_mesh_a1(snapshot, settings)
    explicit = segment_mesh_a1(
        snapshot,
        settings,
        angular_mode=A1AngularMode.LEGACY_SEED_CONE,
    )

    assert implicit == explicit
    assert _segment_face_indices(implicit) == ((0, 1), (2,))


def test_local_dihedral_mode_rejects_sharp_transition_inside_seed_cone():
    snapshot = _replace_face_normals(
        build_three_quad_strip(),
        (0.0, 25.0, -25.0),
    )
    settings = SegmentationSettings(
        angle_limit_degrees=30.0,
        split_uv_boundaries=False,
    )

    legacy = segment_mesh_a1(snapshot, settings)
    hybrid = segment_mesh_a1(
        snapshot,
        settings,
        angular_mode=A1AngularMode.SEED_CONE_AND_LOCAL_DIHEDRAL,
        local_angle_limit_degrees=30.0,
    )

    assert _segment_face_indices(legacy) == ((0, 1, 2),)
    assert _segment_face_indices(hybrid) == ((0, 1), (2,))


@pytest.mark.parametrize(
    "mode, local_limit, error_type",
    (
        ("UNKNOWN", None, ValueError),
        (A1AngularMode.SEED_CONE_AND_LOCAL_DIHEDRAL, -1.0, ValueError),
        (A1AngularMode.SEED_CONE_AND_LOCAL_DIHEDRAL, 181.0, ValueError),
        (A1AngularMode.SEED_CONE_AND_LOCAL_DIHEDRAL, "30", ValueError),
    ),
)
def test_invalid_angular_contract_is_rejected(mode, local_limit, error_type):
    with pytest.raises(error_type):
        segment_mesh_a1(
            build_three_quad_strip(),
            SegmentationSettings(split_uv_boundaries=False),
            angular_mode=mode,
            local_angle_limit_degrees=local_limit,
        )


def test_incremental_frontier_decisions_match_complete_topology_analysis():
    snapshot = build_quad_ring()
    edge_to_faces = build_edge_to_faces(snapshot)
    face_ids = tuple(face.id for face in snapshot.faces)
    adjacency = build_face_adjacency(
        snapshot,
        face_ids,
        edge_to_faces=edge_to_faces,
    )
    topology_index = DiskTopologyIndex(
        snapshot,
        edge_to_faces=edge_to_faces,
    )
    remaining = set(face_ids)
    seed = min(remaining, key=lambda item: item.index)
    state = DiskRegionState.from_face(
        snapshot,
        seed,
        topology_index=topology_index,
    )
    remaining.remove(seed)

    while True:
        candidates = sorted(
            {
                neighbour
                for face_id in state.face_ids
                for neighbour in adjacency[face_id]
                if neighbour in remaining
            },
            key=lambda item: item.index,
        )
        if not candidates:
            break
        accepted = None
        for candidate in candidates:
            addition = state.preview_add_face(candidate)
            complete = analyse_face_region(
                snapshot,
                set(state.face_ids) | {candidate},
                edge_to_faces=edge_to_faces,
            )
            assert (addition is not None) is is_simple_disk(complete)
            if accepted is None and addition is not None:
                accepted = addition
        if accepted is None:
            break
        state.apply_addition(accepted)
        remaining.remove(accepted.face_id)
        assert state.topology == analyse_face_region(
            snapshot,
            state.face_ids,
            edge_to_faces=edge_to_faces,
        )

    # The final face would close the ring and create a second boundary.
    assert len(remaining) == 1
    assert state.preview_add_face(next(iter(remaining))) is None


def test_incremental_ring_decomposition_preserves_previous_exact_partition():
    snapshot = build_quad_ring()
    segmentation = segment_mesh_a1(
        snapshot,
        SegmentationSettings(split_uv_boundaries=False),
    )

    first = decompose_complex_segments(snapshot, segmentation)
    second = decompose_complex_segments(snapshot, segmentation)

    assert first == second
    assert _region_face_indices(first) == (
        (0, 1, 2, 3, 4, 5, 7),
        (6,),
    )
    assert all(is_simple_disk(region.topology) for region in first.regions)


def test_complex_decomposition_uses_complete_analysis_only_for_input_and_outputs(
    monkeypatch,
):
    snapshot = build_quad_ring()
    segmentation = segment_mesh_a1(
        snapshot,
        SegmentationSettings(split_uv_boundaries=False),
    )
    real_analyse = decomposition_module.analyse_face_region
    calls = []

    def counted_analyse(*args, **kwargs):
        calls.append(tuple(sorted(face_id.index for face_id in args[1])))
        return real_analyse(*args, **kwargs)

    monkeypatch.setattr(
        decomposition_module,
        "analyse_face_region",
        counted_analyse,
    )

    plan = decomposition_module.decompose_complex_segments(
        snapshot,
        segmentation,
    )

    assert len(calls) == 1 + len(plan.regions)
    assert calls[0] == tuple(range(len(snapshot.faces)))


def test_stale_incremental_addition_cannot_be_applied_twice():
    snapshot = build_quad_ring()
    edge_to_faces = build_edge_to_faces(snapshot)
    state = DiskRegionState.from_face(
        snapshot,
        FaceId(0),
        topology_index=DiskTopologyIndex(
            snapshot,
            edge_to_faces=edge_to_faces,
        ),
    )
    addition = state.preview_add_face(FaceId(1))
    assert addition is not None

    state.apply_addition(addition)

    with pytest.raises(RegionTopologyError, match="stale"):
        state.apply_addition(addition)
