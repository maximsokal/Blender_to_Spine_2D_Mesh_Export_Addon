from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
    DiskRegionState,
    DiskTopologyIndex,
    FaceId,
    build_edge_to_faces,
)

from test_a1_segmentation_decomposition import build_quad_ring


class _NoFullScanDict(dict):
    """Dict that permits local key access but rejects whole-state iteration."""

    def __iter__(self):
        raise AssertionError("incremental state performed a full mapping scan")

    def items(self):
        raise AssertionError("incremental state performed a full mapping scan")

    def keys(self):
        raise AssertionError("incremental state performed a full mapping scan")

    def values(self):
        raise AssertionError("incremental state performed a full mapping scan")


def test_topology_and_face_addition_do_not_scan_complete_region_state():
    snapshot = build_quad_ring()
    topology_index = DiskTopologyIndex(
        snapshot,
        edge_to_faces=build_edge_to_faces(snapshot),
    )
    state = DiskRegionState.from_face(
        snapshot,
        FaceId(0),
        topology_index=topology_index,
    )
    expected_initial = state.topology

    state._edge_face_counts = _NoFullScanDict(state._edge_face_counts)
    state._boundary_degrees = _NoFullScanDict(state._boundary_degrees)

    assert state.topology == expected_initial
    addition = state.preview_add_face(FaceId(1))
    assert addition is not None
    state.apply_addition(addition)

    assert state.topology == addition.topology
    assert state.face_ids == (FaceId(0), FaceId(1))
    assert state.minimum_face_index == 0
    assert state.maximum_face_index == 1


def test_ordered_face_ids_are_cached_until_state_changes():
    snapshot = build_quad_ring()
    state = DiskRegionState.from_face(
        snapshot,
        FaceId(0),
        topology_index=DiskTopologyIndex(
            snapshot,
            edge_to_faces=build_edge_to_faces(snapshot),
        ),
    )

    first = state.face_ids
    second = state.face_ids
    assert first is second

    addition = state.preview_add_face(FaceId(1))
    assert addition is not None
    state.apply_addition(addition)
    third = state.face_ids

    assert third == (FaceId(0), FaceId(1))
    assert third is state.face_ids
    assert third is not first
